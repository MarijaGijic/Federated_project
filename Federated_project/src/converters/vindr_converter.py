"""
VinDr-Mammo converter.

Expected raw layout (as released on PhysioNet / Kaggle):
    raw_path/
        finding_annotations.csv
        breast_level_annotations.csv
        images/
            <study_id>/
                <series_id>/
                    <image_id>.dicom   (or .dcm)

finding_annotations.csv columns:
    study_id, series_id, image_id, height, width,
    finding_categories, finding_birads, xmin, ymin, xmax, ymax, split

breast_level_annotations.csv columns:
    study_id, series_id, image_id, breast_birads, breast_density, split

finding_categories is a free-text string such as "Mass" or
"Suspicious Calcification". "No Finding" rows represent normal images.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from .base_converter import BaseConverter
from ..utils.dicom_utils import read_dicom_image
from ..utils.image_utils import save_png_img, save_annotations_csv
from ..utils.annotation_schema import CANONICAL_COLUMNS, normalize_row


def _find_image(images_root: Path, study_id: str, series_id: str, image_id: str) -> Path | None:
    for ext in (".dicom", ".dcm", ".DICOM", ".DCM"):
        p = images_root / study_id / series_id / f"{image_id}{ext}"
        if p.exists():
            return p
    # Fallback: search anywhere under images_root by image_id filename
    hits = list(images_root.rglob(f"{image_id}.*"))
    return hits[0] if hits else None


class VinDrConverter(BaseConverter):
    """Converts VinDr-Mammo DICOM + CSV to standardized PNG + CSV."""

    def __init__(self, raw_path: str, client_output_path: str):
        super().__init__(raw_path, client_output_path, dataset_name="VinDr-Mammo")
        self.raw_path    = Path(raw_path)
        self.images_root = self.raw_path / "images"
        self.records     = []

    def convert(self):
        findings_path = self.raw_path / "finding_annotations.csv"
        breast_path   = self.raw_path / "breast_level_annotations.csv"

        if not findings_path.exists():
            raise FileNotFoundError(f"finding_annotations.csv not found in {self.raw_path}")

        findings = pd.read_csv(findings_path)

        # Build density / laterality lookup from breast-level file
        density_lookup: dict[str, int] = {}
        if breast_path.exists():
            breast_df = pd.read_csv(breast_path)
            for _, brow in breast_df.iterrows():
                key = str(brow["image_id"])
                try:
                    density_lookup[key] = int(float(brow.get("breast_density", 0) or 0))
                except (TypeError, ValueError):
                    density_lookup[key] = 0

        # VinDr encodes laterality in folder / series name (e.g. "L_CC")
        # We parse it from series_id if present; otherwise "unknown"
        processed_images: set[str] = set()

        for _, row in findings.iterrows():
            study_id  = str(row["study_id"])
            series_id = str(row["series_id"])
            image_id  = str(row["image_id"])
            key       = image_id

            # Load & save image only once per unique image_id
            if key not in processed_images:
                dcm = _find_image(self.images_root, study_id, series_id, image_id)
                if dcm is None:
                    print(f"  [VinDr] DICOM not found: {study_id}/{series_id}/{image_id}")
                    # Still record the annotation without saving an image
                else:
                    image = read_dicom_image(dcm)
                    if image is not None:
                        save_png_img(image, self.images_dir, key)
                processed_images.add(key)

            img_name = f"{key}.png"

            # Bounding box (VinDr already gives xmin/ymin/xmax/ymax)
            try:
                xmin = float(row["xmin"])
                ymin = float(row["ymin"])
                xmax = float(row["xmax"])
                ymax = float(row["ymax"])
                bbox_xmin  = xmin
                bbox_ymin  = ymin
                bbox_width  = xmax - xmin
                bbox_height = ymax - ymin
            except (TypeError, ValueError):
                bbox_xmin = bbox_ymin = bbox_width = bbox_height = np.nan

            # Laterality from series_id (e.g. contains "_L_" or "_R_")
            lat = "unknown"
            sid_upper = series_id.upper()
            if "_L_" in sid_upper or sid_upper.endswith("_L"):
                lat = "L"
            elif "_R_" in sid_upper or sid_upper.endswith("_R"):
                lat = "R"

            # View from series_id (CC or MLO)
            view = "unknown"
            if "CC" in sid_upper:
                view = "CC"
            elif "MLO" in sid_upper:
                view = "MLO"

            self.records.append(normalize_row({
                "image_name":     img_name,
                "bbox_xmin":      bbox_xmin,
                "bbox_ymin":      bbox_ymin,
                "bbox_width":     bbox_width,
                "bbox_height":    bbox_height,
                "lesion_type":    str(row.get("finding_categories", "No Finding")).strip(),
                "pathology":      None,          # derived from bi_rads in normalize_row
                "bi_rads":        row.get("finding_birads", 0),
                "breast_density": density_lookup.get(key, 0),
                "laterality":     lat,
                "view":           view,
                "dataset_name":   "VinDr-Mammo",
            }))

        save_annotations_csv(self.records, self.annotations_file, columns=CANONICAL_COLUMNS)
        print(f"VinDr-Mammo: saved {len(self.records)} annotations → {self.annotations_file}")
