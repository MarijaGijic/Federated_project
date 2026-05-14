"""
CBIS-DDSM converter.

Expected raw layout (as downloaded from Kaggle / TCIA):
    raw_path/
        csv/
            mass_case_description_train_set.csv
            mass_case_description_test_set.csv
            calc_case_description_train_set.csv
            calc_case_description_test_set.csv
        <DICOM folders — any depth, converter will search recursively>

The mask DICOM (ROI_mask_file_path) is used to derive the bounding box
of each lesion relative to the full mammogram.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import cv2

from .base_converter import BaseConverter
from ..utils.dicom_utils import read_dicom_image
from ..utils.image_utils import save_png_img, save_annotations_csv
from ..utils.annotation_schema import CANONICAL_COLUMNS, normalize_row


def _bbox_from_mask(mask_arr: np.ndarray):
    """Return (xmin, ymin, width, height) of non-zero region, or None."""
    rows = np.any(mask_arr > 0, axis=1)
    cols = np.any(mask_arr > 0, axis=0)
    if not rows.any():
        return None
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return int(cmin), int(rmin), int(cmax - cmin + 1), int(rmax - rmin + 1)


def _find_dicom(base: Path, rel_path: str) -> Path | None:
    """
    Resolve a CSV path entry to an actual file.
    Tries direct join first; if that fails, searches recursively by filename.
    """
    direct = base / rel_path
    if direct.exists():
        return direct
    filename = Path(rel_path).name
    hits = list(base.rglob(filename))
    return hits[0] if hits else None


class CBISDDSMConverter(BaseConverter):
    """
    Converts CBIS-DDSM DICOM images + CSV annotations to standardized PNG + CSV.
    """

    def __init__(self, raw_path: str, client_output_path: str):
        super().__init__(raw_path, client_output_path, dataset_name="CBIS-DDSM")
        self.csv_dir = Path(raw_path) / "csv"
        self.records = []

    def convert(self):
        csv_files = [
            "mass_case_description_train_set.csv",
            "mass_case_description_test_set.csv",
            "calc_case_description_train_set.csv",
            "calc_case_description_test_set.csv",
        ]
        for fname in csv_files:
            path = self.csv_dir / fname
            if path.exists():
                abnormality_type = "mass" if "mass" in fname else "calcification"
                self._process_csv(path, abnormality_type)
            else:
                print(f"  [CBIS-DDSM] CSV not found, skipping: {path.name}")

        save_annotations_csv(self.records, self.annotations_file, columns=CANONICAL_COLUMNS)
        print(f"CBIS-DDSM: saved {len(self.records)} annotations → {self.annotations_file}")

    def _process_csv(self, csv_path: Path, abnormality_type: str):
        df = pd.read_csv(csv_path)
        # Normalise column names (some versions use spaces / capitals differently)
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

        for _, row in df.iterrows():
            self._process_row(row, abnormality_type)

    def _process_row(self, row: pd.Series, abnormality_type: str):
        base = self.raw_path

        # ── Full mammogram ────────────────────────────────────────────────
        img_rel  = str(row.get("image_file_path", "")).strip()
        img_dcm  = _find_dicom(base, img_rel)
        if img_dcm is None:
            print(f"  [CBIS-DDSM] Full image not found: {img_rel}")
            return

        image = read_dicom_image(img_dcm)
        if image is None:
            return

        # Unique filename: patient + side + view + abnormality_id
        patient_id    = str(row.get("patient_id", img_dcm.stem)).replace(" ", "_")
        abnormality_id = str(row.get("abnormality_id", "0"))
        prefix = f"{patient_id}_{abnormality_id}"
        img_name = f"{prefix}.png"

        save_png_img(image, self.images_dir, prefix)

        # ── Bounding box from mask DICOM ──────────────────────────────────
        bbox_xmin = bbox_ymin = bbox_w = bbox_h = np.nan
        mask_rel = str(row.get("roi_mask_file_path", "")).strip()
        mask_dcm = _find_dicom(base, mask_rel)
        if mask_dcm is not None:
            mask_arr = read_dicom_image(mask_dcm)
            if mask_arr is not None:
                # Scale mask to full image size if dimensions differ
                if mask_arr.shape != image.shape:
                    mask_arr = cv2.resize(
                        mask_arr, (image.shape[1], image.shape[0]),
                        interpolation=cv2.INTER_NEAREST
                    )
                bbox = _bbox_from_mask(mask_arr)
                if bbox:
                    bbox_xmin, bbox_ymin, bbox_w, bbox_h = bbox

        # ── Metadata ──────────────────────────────────────────────────────
        self.records.append(normalize_row({
            "image_name":     img_name,
            "bbox_xmin":      bbox_xmin,
            "bbox_ymin":      bbox_ymin,
            "bbox_width":     bbox_w,
            "bbox_height":    bbox_h,
            "lesion_type":    abnormality_type,
            "pathology":      str(row.get("pathology", "")).strip(),
            "bi_rads":        row.get("assessment", 0),
            "breast_density": row.get("breast_density", 0),
            "laterality":     str(row.get("side", "unknown")).strip(),
            "view":           str(row.get("view", "unknown")).strip(),
            "dataset_name":   "CBIS-DDSM",
        }))
