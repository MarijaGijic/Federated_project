"""
CBIS-DDSM converter.

Expected raw layout (Kaggle dataset: awsaf49/cbis-ddsm-breast-cancer-image-dataset):
    raw_path/
        csv/
            dicom_info.csv
            mass_case_description_train_set.csv
            mass_case_description_test_set.csv
            calc_case_description_train_set.csv
            calc_case_description_test_set.csv
        jpeg/
            <SeriesInstanceUID>/
                1-NNN.jpg
                ...

Images are JPEGs organised by Series UID. dicom_info.csv maps UIDs to paths.
The mass/calc CSVs reference UIDs in their image_file_path column (index 1 when
split by "/"), which are looked up in the dicom_info mapping.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import cv2

from .base_converter import BaseConverter
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


def _uid_from_csv_path(csv_path_value: str) -> str:
    """Extract the Series UID from a CSV image_file_path entry."""
    parts = str(csv_path_value).strip().split("/")
    # path format: <Case_folder>/<StudyUID>/<SeriesUID>/<filename>
    return parts[2] if len(parts) >= 3 else ""


def _build_uid_map(dicom_info_path: Path, jpeg_base: Path, series_desc: str) -> dict:
    """Return {uid: absolute_jpeg_path} for a given SeriesDescription."""
    df = pd.read_csv(dicom_info_path)
    subset = df[df["SeriesDescription"] == series_desc]
    uid_map = {}
    for _, row in subset.iterrows():
        uid = str(row["SeriesInstanceUID"]).strip()
        # image_path in CSV is like CBIS-DDSM/jpeg/<uid>/<file>.jpg
        rel = str(row["image_path"]).strip()
        filename = Path(rel).name
        abs_path = jpeg_base / uid / filename
        if abs_path.exists():
            uid_map[uid] = abs_path
        else:
            # fall back to first jpg in that uid folder
            uid_dir = jpeg_base / uid
            if uid_dir.is_dir():
                hits = list(uid_dir.glob("*.jpg")) or list(uid_dir.glob("*.jpeg"))
                if hits:
                    uid_map[uid] = hits[0]
    return uid_map


class CBISDDSMConverter(BaseConverter):
    """
    Converts CBIS-DDSM JPEG images + CSV annotations to standardized PNG + CSV.
    """

    def __init__(self, raw_path: str, client_output_path: str):
        super().__init__(raw_path, client_output_path, dataset_name="CBIS-DDSM")
        self.csv_dir  = Path(raw_path) / "csv"
        jpeg_base     = Path(raw_path) / "jpeg"
        dicom_info    = self.csv_dir / "dicom_info.csv"
        self.full_mammo_map = _build_uid_map(dicom_info, jpeg_base, "full mammogram images")
        self.roi_map        = _build_uid_map(dicom_info, jpeg_base, "ROI mask images")
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
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
        for _, row in df.iterrows():
            self._process_row(row, abnormality_type)

    def _process_row(self, row: pd.Series, abnormality_type: str):
        # ── Full mammogram ────────────────────────────────────────────────
        img_uid = _uid_from_csv_path(row.get("image_file_path", ""))
        img_path = self.full_mammo_map.get(img_uid)
        if img_path is None:
            print(f"  [CBIS-DDSM] Full image not found for UID: {img_uid}")
            return

        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return

        patient_id     = str(row.get("patient_id", img_uid)).replace(" ", "_")
        abnormality_id = str(row.get("abnormality_id", "0"))
        prefix   = f"{patient_id}_{abnormality_id}"
        img_name = f"{prefix}.png"
        save_png_img(image, self.images_dir, prefix)

        # ── Bounding box from ROI mask ────────────────────────────────────
        bbox_xmin = bbox_ymin = bbox_w = bbox_h = np.nan
        mask_uid  = _uid_from_csv_path(row.get("roi_mask_file_path", ""))
        mask_path = self.roi_map.get(mask_uid)
        if mask_path is not None:
            mask_arr = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask_arr is not None:
                if mask_arr.shape != image.shape:
                    mask_arr = cv2.resize(
                        mask_arr, (image.shape[1], image.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
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
            "laterality":     str(row.get("left_or_right_breast", row.get("side", "unknown"))).strip(),
            "view":           str(row.get("image_view", row.get("view", "unknown"))).strip(),
            "dataset_name":   "CBIS-DDSM",
        }))
