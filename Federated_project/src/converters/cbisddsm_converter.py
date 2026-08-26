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
            dicom_info.csv
            meta.csv
        jpeg/
            <SeriesInstanceUID>/
                1-NNN.jpg
                ...

Images are JPEGs organised by Series UID. dicom_info.csv maps exact study/series
pairs to paths. The mass/calc CSV paths contain both identifiers.
"""

from pathlib import Path
import hashlib
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


def _study_series_from_csv_path(csv_path_value: str) -> tuple[str, str]:
    """Extract (StudyInstanceUID, SeriesInstanceUID) from a lesion CSV path."""
    parts = [part.strip() for part in str(csv_path_value).strip().replace("\\", "/").split("/")]
    if len(parts) < 4 or not parts[1] or not parts[2]:
        return "", ""
    return parts[1], parts[2]


def _jpeg_path_from_image_path(image_path: str, jpeg_base: Path):
    """Translate an exact dicom_info image_path to its local JPEG path."""
    parts = Path(str(image_path).strip().replace("\\", "/")).parts
    try:
        jpeg_index = parts.index("jpeg")
    except ValueError:
        return None
    suffix = parts[jpeg_index + 1:]
    return jpeg_base.joinpath(*suffix) if suffix else None


def _build_dicom_index(dicom_info_path: Path, jpeg_base: Path) -> dict:
    """Index dicom_info rows by exact (StudyInstanceUID, SeriesInstanceUID)."""
    df = pd.read_csv(dicom_info_path, keep_default_na=False)
    index = {}
    for _, row in df.iterrows():
        study_uid = str(row["StudyInstanceUID"]).strip()
        series_uid = str(row["SeriesInstanceUID"]).strip()
        if not study_uid or not series_uid:
            continue
        candidate = {
            "series_description": str(row["SeriesDescription"]).strip().lower(),
            "image_path": _jpeg_path_from_image_path(row["image_path"], jpeg_base),
        }
        index.setdefault((study_uid, series_uid), []).append(candidate)
    return index


def _resolve_full_image(dicom_index: dict, study_uid: str, series_uid: str):
    """Resolve one exact full-mammogram metadata row."""
    candidates = dicom_index.get((study_uid, series_uid), [])
    if not candidates:
        return "unresolved", None
    if len(candidates) != 1:
        return "ambiguous", None
    return "resolved", candidates[0]["image_path"]


def _resolve_roi_image(dicom_index: dict, study_uid: str, series_uid: str):
    """Resolve one exact non-cropped ROI-mask metadata row."""
    candidates = [
        candidate
        for candidate in dicom_index.get((study_uid, series_uid), [])
        if candidate["series_description"] != "cropped images"
    ]
    if not candidates:
        return "unresolved", None
    if len(candidates) != 1:
        return "ambiguous", None
    return "resolved", candidates[0]["image_path"]


def _output_identity(
    patient_id: str, laterality: str, view: str, study_uid: str, series_uid: str
) -> str:
    """Return a stable, compact identity for one physical mammogram."""
    digest = hashlib.sha1(f"{study_uid}|{series_uid}".encode("utf-8")).hexdigest()[:12]
    return f"{patient_id}_{laterality}_{view}_{digest}"


class CBISDDSMConverter(BaseConverter):
    """
    Converts CBIS-DDSM JPEG images + CSV annotations to standardized PNG + CSV.
    """

    def __init__(self, raw_path: str, client_output_path: str):
        super().__init__(raw_path, client_output_path, dataset_name="CBIS-DDSM")
        self.csv_dir  = Path(raw_path) / "csv"
        jpeg_base     = Path(raw_path) / "jpeg"
        dicom_info    = self.csv_dir / "dicom_info.csv"
        self.dicom_index = _build_dicom_index(dicom_info, jpeg_base)
        self.records = []
        self.saved_images = set()  # track already-saved mammograms to avoid duplicates
        self.audit = {
            "full_metadata_unresolved": 0,
            "full_metadata_ambiguous": 0,
            "full_image_missing": 0,
            "full_image_unreadable": 0,
            "roi_metadata_unresolved": 0,
            "roi_metadata_ambiguous": 0,
            "roi_image_missing": 0,
            "roi_image_unreadable": 0,
            "empty_or_invalid_roi": 0,
            "retained_annotations": 0,
        }

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
        print("CBIS-DDSM audit summary:")
        for key, value in self.audit.items():
            print(f"  {key}: {value}")

    def _process_csv(self, csv_path: Path, abnormality_type: str):
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
        total = len(df)
        for i, (_, row) in enumerate(df.iterrows()):
            if i % 50 == 0:
                print(f"  [{csv_path.name}] {i}/{total}")
            self._process_row(row, abnormality_type)

    def _process_row(self, row: pd.Series, abnormality_type: str):
        # ── Full mammogram ────────────────────────────────────────────────
        study_uid, series_uid = _study_series_from_csv_path(
            row.get("image_file_path", "")
        )
        full_status, img_path = _resolve_full_image(
            self.dicom_index, study_uid, series_uid
        )
        if full_status != "resolved":
            self.audit[f"full_metadata_{full_status}"] += 1
            print(
                f"  [CBIS-DDSM] Full metadata {full_status}: "
                f"study={study_uid} series={series_uid}"
            )
            return
        if img_path is None or not img_path.is_file():
            self.audit["full_image_missing"] += 1
            print(f"  [CBIS-DDSM] Full image missing: {img_path}")
            return

        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            self.audit["full_image_unreadable"] += 1
            print(f"  [CBIS-DDSM] Full image unreadable: {img_path}")
            return

        patient_id = str(row.get("patient_id", series_uid)).replace(" ", "_")
        view = str(row.get("image_view", row.get("view", ""))).strip()
        laterality = str(
            row.get("left_or_right_breast", row.get("side", ""))
        ).strip()
        image_identity = _output_identity(
            patient_id, laterality, view, study_uid, series_uid
        )
        img_name = f"{image_identity}.png"

        # ── Bounding box from ROI mask ────────────────────────────────────
        roi_study_uid, roi_series_uid = _study_series_from_csv_path(
            row.get("roi_mask_file_path", "")
        )
        roi_status, mask_path = _resolve_roi_image(
            self.dicom_index, roi_study_uid, roi_series_uid
        )
        if roi_status != "resolved":
            self.audit[f"roi_metadata_{roi_status}"] += 1
            print(
                f"  [CBIS-DDSM] ROI metadata {roi_status}: "
                f"study={roi_study_uid} series={roi_series_uid}"
            )
            return
        if mask_path is None or not mask_path.is_file():
            self.audit["roi_image_missing"] += 1
            print(f"  [CBIS-DDSM] ROI image missing: {mask_path}")
            return

        mask_arr = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_arr is None:
            self.audit["roi_image_unreadable"] += 1
            print(f"  [CBIS-DDSM] ROI image unreadable: {mask_path}")
            return

        if mask_arr.shape != image.shape:
            mask_arr = cv2.resize(
                mask_arr, (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        bbox = _bbox_from_mask(mask_arr)
        if bbox is None:
            self.audit["empty_or_invalid_roi"] += 1
            print(f"  [CBIS-DDSM] Empty or invalid ROI: {mask_path}")
            return
        bbox_xmin, bbox_ymin, bbox_w, bbox_h = bbox
        if (
            bbox_xmin < 0
            or bbox_ymin < 0
            or bbox_w <= 0
            or bbox_h <= 0
            or bbox_xmin + bbox_w > image.shape[1]
            or bbox_ymin + bbox_h > image.shape[0]
        ):
            self.audit["empty_or_invalid_roi"] += 1
            print(f"  [CBIS-DDSM] Empty or invalid ROI: {mask_path}")
            return

        if image_identity not in self.saved_images:
            save_png_img(image, self.images_dir, image_identity)
            self.saved_images.add(image_identity)

        self.records.append(normalize_row({
            "image_name": img_name,
            "bbox_xmin": bbox_xmin,
            "bbox_ymin": bbox_ymin,
            "bbox_width": bbox_w,
            "bbox_height": bbox_h,
            "dataset_name": "CBIS-DDSM",
        }))
        self.audit["retained_annotations"] += 1
