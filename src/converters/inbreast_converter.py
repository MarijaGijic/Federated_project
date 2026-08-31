from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd

from .base_converter import BaseConverter
from ..utils.xml_utils import extract_rois_from_xml
from ..utils.dicom_utils import read_dicom_image
from ..utils.image_utils import save_png_img, save_annotations_csv
from ..utils.bbox_utils import (
    clip_bbox_to_image,
    points_to_bbox,
    point_to_resized_bbox,
)
from ..utils.annotation_schema import (
    CANONICAL_COLUMNS, normalize_row
)


class INbreastConverter(BaseConverter):

    ROI_NAME_MAP = {
        "cluster": "cluster",
        "mass": "mass",
        "spiculated region": "spiculated_region",
        "espiculated region": "spiculated_region",
        "asymmetry": "asymmetry",
        "assymetry": "asymmetry",
        "distortion": "distortion",
        "calcification": "calcification",
        "calcifications": "calcification",
    }

    def __init__(self, raw_path, client_output_path, excel_path, target_size=256):
        super().__init__(raw_path, client_output_path, dataset_name="INbreast")
        self.xml_path   = Path(raw_path) / "AllXML"
        self.dicom_path = Path(raw_path) / "AllDICOMs"
        self.df         = pd.read_excel(excel_path)
        if target_size <= 0:
            raise ValueError("target_size must be positive")
        self.target_size = target_size
        self.records = []
        self.audit = Counter({
            "retained_contour": 0,
            "expanded_calcification_point": 0,
            "skipped_unknown": 0,
            "rejected_malformed": 0,
            "annotated_but_no_valid_roi": 0,
        })

    def convert(self):
        dicom_files = list(self.dicom_path.glob("*.dcm"))
        for dicom_file in dicom_files:
            prefix   = dicom_file.stem.split("_")[0]
            xml_file = self.xml_path / f"{prefix}.xml"
            self._process_single_case(dicom_file, xml_file)

        save_annotations_csv(self.records, self.annotations_file, columns=CANONICAL_COLUMNS)
        print(f"INbreast: saved {len(self.records)} annotations → {self.annotations_file}")
        print("INbreast ROI audit:")
        for key in (
            "retained_contour",
            "expanded_calcification_point",
            "skipped_unknown",
            "rejected_malformed",
            "annotated_but_no_valid_roi",
        ):
            print(f"  {key:<34} {self.audit[key]}")

    def _process_single_case(self, dicom_file, xml_file):
        prefix = dicom_file.stem.split("_")[0]

        image = read_dicom_image(dicom_file)
        if image is None:
            return

        img_name = f"{prefix}.png"
        save_png_img(image, self.images_dir, prefix)
        meta = self._read_metadata(prefix)

        if not xml_file.exists():
            self.records.append(normalize_row({
                "image_name":  img_name,
                "bbox_xmin":   np.nan,
                "bbox_ymin":   np.nan,
                "bbox_width":  np.nan,
                "bbox_height": np.nan,
                "dataset_name": "INbreast",
            }))
            return

        rois = extract_rois_from_xml(xml_file)
        if not rois:
            # XML present but empty or unparseable — treat as no finding
            self.records.append(normalize_row({
                "image_name":  img_name,
                "bbox_xmin":   np.nan,
                "bbox_ymin":   np.nan,
                "bbox_width":  np.nan,
                "bbox_height": np.nan,
                "dataset_name": "INbreast",
            }))
            return

        records_before = len(self.records)
        for roi in rois:
            roi_name = self._normalize_roi_name(roi.get("name"))
            points = roi.get("points", [])

            if roi_name == "unknown":
                self.audit["skipped_unknown"] += 1
                print(
                    f"  [INbreast] {prefix} ROI {roi.get('index_in_image')}: "
                    f"skipped unknown name={roi.get('name')!r} type={roi.get('type')}"
                )
                continue

            if roi_name == "calcification" and len(points) == 1:
                bbox = point_to_resized_bbox(
                    points[0], image.shape[1], image.shape[0], self.target_size
                )
                retained_audit_key = "expanded_calcification_point"
            else:
                bbox = points_to_bbox(points) if len(points) >= 2 else None
                retained_audit_key = "retained_contour"

            if bbox is not None:
                bbox = clip_bbox_to_image(bbox, image.shape[1], image.shape[0])

            if bbox is None:
                self.audit["rejected_malformed"] += 1
                print(
                    f"  [INbreast] {prefix} ROI {roi.get('index_in_image')}: "
                    f"rejected malformed {roi_name} bbox"
                )
                continue

            self._append_positive(img_name, bbox)
            self.audit[retained_audit_key] += 1

        if len(self.records) == records_before:
            self.audit["annotated_but_no_valid_roi"] += 1
            print(f"  [INbreast] {prefix}: XML has ROIs but none can be represented safely")

    @classmethod
    def _normalize_roi_name(cls, name):
        key = " ".join((name or "").strip().lower().split())
        return cls.ROI_NAME_MAP.get(key, "unknown")

    def _append_positive(self, image_name, bbox):
        self.records.append(normalize_row({
            "image_name": image_name,
            "bbox_xmin": float(bbox[0]),
            "bbox_ymin": float(bbox[1]),
            "bbox_width": float(bbox[2]),
            "bbox_height": float(bbox[3]),
            "dataset_name": "INbreast",
        }))

    def _read_metadata(self, prefix) -> dict:
        row = self.df[self.df["File Name"] == float(prefix)]
        if row.empty:
            return {"lesion_type": None}
        row = row.iloc[0]

        lesion_types = []
        for col in ["Mass ", "Micros", "Distortion", "Asymmetry"]:
            if pd.notna(row.get(col, None)):
                lesion_types.append(col.strip())

        return {
            "lesion_type": lesion_types[0] if lesion_types else None,
        }
