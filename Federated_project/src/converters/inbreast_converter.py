from pathlib import Path
import numpy as np
import pandas as pd

from .base_converter import BaseConverter
from ..utils.xml_utils import extract_rois_from_xml
from ..utils.dicom_utils import read_dicom_image
from ..utils.image_utils import save_png_img, save_annotations_csv
from ..utils.bbox_utils import points_to_bbox
from ..utils.annotation_schema import (
    CANONICAL_COLUMNS, LESION_TYPE_MAP, normalize_row
)


class INbreastConverter(BaseConverter):

    def __init__(self, raw_path, client_output_path, excel_path):
        super().__init__(raw_path, client_output_path, dataset_name="INbreast")
        self.xml_path   = Path(raw_path) / "AllXML"
        self.dicom_path = Path(raw_path) / "AllDICOMs"
        self.df         = pd.read_excel(excel_path)
        self.records    = []

    def convert(self):
        dicom_files = list(self.dicom_path.glob("*.dcm"))
        for dicom_file in dicom_files:
            prefix   = dicom_file.stem.split("_")[0]
            xml_file = self.xml_path / f"{prefix}.xml"
            self._process_single_case(dicom_file, xml_file)

        save_annotations_csv(self.records, self.annotations_file, columns=CANONICAL_COLUMNS)
        print(f"INbreast: saved {len(self.records)} annotations → {self.annotations_file}")

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
                "lesion_type": "normal",
                "pathology":   "normal",
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
                "lesion_type": "normal",
                "pathology":   "normal",
                "dataset_name": "INbreast",
            }))
            return

        for roi in rois:
            bbox = points_to_bbox(roi["points"])
            if bbox is None:
                continue
            self.records.append(normalize_row({
                "image_name":  img_name,
                "bbox_xmin":   float(bbox[0]),
                "bbox_ymin":   float(bbox[1]),
                "bbox_width":  float(bbox[2]),
                "bbox_height": float(bbox[3]),
                "lesion_type": roi.get("lesion_type", meta["lesion_type"]),
                "pathology":   "unknown",
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
