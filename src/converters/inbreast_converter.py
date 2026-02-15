from pathlib import Path
import os
import pandas as pd
import numpy as np

from .base_converter import BaseConverter   
from ..utils.xml_utils import extract_rois_from_xml
from ..utils.dicom_utils import read_dicom_image
from ..utils.image_utils import save_png_img, save_annotations_csv
from ..utils.bbox_utils import points_to_bbox

class INbreastConverter(BaseConverter):

    def __init__(self, raw_path, client_output_path, excel_path):
        super().__init__(raw_path, client_output_path, dataset_name="INbreast")
        self.xml_path = Path(raw_path) / "AllXML"
        self.dicom_path = Path(raw_path) / "AllDICOMs"
        self.df = pd.read_excel(excel_path)
        self.records = []


    def convert(self):
        xml_files = list(self.xml_path.glob("*.xml"))
        dicom_files = list(self.dicom_path.glob("*.dcm"))

        # for xml_file in xml_files:
        #     self._process_single_case(xml_file)
        
        for dicom_file in dicom_files:

            prefix = dicom_file.stem.split('_')[0]
            xml_file = self.xml_path / f"{prefix}.xml"
            self._process_single_image(dicom_file, xml_file)
        
        save_annotations_csv(self.records, self.annotations_file)
        print(f"Saved {len(self.records)} annotations → {self.annotations_file}")

    
    def _process_single_case(self, dicom_file, xml_file):

        prefix = dicom_file.stem.split('_')[0]

        # matching = list(self.dicom_path.glob(f"{prefix}_*.dcm"))
        # if not matching:
        #     print(f"No DICOM for {prefix}")
        #     return
    
        # dicom_file = matching[0]

        # rois = extract_rois_from_xml(xml_file)
        # # if not rois:
        # #     return
        
        image = read_dicom_image(dicom_file)
        if image is None:
            return
        
        img_name = f"{prefix}.png"
        save_png_img(image, self.images_dir, prefix)
        
        # metadata from excel
        meta = self._read_metadata(prefix)

        # if no lesions -> negative sample
        if not xml_file.exists():

            self.records.append({
                "image_name": img_name,
                "bbox_xmin": np.nan,
                "bbox_ymin": np.nan,
                "bbox_width": np.nan,
                "bbox_height": np.nan,
                "lesion_type": "normal",
                "bi_rads": meta["bi_rads"],
                "density": meta["density"],
                "dataset_name": "INbreast"
            })
            return
        
        rois = extract_rois_from_xml(xml_file)

        if not rois:
            return
        
        # one annotation per roi
        for roi in rois:
            bbox = points_to_bbox(roi["points"])
            if bbox is None:
                continue

            record = {
                "image_name": img_name,
                "bbox_xmin": bbox[0],
                "bbox_ymin": bbox[1],
                "bbox_width": bbox[2],
                "bbox_height": bbox[3],
                "lesion_type": roi.get("lesion_type", meta["lesion_type"]),
                "bi_rads": meta["bi_rads"],
                "density": meta["density"],
                "dataset_name": "INbreast"
            }
            
            self.records.append(record)
        

    def _read_metadata(self, prefix):
        row = self.df[self.df['File Name'] == float(prefix)]

        if row.empty:
            return {"bi_rads": None, "density": None, "lesion_type": None}

        row = row.iloc[0]

        lesion_types = []
        for t in ['Mass ', 'Micros', 'Distortion', 'Asymmetry']:
            if pd.notna(row.get(t, None)):
                lesion_types.append(t.strip())

        return {
            "bi_rads": row.get('Bi-Rads', None),
            "density": row.get('ACR', None),
            "lesion_type": ",".join(lesion_types) if lesion_types else None
        }


