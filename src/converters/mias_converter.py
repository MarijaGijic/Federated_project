"""
MIAS converter.

Expected raw layout:
    raw_path/
        all-mias.txt      (annotation file)
        *.pgm             (1024x1024 grayscale images)

all-mias.txt format (space-separated, '#' comment lines ignored):
    REFNUM  TISSUE  CLASS  SEVERITY  X   Y   RADIUS
    mdb001  G       CIRC   B         535 425 197
    mdb003  D       NORM
    ...

- TISSUE:   F (fatty), G (fatty-glandular), D (dense)
- CLASS:    CALC, CIRC, SPIC, MISC, ARCH, ASYM, NORM
- SEVERITY: B (benign), M (malignant) — absent for NORM
- X, Y:     center of lesion in pixels (origin = top-left)
- RADIUS:   approximate radius in pixels

Laterality is NOT encoded in MIAS → set to "unknown".
BI-RADS is NOT in MIAS → set to 0 (unknown).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import cv2

from .base_converter import BaseConverter
from ..utils.image_utils import save_png_img, save_annotations_csv
from ..utils.annotation_schema import CANONICAL_COLUMNS, normalize_row


def _parse_mias_txt(txt_path: Path) -> pd.DataFrame:
    records = []
    with open(txt_path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or line.lower().startswith("refnum"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue

            refnum  = parts[0]
            tissue  = parts[1]
            cls     = parts[2]

            severity = parts[3] if len(parts) > 3 else ""
            x        = int(parts[4]) if len(parts) > 4 else np.nan
            y        = int(parts[5]) if len(parts) > 5 else np.nan
            radius   = int(parts[6]) if len(parts) > 6 else np.nan

            records.append({
                "refnum": refnum, "tissue": tissue, "cls": cls,
                "severity": severity, "x": x, "y": y, "radius": radius,
            })
    return pd.DataFrame(records)


class MIASConverter(BaseConverter):
    """Converts MIAS PGM images + all-mias.txt to standardized PNG + CSV."""

    def __init__(self, raw_path: str, client_output_path: str):
        super().__init__(raw_path, client_output_path, dataset_name="MIAS")
        self.raw_path = Path(raw_path)
        self.records  = []

    def convert(self):
        txt_candidates = ["Info.txt", "info.txt", "all-mias.txt", "all_mias.txt", "ALL-MIAS.txt"]
        txt_path = None
        for name in txt_candidates:
            candidate = self.raw_path / name
            if candidate.exists():
                txt_path = candidate
                break

        if txt_path is None:
            raise FileNotFoundError(
                f"Could not find annotation file in {self.raw_path}. "
                f"Expected one of: {txt_candidates}"
            )

        df = _parse_mias_txt(txt_path)

        for _, row in df.iterrows():
            self._process_row(row)

        save_annotations_csv(self.records, self.annotations_file, columns=CANONICAL_COLUMNS)
        print(f"MIAS: saved {len(self.records)} annotations → {self.annotations_file}")

    def _process_row(self, row: pd.Series):
        refnum = row["refnum"]

        # ── Load PGM image ────────────────────────────────────────────────
        pgm_path = self.raw_path / f"{refnum}.pgm"
        if not pgm_path.exists():
            pgm_path = self.raw_path / "all-mias" / f"{refnum}.pgm"
        if not pgm_path.exists():
            print(f"  [MIAS] Image not found: {refnum}.pgm")
            return

        image = cv2.imread(str(pgm_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"  [MIAS] Failed to read: {pgm_path}")
            return

        img_name = f"{refnum}.png"
        save_png_img(image, self.images_dir, refnum)

        # ── Bounding box from center + radius ─────────────────────────────
        is_normal = str(row["cls"]).upper() == "NORM"
        if is_normal or pd.isna(row["x"]):
            bbox_xmin = bbox_ymin = bbox_w = bbox_h = np.nan
        else:
            h, w = image.shape[:2]
            r = int(row["radius"])
            cx, cy = int(row["x"]), int(row["y"])
            # MIAS origin is bottom-left; flip Y axis to top-left convention
            cy = h - cy
            bbox_xmin = float(max(0, cx - r))
            bbox_ymin = float(max(0, cy - r))
            bbox_w    = float(min(2 * r, w - bbox_xmin))
            bbox_h    = float(min(2 * r, h - bbox_ymin))

        # ── Metadata ──────────────────────────────────────────────────────
        self.records.append(normalize_row({
            "image_name":  img_name,
            "bbox_xmin":   bbox_xmin,
            "bbox_ymin":   bbox_ymin,
            "bbox_width":  bbox_w,
            "bbox_height": bbox_h,
            "dataset_name": "MIAS",
        }))
