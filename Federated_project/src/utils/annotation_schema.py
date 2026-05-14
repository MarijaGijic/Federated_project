"""
Canonical annotation schema shared by all four database converters.

Every converter outputs a CSV with exactly these columns in this order.
"""

import numpy as np
import pandas as pd

# ── Canonical column order ────────────────────────────────────────────────────

CANONICAL_COLUMNS = [
    "image_name",       # str  — PNG filename (e.g. "20010001.png")
    "bbox_xmin",        # float — NaN for normal / no-finding cases
    "bbox_ymin",        # float
    "bbox_width",       # float
    "bbox_height",      # float
    "lesion_type",      # str  — see VALID_LESION_TYPES
    "pathology",        # str  — see VALID_PATHOLOGIES
    "bi_rads",          # int  — 0 = unknown, 1-6
    "breast_density",   # int  — 0 = unknown, 1-4  (ACR density)
    "laterality",       # str  — "L", "R", "unknown"
    "view",             # str  — "CC", "MLO", "unknown"
    "dataset_name",     # str  — {INbreast, CBIS-DDSM, MIAS, VinDr-Mammo}
]

VALID_LESION_TYPES = {"mass", "calcification", "distortion", "asymmetry", "normal", "other"}
VALID_PATHOLOGIES   = {"benign", "malignant", "probably_benign", "normal", "unknown"}
VALID_LATERALITIES  = {"L", "R", "unknown"}
VALID_VIEWS         = {"CC", "MLO", "unknown"}

# ── Normalization maps ────────────────────────────────────────────────────────

LESION_TYPE_MAP: dict[str, str] = {
    # INbreast (Excel / XML Name field)
    "mass": "mass", "mass ": "mass",
    "micros": "calcification", "micros ": "calcification",
    "distortion": "distortion", "distortion ": "distortion",
    "asymmetry": "asymmetry", "asymmetry ": "asymmetry",
    "nodule": "mass",

    # CBIS-DDSM (abnormality_type column)
    "calcification": "calcification",

    # MIAS (CLASS column)
    "calc": "calcification",
    "circ": "mass",         # circumscribed mass
    "spic": "mass",         # spiculated mass
    "misc": "mass",         # other mass
    "arch": "distortion",
    "asym": "asymmetry",
    "norm": "normal",

    # VinDr-Mammo (finding_categories column)
    "suspicious calcification": "calcification",
    "architectural distortion": "distortion",
    "focal asymmetry": "asymmetry",
    "global asymmetry": "asymmetry",
    "no finding": "normal",
    "skin thickening": "other",
    "nipple retraction": "other",
    "suspicious lymph node": "other",
    "axillary adenopathy": "other",
}

PATHOLOGY_MAP: dict[str, str] = {
    # CBIS-DDSM
    "benign": "benign",
    "malignant": "malignant",
    "benign_without_callback": "benign",

    # MIAS
    "b": "benign",
    "m": "malignant",
    "": "normal",

    # Generic
    "normal": "normal",
    "unknown": "unknown",
}

LATERALITY_MAP: dict[str, str] = {
    "left": "L", "l": "L",
    "right": "R", "r": "R",
    "unknown": "unknown", "": "unknown",
}

VIEW_MAP: dict[str, str] = {
    "cc": "CC",
    "mlo": "MLO",
    "unknown": "unknown", "": "unknown",
}

DENSITY_MAP: dict[str, int] = {
    # MIAS background tissue → ACR density approximation
    "f": 1,   # fatty
    "g": 2,   # fatty-glandular
    "d": 4,   # dense
    # Strings that already are numbers
    "1": 1, "2": 2, "3": 3, "4": 4,
}

# ── Derivation helpers ────────────────────────────────────────────────────────

def birads_to_pathology(birads) -> str:
    """Derive pathology label from BI-RADS score when not explicitly given."""
    try:
        b = int(float(birads))
    except (TypeError, ValueError):
        return "unknown"
    if b == 0:
        return "unknown"
    if b == 1:
        return "normal"
    if b in (2, 3):
        return "benign" if b == 2 else "probably_benign"
    if b in (4, 5, 6):
        return "malignant"
    return "unknown"


# ── Normalization ─────────────────────────────────────────────────────────────

def _norm(value, mapping: dict, default: str = "unknown") -> str:
    if pd.isna(value) or str(value).strip() == "":
        return default
    return mapping.get(str(value).strip().lower(), default)


def normalize_row(row: dict) -> dict:
    """Apply all normalization rules to a single record dict."""
    row = dict(row)

    row["lesion_type"] = _norm(row.get("lesion_type"), LESION_TYPE_MAP, "other")
    row["laterality"]  = _norm(row.get("laterality"),  LATERALITY_MAP,  "unknown")
    row["view"]        = _norm(row.get("view"),         VIEW_MAP,        "unknown")

    # Pathology: use explicit value if present, else derive from BI-RADS
    raw_path = str(row.get("pathology", "")).strip().lower()
    if raw_path in PATHOLOGY_MAP:
        row["pathology"] = PATHOLOGY_MAP[raw_path]
    elif raw_path == "" or pd.isna(row.get("pathology")):
        row["pathology"] = birads_to_pathology(row.get("bi_rads"))
    else:
        row["pathology"] = "unknown"

    # bi_rads: coerce to int, 0 = unknown
    try:
        row["bi_rads"] = int(float(row.get("bi_rads", 0) or 0))
    except (TypeError, ValueError):
        row["bi_rads"] = 0

    # breast_density: coerce to int, 0 = unknown
    raw_density = str(row.get("breast_density", "")).strip().lower()
    if raw_density in DENSITY_MAP:
        row["breast_density"] = DENSITY_MAP[raw_density]
    else:
        try:
            row["breast_density"] = int(float(raw_density))
        except (TypeError, ValueError):
            row["breast_density"] = 0

    return row


# ── Validation ────────────────────────────────────────────────────────────────

def validate_dataframe(df: pd.DataFrame, images_dir: str) -> list[str]:
    """Return a list of warning strings. Empty list = clean."""
    import os
    issues = []

    missing_cols = [c for c in CANONICAL_COLUMNS if c not in df.columns]
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")
        return issues  # cannot check further

    for _, row in df.iterrows():
        img = row["image_name"]
        path = os.path.join(images_dir, img)
        if not os.path.isfile(path):
            issues.append(f"Image file missing: {img}")

        if row["lesion_type"] not in VALID_LESION_TYPES:
            issues.append(f"{img}: unknown lesion_type '{row['lesion_type']}'")

        if row["pathology"] not in VALID_PATHOLOGIES:
            issues.append(f"{img}: unknown pathology '{row['pathology']}'")

        if not pd.isna(row["bbox_xmin"]):
            for coord in ["bbox_xmin", "bbox_ymin", "bbox_width", "bbox_height"]:
                if row[coord] < 0:
                    issues.append(f"{img}: negative {coord} = {row[coord]}")

    return issues
