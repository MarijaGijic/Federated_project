"""
Canonical annotation schema shared by all database converters.

Every converter outputs a CSV with exactly these columns in this order.
"""

import pandas as pd

# ── Canonical column order ────────────────────────────────────────────────────

CANONICAL_COLUMNS = [
    "image_name",       # str   — PNG filename (e.g. "20010001.png")
    "bbox_xmin",        # float — NaN for normal / no-finding cases
    "bbox_ymin",        # float
    "bbox_width",       # float
    "bbox_height",      # float
    "label",            # int   — 0 = no lesion, 1 = lesion present
    "dataset_name",     # str   — {INbreast, CBIS-DDSM, MIAS}
]

# VALID_LESION_TYPES = {"mass", "calcification", "distortion", "asymmetry", "normal", "other"}
# VALID_PATHOLOGIES  = {"benign", "malignant", "probably_benign", "normal", "unknown"}

# # ── Normalization maps ────────────────────────────────────────────────────────

# LESION_TYPE_MAP: dict[str, str] = {
#     # INbreast (Excel / XML Name field)
#     "mass": "mass", "mass ": "mass",
#     "micros": "calcification", "micros ": "calcification",
#     "distortion": "distortion", "distortion ": "distortion",
#     "asymmetry": "asymmetry", "asymmetry ": "asymmetry",
#     "nodule": "mass",

#     # CBIS-DDSM (abnormality_type column)
#     "calcification": "calcification",

#     # MIAS (CLASS column)
#     "calc": "calcification",
#     "circ": "mass",         # circumscribed mass
#     "spic": "mass",         # spiculated mass
#     "misc": "mass",         # other mass
#     "arch": "distortion",
#     "asym": "asymmetry",
#     "norm": "normal",
# }

# PATHOLOGY_MAP: dict[str, str] = {
#     # CBIS-DDSM
#     "benign": "benign",
#     "malignant": "malignant",
#     "benign_without_callback": "benign",

#     # MIAS
#     "b": "benign",
#     "m": "malignant",
#     "": "normal",

#     # Generic
#     "normal": "normal",
#     "unknown": "unknown",
# }

# ── Normalization ─────────────────────────────────────────────────────────────

# def _norm(value, mapping: dict, default: str = "unknown") -> str:
#     if pd.isna(value) or str(value).strip() == "":
#         return default
#     return mapping.get(str(value).strip().lower(), default)


def normalize_row(row: dict) -> dict:
    """Apply all normalization rules to a single record dict."""
    row = dict(row)

    # row["lesion_type"] = _norm(row.get("lesion_type"), LESION_TYPE_MAP, "other")

    # raw_path = str(row.get("pathology", "")).strip().lower()
    # if raw_path in PATHOLOGY_MAP:
    #     row["pathology"] = PATHOLOGY_MAP[raw_path]
    # else:
    #     row["pathology"] = "unknown"

    row["label"] = 0 if pd.isna(row.get("bbox_xmin")) else 1

    return row


# ── Validation ────────────────────────────────────────────────────────────────

def validate_dataframe(df: pd.DataFrame, images_dir: str) -> list[str]:
    """Return a list of warning strings. Empty list = clean."""
    import os
    issues = []

    missing_cols = [c for c in CANONICAL_COLUMNS if c not in df.columns]
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")
        return issues

    for _, row in df.iterrows():
        if not os.path.isfile(os.path.join(images_dir, row['image_name'])):
            issues.append(f"Image missing: {row['image_name']}")
        if row['label'] not in (0, 1):
            issues.append(f"{row['image_name']}: invalid label {row['label']}")
        if row['label'] == 1:
            for coord in ["bbox_xmin", "bbox_ymin"]:
                if pd.isna(row[coord]) or row[coord] < 0:
                    issues.append(f"{row['image_name']}: bad {coord}")
            for dimension in ["bbox_width", "bbox_height"]:
                if pd.isna(row[dimension]) or row[dimension] <= 0:
                    issues.append(f"{row['image_name']}: bad {dimension}")

    return issues
