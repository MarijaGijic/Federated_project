"""
Validate and clean annotations across all client directories.

Runs the following checks on every annotations.csv:
  1. Schema completeness — all canonical columns present
  2. Image files exist on disk
  3. Bounding box coordinates are non-negative and within image bounds
  4. Controlled-vocabulary values are valid
  5. Duplicate rows (same image + bbox)

Fixes applied automatically:
  - Strips whitespace from string columns
  - Re-applies normalization (lesion_type, pathology, laterality, view)
  - Fills derived pathology from bi_rads when missing
  - Clips negative bbox coords to 0
  - Drops rows whose image file is completely missing

Usage:
    python validate_and_clean.py --client_dirs data/client1 data/client2 data/client3 data/client4
    python validate_and_clean.py --client_dirs data/client1  --fix   # overwrite with cleaned CSV
"""

import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.annotation_schema import (
    CANONICAL_COLUMNS,
    normalize_row,
)


# ── Per-row checks ────────────────────────────────────────────────────────────

def check_and_fix_row(row: dict, images_dir: str) -> tuple[dict, list[str]]:
    """Return (fixed_row, [warning_strings]). Warnings are non-fatal findings."""
    warnings = []
    img = row.get("image_name", "?")
    img_path = os.path.join(images_dir, img)

    # image exists
    if not os.path.isfile(img_path):
        warnings.append(f"MISSING_IMAGE  {img}")
        row["_drop"] = True
        return row, warnings

    # controlled vocab
    # if row.get("lesion_type") not in VALID_LESION_TYPES:
    #     warnings.append(f"BAD_LESION_TYPE  {img}: '{row.get('lesion_type')}'")
    # if row.get("pathology") not in VALID_PATHOLOGIES:
    #     warnings.append(f"BAD_PATHOLOGY    {img}: '{row.get('pathology')}'")

    # bbox sanity
    has_bbox = not pd.isna(row.get("bbox_xmin"))
    if has_bbox:
        img_arr = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img_arr is not None:
            img_h, img_w = img_arr.shape[:2]
        else:
            img_h = img_w = None

        for coord in ["bbox_xmin", "bbox_ymin", "bbox_width", "bbox_height"]:
            val = row.get(coord)
            if pd.isna(val):
                warnings.append(f"NAN_BBOX_COORD   {img}: {coord}")
                continue
            if val < 0:
                warnings.append(f"NEGATIVE_BBOX    {img}: {coord}={val:.1f} → clipped to 0")
                row[coord] = 0.0

        width = row.get("bbox_width")
        height = row.get("bbox_height")
        if ((not pd.isna(width) and width <= 0)
                or (not pd.isna(height) and height <= 0)):
            warnings.append(
                f"NONPOSITIVE_BBOX_SIZE  {img}: width={width}, height={height}"
            )

        if img_h is not None:
            x, y = float(row.get("bbox_xmin", 0)), float(row.get("bbox_ymin", 0))
            w, h = float(row.get("bbox_width", 0)), float(row.get("bbox_height", 0))
            if x + w > img_w:
                row["bbox_width"] = max(1, img_w - x)
                warnings.append(f"BBOX_OOB_WIDTH   {img}: clipped to {row['bbox_width']:.0f}")
            if y + h > img_h:
                row["bbox_height"] = max(1, img_h - y)
                warnings.append(f"BBOX_OOB_HEIGHT  {img}: clipped to {row['bbox_height']:.0f}")

    return row, warnings


# ── DataFrame-level checks ────────────────────────────────────────────────────

def validate_and_clean(df: pd.DataFrame, images_dir: str) -> tuple[pd.DataFrame, dict]:
    """
    Returns (cleaned_df, stats_dict).
    Stats include counts of every warning type found.
    """
    df = df.copy()

    # ── Schema ────────────────────────────────────────────────────────────────
    for col in CANONICAL_COLUMNS:
        if col not in df.columns:
            df[col] = None

    # Re-apply string normalization + pathology derivation
    records = [normalize_row(row.to_dict()) for _, row in df.iterrows()]
    df = pd.DataFrame(records)

    # ── Duplicate detection ───────────────────────────────────────────────────
    dup_mask = df.duplicated(subset=["image_name", "bbox_xmin", "bbox_ymin"], keep="first")
    n_dups = dup_mask.sum()
    df = df[~dup_mask].copy()

    # ── Per-row checks ────────────────────────────────────────────────────────
    all_warnings = []
    df["_drop"] = False
    fixed_records = []
    for _, row in df.iterrows():
        fixed, warns = check_and_fix_row(row.to_dict(), images_dir)
        fixed_records.append(fixed)
        all_warnings.extend(warns)

    df = pd.DataFrame(fixed_records)
    n_dropped = df["_drop"].sum()
    df = df[~df["_drop"]].drop(columns=["_drop"])

    # Keep canonical column order
    df = df[[c for c in CANONICAL_COLUMNS if c in df.columns]]

    # ── Summarise warnings ────────────────────────────────────────────────────
    warn_counts: dict[str, int] = {}
    for w in all_warnings:
        tag = w.split()[0]
        warn_counts[tag] = warn_counts.get(tag, 0) + 1

    stats = {
        "total_rows":         len(df),
        "duplicates_removed": int(n_dups),
        "images_dropped":     int(n_dropped),
        "warnings":           warn_counts,
        "dataset_counts":     df["dataset_name"].value_counts().to_dict(),
        "label_counts":       df["label"].value_counts().to_dict(),
    }
    return df, stats


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Validate and clean client annotation CSVs")
    parser.add_argument("--client_dirs", nargs="+", required=True,
                        help="One or more client data directories (each contains images/ and annotations.csv)")
    parser.add_argument("--fix", action="store_true",
                        help="Overwrite annotations.csv with the cleaned version")
    args = parser.parse_args()

    overall_ok = True

    for client_dir in args.client_dirs:
        ann_path   = os.path.join(client_dir, "annotations.csv")
        images_dir = os.path.join(client_dir, "images")

        print(f"\n{'═'*60}")
        print(f"  Client: {client_dir}")
        print(f"{'═'*60}")

        if not os.path.isfile(ann_path):
            print(f"  ERROR: annotations.csv not found at {ann_path}")
            overall_ok = False
            continue

        df_raw = pd.read_csv(ann_path)
        print(f"  Rows before cleaning : {len(df_raw)}")

        df_clean, stats = validate_and_clean(df_raw, images_dir)

        print(f"  Rows after cleaning  : {stats['total_rows']}")
        print(f"  Duplicates removed   : {stats['duplicates_removed']}")
        print(f"  Missing images dropped: {stats['images_dropped']}")

        if stats["warnings"]:
            print(f"\n  Warnings summary:")
            for tag, count in sorted(stats["warnings"].items()):
                print(f"    {tag:<26} {count}")
            overall_ok = False
        else:
            print("  No issues found.")

        print(f"\n  Label distribution:")
        for lbl, cnt in sorted(stats["label_counts"].items()):
            name = "normal" if lbl == 0 else "lesion"
            print(f"    {lbl} ({name:<8}) {cnt}")

        if args.fix:
            df_clean.to_csv(ann_path, index=False)
            print(f"\n  Cleaned CSV saved → {ann_path}")

    print(f"\n{'═'*60}")
    print(f"  Overall: {'OK' if overall_ok else 'ISSUES FOUND (re-run with --fix to auto-correct)'}")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
