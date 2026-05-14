"""
Prepare VinDr-Mammo dataset for federated learning.

Usage (local or Colab):
    python prepare_vindr.py \
        --raw_path  /path/to/vindr-mammo \
        --output_path /path/to/data/client4
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.converters.vindr_converter import VinDrConverter


def main():
    parser = argparse.ArgumentParser(description="Prepare VinDr-Mammo dataset")
    parser.add_argument("--raw_path",    required=True,
                        help="Path to VinDr-Mammo root "
                             "(contains finding_annotations.csv, breast_level_annotations.csv, images/)")
    parser.add_argument("--output_path", required=True,
                        help="Output client directory")
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    converter = VinDrConverter(args.raw_path, args.output_path)
    converter.convert()
    print(f"Done. Check: {args.output_path}")


if __name__ == "__main__":
    main()
