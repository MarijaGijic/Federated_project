"""
Prepare MIAS dataset for federated learning.

Usage (local or Colab):
    python prepare_mias.py \
        --raw_path  /path/to/MIAS \
        --output_path /path/to/data/client3
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.converters.mias_converter import MIASConverter


def main():
    parser = argparse.ArgumentParser(description="Prepare MIAS dataset")
    parser.add_argument("--raw_path",    required=True,
                        help="Path to MIAS root (contains all-mias.txt and *.pgm files)")
    parser.add_argument("--output_path", required=True,
                        help="Output client directory")
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    converter = MIASConverter(args.raw_path, args.output_path)
    converter.convert()
    print(f"Done. Check: {args.output_path}")


if __name__ == "__main__":
    main()
