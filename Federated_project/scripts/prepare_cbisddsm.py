"""
Prepare CBIS-DDSM dataset for federated learning.

Usage (local or Colab):
    python prepare_cbisddsm.py \
        --raw_path  /path/to/CBIS-DDSM \
        --output_path /path/to/data/client2
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.converters.cbisddsm_converter import CBISDDSMConverter


def main():
    parser = argparse.ArgumentParser(description="Prepare CBIS-DDSM dataset")
    parser.add_argument("--raw_path",    required=True,
                        help="Path to CBIS-DDSM root (contains csv/ and DICOM folders)")
    parser.add_argument("--output_path", required=True,
                        help="Output client directory (images/ + annotations.csv created here)")
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    converter = CBISDDSMConverter(args.raw_path, args.output_path)
    converter.convert()
    print(f"Done. Check: {args.output_path}")


if __name__ == "__main__":
    main()
