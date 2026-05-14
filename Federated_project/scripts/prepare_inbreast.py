import os
import argparse
from src.converters.inbreast_converter import INbreastConverter

def main():
    parser = argparse.ArgumentParser(description="Prepare INbreast dataset for federated learning")
    parser.add_argument(
        "--raw_path",
        type=str,
        required=True,
        help="Path to raw INbreast folder containing AllDICOMs, AllXML, INbreast.xls"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path where client1 folder will be created (images + annotations.csv)"
    )
    parser.add_argument(
        "--excel_path",
        type=str,
        required=True,
        help="Path to INbreast.xls file with metadata"
    )
    args = parser.parse_args()

    # Create output folder if it doesn't exist
    os.makedirs(args.output_path, exist_ok=True)

    # Run converter
    converter = INbreastConverter(
        raw_path=args.raw_path,
        client_output_path=args.output_path,
        excel_path=args.excel_path
    )
    converter.convert()

    print(f"INbreast preparation complete! Check folder: {args.output_path}")

if __name__ == "__main__":
    main()