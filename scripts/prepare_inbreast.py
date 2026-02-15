import os
import argparse
from src.converters.inbreast_converter import InbreastConverter

def main():
    parser = argparse.ArgumentParser(description="Prepare INbreast dataset for federated learning")
    parser.add_argument("--raw_path", type=str, required=True,
                        help="Path to raw INbreast folder (DICOM + XML + Excel)")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to output folder for client1 (images + CSV)")
    parser.add_argument("--dataset_name", type=str, default="INbreast",
                        help="Name of the dataset to store in CSV")
    args = parser.parse_args()

    # Make output folders
    images_dir = os.path.join(args.output_path, "images")
    os.makedirs(images_dir, exist_ok=True)

    # Run converter
    converter = InbreastConverter(
        raw_path=args.raw_path,
        images_dir=images_dir,
        dataset_name=args.dataset_name
    )
    converter.run()

    print(f"INbreast preprocessing finished!")
    print(f"Images saved in: {images_dir}")
    print(f"CSV saved in: {args.output_path}/annotations.csv")

if __name__ == "__main__":
    main()