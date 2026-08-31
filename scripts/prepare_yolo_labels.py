import argparse
import os
import random
import shutil

import cv2
import pandas as pd
import yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_dirs", nargs="+", required=True,
                        help="Prepared client folders (each has images/ and annotations.csv)")
    parser.add_argument("--output_dir", required=True,
                        help="Where to write the YOLO dataset")
    parser.add_argument("--val_split", type=float, default=0.2)
    return parser.parse_args()


def create_dirs(output_dir):
    for split in ("train", "val"):
        os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)


def split_image_names(image_names, val_split, seed=42):
    names = sorted(image_names)
    random.seed(seed)
    random.shuffle(names)
    n_val = max(1, int(len(names) * val_split))
    val   = set(names[:n_val])
    train = set(names[n_val:])
    return train, val


def rows_to_yolo_lines(rows, img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    h, w = img.shape

    lines = []
    for _, row in rows.iterrows():
        if row["label"] == 0 or pd.isna(row["bbox_xmin"]):
            continue

        cx = (row["bbox_xmin"] + row["bbox_width"]  / 2) / w
        cy = (row["bbox_ymin"] + row["bbox_height"] / 2) / h
        bw =  row["bbox_width"]  / w
        bh =  row["bbox_height"] / h

        cx, cy, bw, bh = (max(0.0, min(1.0, v)) for v in (cx, cy, bw, bh))

        lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    return lines


def process_client(client_dir, output_dir, val_split):
    ann_path   = os.path.join(client_dir, "annotations.csv")
    images_dir = os.path.join(client_dir, "images")

    df = pd.read_csv(ann_path)
    image_names = df["image_name"].unique().tolist()

    train_names, val_names = split_image_names(image_names, val_split)

    counts = {"train": 0, "val": 0}
    for img_name in image_names:
        split = "train" if img_name in train_names else "val"
        rows  = df[df["image_name"] == img_name]
        src   = os.path.join(images_dir, img_name)

        if not os.path.isfile(src):
            print(f"  [WARN] Image not found, skipping: {src}")
            continue

        dst_img = os.path.join(output_dir, "images", split, img_name)
        shutil.copy2(src, dst_img)

        label_name = os.path.splitext(img_name)[0] + ".txt"
        dst_lbl    = os.path.join(output_dir, "labels", split, label_name)
        lines      = rows_to_yolo_lines(rows, src)
        with open(dst_lbl, "w") as f:
            f.write("\n".join(lines))

        counts[split] += 1

    print(f"  train: {counts['train']} images  |  val: {counts['val']} images")


def write_dataset_yaml(output_dir):
    config = {
        "path":  os.path.abspath(output_dir),
        "train": "images/train",
        "val":   "images/val",
        "nc":    1,
        "names": ["lesion"],
    }
    yaml_path = os.path.join(output_dir, "dataset.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Wrote {yaml_path}")


def main():
    args = parse_args()
    create_dirs(args.output_dir)

    for client_dir in args.client_dirs:
        print(f"\nProcessing: {client_dir}")
        process_client(client_dir, args.output_dir, args.val_split)

    write_dataset_yaml(args.output_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
