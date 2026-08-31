"""Create deterministic, metadata-preserving smoke subsets for three clients."""

from pathlib import Path
import shutil

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "data"
SMOKE_ROOT = ROOT / "data_smoke"
CLIENTS = ("client1", "client2", "client3")
SAMPLE_SIZE = 300
RANDOM_STATE = 42
CANONICAL_COLUMNS = [
    "image_name",
    "bbox_xmin",
    "bbox_ymin",
    "bbox_width",
    "bbox_height",
    "label",
    "dataset_name",
]


def _verify_subset(
    annotations: pd.DataFrame,
    images_dir: Path,
    sampled_images: set[str],
    source_columns: list[str],
) -> None:
    """Verify image membership, image existence, and annotation schema."""
    csv_images = set(annotations["image_name"])
    copied_images = {path.name for path in images_dir.iterdir() if path.is_file()}

    missing_images = csv_images - copied_images
    if missing_images:
        raise RuntimeError(f"CSV references missing images: {sorted(missing_images)}")
    if copied_images != sampled_images:
        raise RuntimeError(
            "Copied image set differs from sampled image set: "
            f"extra={sorted(copied_images - sampled_images)}, "
            f"missing={sorted(sampled_images - copied_images)}"
        )
    if list(annotations.columns) != source_columns:
        raise RuntimeError("Annotation schema changed while creating subset")
    if source_columns != CANONICAL_COLUMNS:
        raise RuntimeError(f"Source schema is not canonical: {source_columns}")


def _create_client_subset(client: str) -> None:
    source_dir = SOURCE_ROOT / client
    source_images_dir = source_dir / "images"
    source_annotations = pd.read_csv(source_dir / "annotations.csv")
    source_columns = list(source_annotations.columns)
    if source_columns != CANONICAL_COLUMNS:
        raise RuntimeError(f"{client}: unexpected annotation schema: {source_columns}")

    unique_images = sorted(source_annotations["image_name"].unique())
    sample_size = min(SAMPLE_SIZE, len(unique_images))
    if sample_size < SAMPLE_SIZE:
        print(
            f"WARNING: {client} has only {sample_size} unique images; "
            "using all available images."
        )

    sampled = pd.Series(unique_images).sample(
        n=sample_size, random_state=RANDOM_STATE, replace=False
    )
    sampled_images = set(sampled.tolist())
    filtered = source_annotations[
        source_annotations["image_name"].isin(sampled_images)
    ].copy()

    target_dir = SMOKE_ROOT / client
    target_images_dir = target_dir / "images"
    target_images_dir.mkdir(parents=True, exist_ok=True)
    for image_name in sorted(sampled_images):
        source_image = source_images_dir / image_name
        if not source_image.is_file():
            raise FileNotFoundError(f"{client}: source image missing: {source_image}")
        shutil.copy2(source_image, target_images_dir / image_name)

    filtered.to_csv(target_dir / "annotations.csv", index=False, columns=source_columns)
    written = pd.read_csv(target_dir / "annotations.csv")
    _verify_subset(written, target_images_dir, sampled_images, source_columns)

    label_counts = written["label"].value_counts().to_dict()
    print(f"{client}:")
    print(f"  source unique images : {len(unique_images)}")
    print(f"  sampled unique images: {len(sampled_images)}")
    print(f"  annotation rows      : {len(written)}")
    print(f"  positive labels      : {int(label_counts.get(1, 0))}")
    print(f"  negative labels      : {int(label_counts.get(0, 0))}")
    print("  verification         : PASS")


def main() -> None:
    if SMOKE_ROOT.exists():
        shutil.rmtree(SMOKE_ROOT)
    SMOKE_ROOT.mkdir(parents=True)

    for client in CLIENTS:
        _create_client_subset(client)


if __name__ == "__main__":
    main()
