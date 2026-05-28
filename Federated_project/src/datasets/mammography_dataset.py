import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class MammographyDataset(Dataset):
    """
    Loads mammography images and binary segmentation masks per federated client.

    Directory layout expected:
        client_dir/
            images/          PNG images (output of converters)
            annotations.csv  columns: image_name, bbox_xmin, bbox_ymin,
                             bbox_width, bbox_height, ...
    """

    def __init__(self, client_dir: str, image_size: int = 512, augment: bool = False):
        self.images_dir = os.path.join(client_dir, "images")
        self.image_size = image_size
        self.augment = augment

        ann_path = os.path.join(client_dir, "annotations.csv")
        df = pd.read_csv(ann_path)
        self.df = df
        self.image_names = df["image_name"].unique().tolist()

    # ------------------------------------------------------------------
    def _create_mask(self, h: int, w: int, rows: pd.DataFrame) -> np.ndarray:
        mask = np.zeros((h, w), dtype=np.float32)
        for _, row in rows.iterrows():
            if pd.isna(row.get("bbox_xmin")):
                continue
            x = max(0, int(row["bbox_xmin"]))
            y = max(0, int(row["bbox_ymin"]))
            bw = max(1, int(row["bbox_width"]))
            bh = max(1, int(row["bbox_height"]))
            mask[y: y + bh, x: x + bw] = 1.0
        return mask

    def _augment(self, image: np.ndarray, mask: np.ndarray):
        if np.random.rand() > 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
        if np.random.rand() > 0.5:
            angle = np.random.uniform(-15, 15)
            center = (image.shape[1] // 2, image.shape[0] // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
            mask = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]))
        if np.random.rand() > 0.7:
            noise = np.random.normal(0, 5, image.shape).astype(np.float32)
            image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return image, mask

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx: int):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.images_dir, img_name)

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        h, w = image.shape
        rows = self.df[self.df["image_name"] == img_name]
        mask = self._create_mask(h, w, rows)

        if self.augment:
            image, mask = self._augment(image, mask)

        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        # Normalize image to [-1, 1]
        image_t = torch.from_numpy(image.astype(np.float32) / 127.5 - 1.0).unsqueeze(0)
        mask_t = torch.from_numpy(mask).unsqueeze(0)

        return image_t, mask_t
