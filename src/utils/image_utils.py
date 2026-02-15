import os
import cv2
import numpy as np
import pandas as pd

def resize_image(image, target_size=(1024, 1024)):
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

def normalize_uint8(image):
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")

def save_png(image, path):
    cv2.imwrite(path, image)

def save_png_pair(image, mask, images_dir, masks_dir, prefix):
    img_path = os.path.join(images_dir, f"{prefix}.png")
    mask_path = os.path.join(masks_dir, f"{prefix}_mask.png")

    cv2.imwrite(img_path, image)
    cv2.imwrite(mask_path, mask * 255)

    return img_path, mask_path

def save_png_img(image, images_dir, prefix):
    os.makedirs(images_dir, exist_ok=True)
    img_path = os.path.join(images_dir, f"{prefix}.png")
    cv2.imwrite(img_path, image)
    return img_path

def save_annotations_csv(records, path):
    df = pd.DataFrame(records)
    df.to_csv(path, index=False)