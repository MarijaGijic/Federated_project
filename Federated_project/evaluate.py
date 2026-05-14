"""
Evaluate a trained global model on one or more client datasets.

Usage:
    python evaluate.py --model results/checkpoints/global_model_round_10.pt \
                       --client_dir data/client1 \
                       --config config.yaml
"""

import argparse
import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.models.unet import UNet
from src.datasets.mammography_dataset import MammographyDataset
from src.federated.client import dice_score, combined_loss


def iou_score(logits: torch.Tensor, targets: torch.Tensor,
              threshold: float = 0.5, smooth: float = 1.0) -> float:
    preds = (torch.sigmoid(logits) > threshold).float().view(-1)
    tgts = targets.view(-1)
    intersection = (preds * tgts).sum()
    union = preds.sum() + tgts.sum() - intersection
    return ((intersection + smooth) / (union + smooth)).item()


def evaluate(model, loader, device):
    model.eval()
    total_loss, total_dice, total_iou = 0.0, 0.0, 0.0
    n = len(loader)
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            total_loss += combined_loss(logits, masks).item()
            total_dice += dice_score(logits, masks)
            total_iou += iou_score(logits, masks)
    return total_loss / n, total_dice / n, total_iou / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to saved .pt model")
    parser.add_argument("--client_dir", required=True, help="Client data directory")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        features=cfg["model"]["features"],
    ).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    print(f"Loaded model from: {args.model}")

    dataset = MammographyDataset(
        client_dir=args.client_dir,
        image_size=cfg["data"]["image_size"],
        augment=False,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Dataset: {len(dataset)} images from {args.client_dir}")

    loss, dice, iou = evaluate(model, loader, device)
    print(f"\n{'─'*40}")
    print(f"  Loss : {loss:.4f}")
    print(f"  Dice : {dice:.4f}")
    print(f"  IoU  : {iou:.4f}")
    print(f"{'─'*40}")


if __name__ == "__main__":
    main()
