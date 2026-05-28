import sys
import os
from collections import OrderedDict
from typing import Dict, List, Tuple

import flwr as fl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split

# Allow imports from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.datasets.mammography_dataset import MammographyDataset
from src.models.unet import UNet


# ─── Loss helpers ────────────────────────────────────────────────────────────

def dice_loss(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    probs = torch.sigmoid(logits).view(-1)
    tgts = targets.view(-1)
    intersection = (probs * tgts).sum()
    return 1.0 - (2.0 * intersection + smooth) / (probs.sum() + tgts.sum() + smooth)


def combined_loss(logits, targets):
    bce = nn.BCEWithLogitsLoss()(logits, targets)
    dl = dice_loss(logits, targets)
    return 0.5 * bce + 0.5 * dl


def dice_score(logits: torch.Tensor, targets: torch.Tensor,
               threshold: float = 0.5, smooth: float = 1.0) -> float:
    preds = (torch.sigmoid(logits) > threshold).float().view(-1)
    tgts = targets.view(-1)
    intersection = (preds * tgts).sum()
    return ((2.0 * intersection + smooth) / (preds.sum() + tgts.sum() + smooth)).item()


# ─── Train / eval loops ───────────────────────────────────────────────────────

def train_one_round(model, loader, optimizer, device, epochs: int) -> float:
    model.train()
    total_loss = 0.0
    for _ in range(epochs):
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = combined_loss(logits, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    return total_loss / (epochs * len(loader))


def evaluate_model(model, loader, device) -> Tuple[float, float]:
    model.eval()
    total_loss, total_dice = 0.0, 0.0
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            total_loss += combined_loss(logits, masks).item()
            total_dice += dice_score(logits, masks)
    n = len(loader)
    return total_loss / n, total_dice / n


# ─── Flower client ────────────────────────────────────────────────────────────

class MammographyClient(fl.client.NumPyClient):
    def __init__(self, client_dir: str, config: dict):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config

        self.model = UNet(
            in_channels=config.get("in_channels", 1),
            out_channels=config.get("out_channels", 1),
            features=config.get("features", [64, 128, 256, 512]),
        ).to(self.device)

        dataset = MammographyDataset(
            client_dir=client_dir,
            image_size=config.get("image_size", 512),
            augment=True,
        )
        val_size = max(1, int(0.2 * len(dataset)))
        train_size = len(dataset) - val_size
        train_ds, val_split = random_split(dataset, [train_size, val_size])

        # Build a separate no-augmentation dataset for validation using the same indices.
        # random_split returns Subsets sharing the original dataset object, so mutating
        # .dataset.augment would disable augmentation for training too.
        val_dataset = MammographyDataset(
            client_dir=client_dir,
            image_size=config.get("image_size", 512),
            augment=False,
        )
        val_ds = Subset(val_dataset, val_split.indices)

        bs = config.get("batch_size", 4)
        self.train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0)
        self.val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.get("learning_rate", 1e-3),
        )

    # ── Parameter exchange ────────────────────────────────────────────────────

    def get_parameters(self, config) -> List:
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters: List) -> None:
        state_dict = OrderedDict(
            {k: torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), parameters)}
        )
        self.model.load_state_dict(state_dict, strict=True)

    # ── Flower interface ──────────────────────────────────────────────────────

    def fit(self, parameters: List, config: Dict) -> Tuple[List, int, Dict]:
        self.set_parameters(parameters)
        epochs = int(config.get("local_epochs", self.config.get("local_epochs", 3)))
        loss = train_one_round(self.model, self.train_loader, self.optimizer, self.device, epochs)
        return self.get_parameters(config={}), len(self.train_loader.dataset), {"train_loss": float(loss)}

    def evaluate(self, parameters: List, config: Dict) -> Tuple[float, int, Dict]:
        self.set_parameters(parameters)
        loss, dice = evaluate_model(self.model, self.val_loader, self.device)
        return float(loss), len(self.val_loader.dataset), {"dice": float(dice)}
