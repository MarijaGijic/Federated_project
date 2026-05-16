"""
Federated training entry point using Flower simulation mode.

Usage:
    python train_federated.py --config config.yaml

Simulation mode runs all clients in a single process — no separate
server/client processes needed. Ideal for development and academic use.
"""

import argparse
import os
import sys

import yaml
import flwr as fl

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.federated.client import MammographyClient
from src.federated.server import build_strategy


def load_config(path: str) -> dict:
    if not os.path.isabs(path):
        path = os.path.join(ROOT, path)
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    client_dirs = [
        os.path.join(ROOT, d) for d in cfg["data"]["client_dirs"]
    ]
    num_clients = len(client_dirs)

    client_cfg = {
        "in_channels": cfg["model"]["in_channels"],
        "out_channels": cfg["model"]["out_channels"],
        "features": cfg["model"]["features"],
        "image_size": cfg["data"]["image_size"],
        "batch_size": cfg["training"]["batch_size"],
        "learning_rate": cfg["training"]["learning_rate"],
        "local_epochs": cfg["training"]["local_epochs"],
    }

    # ── Client factory ────────────────────────────────────────────────────────
    def client_fn(cid: str) -> fl.client.Client:
        client_dir = client_dirs[int(cid)]
        print(f"  [Client {cid}] Loading data from: {client_dir}")
        return MammographyClient(client_dir, client_cfg).to_client()

    # ── Strategy ──────────────────────────────────────────────────────────────
    strategy = build_strategy(
        num_clients=num_clients,
        fraction_fit=cfg["federated"]["fraction_fit"],
        fraction_evaluate=cfg["federated"]["fraction_evaluate"],
        model_save_path=os.path.join(ROOT, cfg["output"]["model_save_dir"]),
        unet_config=cfg["model"],
    )

    # ── Run simulation ────────────────────────────────────────────────────────
    print(f"\nStarting federated simulation")
    print(f"  Clients      : {num_clients}")
    print(f"  Rounds       : {cfg['federated']['num_rounds']}")
    print(f"  Local epochs : {cfg['training']['local_epochs']}")
    print(f"  Image size   : {cfg['data']['image_size']}x{cfg['data']['image_size']}\n")

    # Run 1 client at a time: num_gpus=1.0 caps concurrency to 1 (one GPU total),
    # which prevents parallel clients from exhausting RAM on constrained runtimes.
    import torch
    num_gpus = 1.0 if torch.cuda.is_available() else 0.0
    client_resources = {"num_cpus": 1, "num_gpus": num_gpus}

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg["federated"]["num_rounds"]),
        strategy=strategy,
        client_resources=client_resources,
    )

    # ── Save training history ─────────────────────────────────────────────────
    import json
    results_dir = os.path.join(ROOT, cfg["output"]["results_dir"])
    os.makedirs(results_dir, exist_ok=True)
    history_path = os.path.join(results_dir, "history.json")
    with open(history_path, "w") as f:
        json.dump(
            {
                "losses_distributed": history.losses_distributed,
                "metrics_distributed": history.metrics_distributed,
            },
            f,
            indent=2,
        )
    print(f"\nTraining history saved → {history_path}")


if __name__ == "__main__":
    main()
