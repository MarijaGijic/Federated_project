from typing import Dict, List, Optional, Tuple

import csv
import os

import flwr as fl
import torch

from src.models.unet import UNet


def weighted_average(metrics: List[Tuple[int, Dict]]) -> Dict:
    """Aggregate per-client metrics weighted by dataset size."""
    total = sum(n for n, _ in metrics)
    aggregated = {}

    for n, m in metrics:
        for key, val in m.items():
            aggregated[key] = aggregated.get(key, 0.0) + val * (n / total)

    return aggregated


def fit_config(server_round: int, local_epochs: int) -> Dict:
    return {
        "local_epochs": local_epochs,
        "server_round": server_round,
    }


def build_strategy(
    num_clients: int,
    fraction_fit: float = 1.0,
    fraction_evaluate: float = 1.0,
    local_epochs: int = 3,
    model_save_path: Optional[str] = None,
    metrics_save_path: Optional[str] = None,
    unet_config: Optional[dict] = None,
) -> fl.server.strategy.Strategy:
    """Return FedAvg strategy with model and metric checkpointing."""

    class SaveModelFedAvg(fl.server.strategy.FedAvg):

        def aggregate_fit(self, server_round, results, failures):
            aggregated_parameters, metrics = super().aggregate_fit(
                server_round,
                results,
                failures,
            )

            if aggregated_parameters is not None and model_save_path:
                _save_global_model(
                    aggregated_parameters,
                    server_round,
                    model_save_path,
                    unet_config,
                )

            return aggregated_parameters, metrics

        def aggregate_evaluate(self, server_round, results, failures):
            loss, metrics = super().aggregate_evaluate(
                server_round,
                results,
                failures,
            )

            if loss is not None and metrics_save_path:
                os.makedirs(metrics_save_path, exist_ok=True)

                csv_path = os.path.join(
                    metrics_save_path,
                    "federated_metrics.csv",
                )

                file_exists = os.path.exists(csv_path)

                with open(csv_path, "a", newline="") as f:
                    writer = csv.writer(f)

                    if not file_exists:
                        writer.writerow(["round", "loss", "dice"])

                    writer.writerow(
                        [
                            server_round,
                            float(loss),
                            float(metrics.get("dice", 0.0)),
                        ]
                    )

                print(
                    f"[Server] Saved metrics → "
                    f"round={server_round}, "
                    f"loss={loss:.6f}, "
                    f"dice={metrics.get('dice', 0.0):.6f}"
                )

            return loss, metrics

    strategy = SaveModelFedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        on_fit_config_fn=lambda server_round: fit_config(
            server_round,
            local_epochs,
        ),
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=weighted_average,
    )

    return strategy


def _save_global_model(
    parameters,
    server_round: int,
    save_path: str,
    unet_config: dict,
):
    from collections import OrderedDict
    from flwr.common import parameters_to_ndarrays

    cfg = unet_config or {}

    model = UNet(
        in_channels=cfg.get("in_channels", 1),
        out_channels=cfg.get("out_channels", 1),
        features=cfg.get("features", [64, 128, 256, 512]),
    )

    ndarrays = parameters_to_ndarrays(parameters)

    state_dict = OrderedDict(
        {
            k: torch.tensor(v)
            for k, v in zip(model.state_dict().keys(), ndarrays)
        }
    )

    model.load_state_dict(state_dict, strict=True)

    os.makedirs(save_path, exist_ok=True)

    path = os.path.join(
        save_path,
        f"global_model_round_{server_round}.pt",
    )

    torch.save(model.state_dict(), path)

    print(f"[Server] Saved global model → {path}")
