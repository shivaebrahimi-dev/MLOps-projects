"""Training entry point: ``fashion-train --config configs/train.toml``."""

from __future__ import annotations

import argparse
import copy
import logging
import random
import sys
from collections.abc import Callable
from dataclasses import dataclass, field

import httpx
import mlflow
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from fashion_mlops.config import ServiceSettings, TrainConfig, load_train_config
from fashion_mlops.data import make_loaders
from fashion_mlops.model import LeNet5

logger = logging.getLogger("fashion_mlops.train")

EpochCallback = Callable[[int, dict[str, float]], None]


@dataclass
class TrainResult:
    model: nn.Module
    # Metrics of the epoch whose weights are in ``model``.
    final_metrics: dict[str, float]
    epochs_run: int
    history: list[dict[str, float]] = field(default_factory=list)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_optimizer(config: TrainConfig, model: nn.Module) -> torch.optim.Optimizer:
    if config.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    return torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )


def build_loss(name: str) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Both losses take the model's raw logits and integer class labels."""
    if name == "nll":
        return lambda logits, labels: F.nll_loss(F.log_softmax(logits, dim=1), labels)
    return F.cross_entropy


@torch.no_grad()
def evaluate(model, loader, loss_fn, device) -> tuple[float, float]:
    """Return (mean loss, accuracy) over ``loader``."""
    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += loss_fn(logits, labels).item() * len(labels)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        count += len(labels)
    return total_loss / count, correct / count


def train_model(
    config: TrainConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    on_epoch_end: EpochCallback | None = None,
) -> TrainResult:
    """Train LeNet-5 and report train/validation metrics after every epoch."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LeNet5().to(device)
    optimizer = build_optimizer(config, model)
    loss_fn = build_loss(config.loss)

    history: list[dict[str, float]] = []
    best_val_loss = float("inf")
    best_state, best_metrics = None, None
    epochs_without_improvement = 0
    final_metrics: dict[str, float] = {}

    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss, count = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(labels)
            count += len(labels)

        val_loss, val_accuracy = evaluate(model, val_loader, loss_fn, device)
        final_metrics = {
            "train_loss": running_loss / count,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        }
        history.append(final_metrics)
        logger.info(
            "epoch %d/%d  train_loss=%.4f  val_loss=%.4f  val_accuracy=%.4f",
            epoch, config.epochs, final_metrics["train_loss"], val_loss, val_accuracy,
        )
        if on_epoch_end is not None:
            on_epoch_end(epoch, final_metrics)

        if config.early_stopping_patience > 0:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                best_metrics = final_metrics
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= config.early_stopping_patience:
                    logger.info("early stopping at epoch %d; restoring best weights", epoch)
                    break

    if best_state is not None:
        # With early stopping enabled the returned model is the best one seen, not the last.
        model.load_state_dict(best_state)
        final_metrics = best_metrics

    return TrainResult(
        model=model.cpu(), final_metrics=final_metrics, epochs_run=len(history), history=history
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LeNet-5 on Fashion-MNIST.")
    parser.add_argument("--config", help="path to a TOML training configuration file")
    # One optional flag per TrainConfig field, e.g. --batch-size, so the CLI never drifts from it.
    for name, info in TrainConfig.model_fields.items():
        parser.add_argument(
            "--" + name.replace("_", "-"), dest=name, help=f"default: {info.default}"
        )
    return parser.parse_args(argv)


def check_mlflow_reachable(tracking_uri: str) -> None:
    """Fail fast with a clear message instead of waiting through MLflow's long retries."""
    if not tracking_uri.startswith("http"):
        return
    try:
        httpx.get(tracking_uri.rstrip("/") + "/health", timeout=5).raise_for_status()
    except httpx.HTTPError as exc:
        raise RuntimeError(
            f"MLflow tracking server is not reachable at {tracking_uri} ({exc}). "
            "Start it with `docker compose up -d mlflow` or set MLFLOW_TRACKING_URI."
        ) from exc


def run_training(config: TrainConfig, settings: ServiceSettings) -> str:
    """Run one tracked training run and return its MLflow run ID.

    Any exception inside the ``with`` block marks the MLflow run as FAILED.
    """
    check_mlflow_reachable(settings.mlflow_tracking_uri)
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(config.experiment_name)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info("MLflow run %s in experiment '%s'", run_id, config.experiment_name)
        # The resolved configuration is logged twice: as searchable params and as one
        # file that can be fed straight back in to reproduce the run.
        mlflow.log_params(config.model_dump())
        mlflow.log_dict({"training": config.model_dump()}, "config.json")

        set_seed(config.seed)
        train_loader, val_loader = make_loaders(config)
        result = train_model(
            config,
            train_loader,
            val_loader,
            on_epoch_end=lambda epoch, metrics: mlflow.log_metrics(metrics, step=epoch),
        )

        mlflow.log_metrics(
            {
                "final_val_accuracy": result.final_metrics["val_accuracy"],
                "final_val_loss": result.final_metrics["val_loss"],
                "epochs_run": result.epochs_run,
            }
        )
        logger.info("final validation accuracy: %.4f", result.final_metrics["val_accuracy"])
    return run_id


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = vars(parse_args(argv))
    config = load_train_config(args.pop("config"), overrides=args)
    logger.info("training configuration: %s", config.model_dump())

    try:
        run_training(config, ServiceSettings())
    except Exception:
        logger.exception("training run FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
