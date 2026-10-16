"""Training-complete webhook notification."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx


class WebhookError(RuntimeError):
    """The webhook receiver could not be reached or answered with an error status."""


def build_payload(
    *,
    run_id: str,
    experiment: str,
    bucket: str,
    object_key: str,
    val_accuracy: float,
    val_loss: float,
    epochs_run: int,
) -> dict[str, Any]:
    return {
        "event": "training_complete",
        "status": "success",
        "run_id": run_id,
        "experiment": experiment,
        "model": {
            "bucket": bucket,
            "object_key": object_key,
            "uri": f"s3://{bucket}/{object_key}",
        },
        "metrics": {
            "val_accuracy": val_accuracy,
            "val_loss": val_loss,
            "epochs_run": epochs_run,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def send_webhook(
    url: str,
    payload: dict[str, Any],
    timeout: float = 10.0,
    transport: httpx.BaseTransport | None = None,
) -> None:
    """POST ``payload`` as JSON. ``transport`` exists so tests can avoid the network."""
    try:
        with httpx.Client(timeout=timeout, transport=transport) as client:
            client.post(url, json=payload).raise_for_status()
    except httpx.HTTPError as exc:
        raise WebhookError(f"Webhook to {url} failed: {exc}") from exc
