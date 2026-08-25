from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from .tracker import Tracker


class TorchLogger:
    """Optional, import-light helpers for common PyTorch training-loop metrics.

    This module intentionally uses duck typing so importing ScalarPulse never imports
    PyTorch or changes CUDA initialization behavior.
    """

    def __init__(
        self,
        tracker: Tracker,
        *,
        model: Any | None = None,
        optimizer: Any | None = None,
        prefix: str = "train",
        track_grad_norm: bool = False,
    ) -> None:
        self.tracker = tracker
        self.model = model
        self.optimizer = optimizer
        self.prefix = prefix.strip("/")
        self.track_grad_norm = track_grad_norm

    def log(
        self,
        *,
        loss: Any | None = None,
        metrics: Mapping[str, Any] | None = None,
        step: int | None = None,
    ) -> None:
        values: dict[str, Any] = dict(metrics or {})
        if loss is not None:
            values.setdefault("loss", loss)
        if self.optimizer is not None:
            values.update(learning_rates(self.optimizer))
        if self.track_grad_norm and self.model is not None:
            values["grad_norm"] = gradient_norm(self.model)
        if not values:
            raise ValueError("TorchLogger.log requires loss, metrics, an optimizer, or gradient tracking")
        payload = {self.prefix: values} if self.prefix else values
        self.tracker.log(payload, step=step)


def learning_rates(optimizer: Any) -> dict[str, float]:
    groups = getattr(optimizer, "param_groups", None)
    if groups is None:
        raise TypeError("optimizer must expose param_groups")
    rates: dict[str, float] = {}
    for index, group in enumerate(groups):
        if "lr" not in group:
            continue
        key = "lr" if index == 0 else f"lr/group_{index}"
        rates[key] = float(group["lr"])
    return rates


def gradient_norm(model: Any, norm_type: float = 2.0) -> float:
    if math.isnan(norm_type) or norm_type <= 0:
        raise ValueError("norm_type must be positive or infinity")
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        raise TypeError("model must expose a parameters() method")

    norms: list[float] = []
    for parameter in parameters():
        gradient = getattr(parameter, "grad", None)
        if gradient is None:
            continue
        if hasattr(gradient, "detach"):
            gradient = gradient.detach()
        norm = gradient.norm(norm_type)
        if hasattr(norm, "item"):
            norm = norm.item()
        norms.append(float(norm))
    if not norms:
        return 0.0
    if math.isinf(norm_type):
        return max(norms)
    return sum(value**norm_type for value in norms) ** (1.0 / norm_type)
