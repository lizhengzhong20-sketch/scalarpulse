from __future__ import annotations

import math
import numbers
import re
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any


_SAFE_ID = re.compile(r"[^a-zA-Z0-9_.-]+")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def safe_id(value: str, *, fallback: str = "run") -> str:
    cleaned = _SAFE_ID.sub("-", value.strip()).strip("-._")
    return cleaned[:80] or fallback


def to_scalar(value: Any, *, name: str) -> int | float:
    """Turn Python, NumPy, or framework scalar objects into JSON-safe numbers."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (ValueError, RuntimeError):
            pass

    if isinstance(value, bool):
        return int(value)
    if not isinstance(value, numbers.Real):
        raise TypeError(f"metric {name!r} must be a scalar number, got {type(value).__name__}")

    number = int(value) if isinstance(value, numbers.Integral) else float(value)
    if not math.isfinite(float(number)):
        raise ValueError(f"metric {name!r} must be finite, got {number!r}")
    return number


def flatten_metrics(metrics: Mapping[str, Any], prefix: str = "") -> dict[str, int | float]:
    flattened: dict[str, int | float] = {}
    for raw_name, value in metrics.items():
        name = str(raw_name).strip()
        if not name:
            raise ValueError("metric names cannot be empty")
        full_name = f"{prefix}/{name}" if prefix else name
        if isinstance(value, Mapping):
            flattened.update(flatten_metrics(value, full_name))
        else:
            flattened[full_name] = to_scalar(value, name=full_name)
    return flattened
