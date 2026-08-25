from __future__ import annotations

import secrets
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ._util import flatten_metrics, safe_id, utc_now
from .server import DashboardError, DashboardHandle, start_dashboard
from .store import RunStore


class Tracker:
    """Log scalar metrics and optionally host a live local dashboard."""

    def __init__(
        self,
        *,
        project: str = "default",
        name: str | None = None,
        config: Mapping[str, Any] | None = None,
        tags: Sequence[str] | None = None,
        framework: str | None = None,
        log_dir: str | Path = ".scalarpulse",
        run_id: str | None = None,
        launch: bool = True,
        host: str = "127.0.0.1",
        port: int = 8765,
        open_browser: bool = False,
        quiet: bool = False,
    ) -> None:
        self.store = RunStore(log_dir)
        self._lock = threading.RLock()
        self._step = 0
        self._seq = 0
        self._closed = False
        self._summary: dict[str, int | float] = {}
        self._dashboard: DashboardHandle | None = None
        self._dashboard_error: DashboardError | None = None

        self.id = safe_id(run_id or _new_run_id())
        self.name = name or self.id
        self.project = project
        self.metadata = self.store.create_run(
            {
                "id": self.id,
                "project": project,
                "name": self.name,
                "started_at": utc_now(),
                "framework": framework,
                "tags": list(tags or []),
                "config": dict(config or {}),
            }
        )
        if launch:
            try:
                self._dashboard = start_dashboard(
                    self.store.log_dir,
                    host=host,
                    port=port,
                    open_browser=open_browser,
                )
            except DashboardError as error:
                # Metric logging remains useful when the dashboard port is unavailable.
                self._dashboard_error = error
                self._dashboard = None
        if not quiet:
            if self.url:
                print(f"ScalarPulse run {self.name!r} -> {self.url}")
            elif self._dashboard_error:
                print(f"ScalarPulse run {self.name!r} is logging to {self.store.log_dir}")
                print(f"Dashboard was not started: {self._dashboard_error}")
            else:
                print(f"ScalarPulse run {self.name!r} is logging to {self.store.log_dir} (dashboard disabled)")

    @property
    def url(self) -> str | None:
        return self._dashboard.url if self._dashboard else None

    def log(self, metrics: Mapping[str, Any], *, step: int | None = None) -> None:
        flattened = flatten_metrics(metrics)
        if not flattened:
            raise ValueError("metrics cannot be empty")
        with self._lock:
            self._ensure_open()
            actual_step = self._step if step is None else _validate_step(step)
            if step is None:
                self._step += 1
            else:
                self._step = max(self._step, actual_step + 1)
            record = {
                "run_id": self.id,
                "seq": self._seq,
                "step": actual_step,
                "time": time.time(),
                "metrics": flattened,
            }
            self._seq += 1
            self._summary.update(flattened)
            self.store.append(self.id, record)

    def finish(self, status: str = "completed") -> None:
        allowed = {"completed", "failed", "stopped"}
        if status not in allowed:
            raise ValueError(f"status must be one of {sorted(allowed)}, got {status!r}")
        with self._lock:
            if self._closed:
                return
            self.metadata = self.store.update_run(
                self.id,
                status=status,
                ended_at=utc_now(),
                summary=dict(self._summary),
            )
            self._closed = True

    def stop_dashboard(self) -> None:
        if self._dashboard is not None:
            self._dashboard.stop()
            self._dashboard = None

    def __enter__(self) -> "Tracker":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.finish("failed" if exc_type else "completed")

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("cannot log to a finished run")


def init(**kwargs: Any) -> Tracker:
    """Create a :class:`Tracker`; mirrors the familiar ``scalarpulse.init(...)`` style."""
    return Tracker(**kwargs)


def _new_run_id() -> str:
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    return f"{timestamp}-{secrets.token_hex(3)}"


def _validate_step(step: Any) -> int:
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError(f"step must be a non-negative integer, got {step!r}")
    return step
