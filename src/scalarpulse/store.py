from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any
from collections import Counter

from ._util import safe_id, utc_now


class RunStore:
    """Small JSON/JSONL store designed for append-heavy scalar training logs."""

    def __init__(self, log_dir: str | os.PathLike[str]) -> None:
        self.log_dir = Path(log_dir).expanduser().resolve()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def _run_dir(self, run_id: str) -> Path:
        return self.log_dir / safe_id(run_id)

    def create_run(self, metadata: dict[str, Any]) -> dict[str, Any]:
        run_id = safe_id(str(metadata["id"]))
        run_dir = self._run_dir(run_id)
        with self._lock:
            run_dir.mkdir(parents=True, exist_ok=False)
            normalized = {
                "id": run_id,
                "project": str(metadata.get("project") or "default"),
                "name": str(metadata.get("name") or run_id),
                "started_at": metadata.get("started_at") or utc_now(),
                "ended_at": None,
                "status": "running",
                "framework": metadata.get("framework"),
                "tags": list(metadata.get("tags") or []),
                "config": dict(metadata.get("config") or {}),
                "summary": {},
            }
            self._write_json_atomic(run_dir / "meta.json", normalized)
            (run_dir / "metrics.jsonl").touch(exist_ok=True)
        return normalized

    def update_run(self, run_id: str, **updates: Any) -> dict[str, Any]:
        meta_path = self._run_dir(run_id) / "meta.json"
        with self._lock:
            metadata = self.read_run(run_id)
            metadata.update(updates)
            self._write_json_atomic(meta_path, metadata)
        return metadata

    def append(self, run_id: str, record: dict[str, Any]) -> None:
        path = self._run_dir(run_id) / "metrics.jsonl"
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        with self._lock, path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(line + "\n")
            stream.flush()

    def read_run(self, run_id: str) -> dict[str, Any]:
        path = self._run_dir(run_id) / "meta.json"
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream)

    def list_runs(self) -> list[dict[str, Any]]:
        runs: list[dict[str, Any]] = []
        if not self.log_dir.exists():
            return runs
        for meta_path in self.log_dir.glob("*/meta.json"):
            try:
                with meta_path.open("r", encoding="utf-8") as stream:
                    runs.append(json.load(stream))
            except (OSError, json.JSONDecodeError):
                continue
        return sorted(runs, key=lambda run: str(run.get("started_at", "")), reverse=True)

    def records(self, run_id: str) -> list[dict[str, Any]]:
        path = self._run_dir(run_id) / "metrics.jsonl"
        records: list[dict[str, Any]] = []
        if not path.exists():
            return records
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    # An interrupted final append is ignored; prior records remain readable.
                    continue
        return records

    def state(self, run_id: str | None = None, *, max_records: int = 20_000) -> dict[str, Any]:
        runs = self.list_runs()
        known_ids = {run["id"] for run in runs}
        selected_id = run_id if run_id in known_ids else (runs[0]["id"] if runs else None)
        selected = next((run for run in runs if run["id"] == selected_id), None)
        records = []
        if selected_id:
            if max_records <= 0:
                records = self.records(selected_id)
            else:
                path = self._run_dir(selected_id) / "metrics.jsonl"
                if path.exists():
                    with path.open("rb") as stream:
                        records = _sample_stream(stream, os.fstat(stream.fileno()).st_size, max_records)
        return {
            "runs": runs,
            "selected_run_id": selected_id,
            "run": selected,
            "records": records,
            "server_time": utc_now(),
        }

    @staticmethod
    def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
        temp_path = path.with_suffix(path.suffix + ".tmp")
        with temp_path.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temp_path.replace(path)


def _evenly_sample(records: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 1:
        return records[-1:]
    last = len(records) - 1
    indices = {round(index * last / (limit - 1)) for index in range(limit)}
    return [records[index] for index in sorted(indices)]


def _stream_records(stream, size: int):
    """Rewind a fixed byte snapshot, ignoring incomplete/non-object records."""
    stream.seek(0)
    while stream.tell() < size:
        line = stream.readline(size - stream.tell())
        if not line or not line.endswith(b"\n"):
            break
        try:
            record = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            continue
        if isinstance(record, dict):
            yield record


def _pick_positions(count: int, limit: int) -> set[int]:
    if count <= 0 or limit <= 0:
        return set()
    if count <= limit:
        return set(range(count))
    if limit == 1:
        return {count - 1}
    return {round(i * (count - 1) / (limit - 1)) for i in range(limit)}


def _metric_names(record):
    metrics = record.get("metrics")
    return metrics.keys() if isinstance(metrics, dict) else ()


def _sample_stream(stream, size: int, limit: int) -> list[dict[str, Any]]:
    """Three passes, O(limit + metric count) retained state; no full log list."""
    counts: Counter[str] = Counter()
    total = 0
    for record in _stream_records(stream, size):
        total += 1
        counts.update(_metric_names(record))
    if total <= limit:
        return list(_stream_records(stream, size))
    targets = {}
    if counts:
        quota, remainder = divmod(limit, len(counts))
        targets = {name: _pick_positions(counts[name], quota + (i < remainder))
                   for i, name in enumerate(sorted(counts))}
    selected: set[int] = set()
    occurrences: Counter[str] = Counter()
    for index, record in enumerate(_stream_records(stream, size)):
        for name in _metric_names(record):
            if occurrences[name] in targets.get(name, ()):
                selected.add(index)
            occurrences[name] += 1
    # Fill overlapping quotas using positions among unselected records.
    fillers = _pick_positions(total - len(selected), limit - len(selected))
    result = []
    remaining_index = 0
    for index, record in enumerate(_stream_records(stream, size)):
        if index in selected:
            result.append(record)
        else:
            if remaining_index in fillers:
                result.append(record)
            remaining_index += 1
    return result
