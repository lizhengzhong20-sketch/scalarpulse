from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor

import pytest

from scalarpulse import Tracker
from scalarpulse.store import RunStore, _evenly_sample


def test_tracker_writes_flattened_strict_jsonl_and_updates_summary(tmp_path):
    tracker = Tracker(
        log_dir=tmp_path,
        run_id="unicode-run",
        name="训练实验",
        config={"batch_size": 32},
        tags=["烟雾"],
        launch=False,
        quiet=True,
    )

    tracker.log({"loss": 1, "validation": {"accuracy": 0.75}, "enabled": True})
    tracker.log({"loss": 0.5}, step=4)
    tracker.log({"loss": 0.25})
    tracker.finish()

    metrics_path = tmp_path / tracker.id / "metrics.jsonl"
    raw_lines = metrics_path.read_text(encoding="utf-8").splitlines()
    assert len(raw_lines) == 3
    records = [json.loads(line) for line in raw_lines]
    assert [record["seq"] for record in records] == [0, 1, 2]
    assert [record["step"] for record in records] == [0, 4, 5]
    assert records[0]["metrics"] == {
        "loss": 1,
        "validation/accuracy": 0.75,
        "enabled": 1,
    }
    assert raw_lines[0].endswith("}")
    assert metrics_path.read_bytes().endswith(b"\n")

    metadata = tracker.store.read_run(tracker.id)
    assert metadata["name"] == "训练实验"
    assert metadata["tags"] == ["烟雾"]
    assert metadata["status"] == "completed"
    assert metadata["ended_at"].endswith("Z")
    assert metadata["summary"] == {
        "loss": 0.25,
        "validation/accuracy": 0.75,
        "enabled": 1,
    }


@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf])
def test_tracker_rejects_non_finite_values_without_writing_partial_line(tmp_path, bad_value):
    tracker = Tracker(log_dir=tmp_path, launch=False, quiet=True)

    with pytest.raises(ValueError, match="must be finite"):
        tracker.log({"loss": bad_value})

    metrics_path = tmp_path / tracker.id / "metrics.jsonl"
    assert metrics_path.read_bytes() == b""
    tracker.finish()


def test_tracker_rejects_invalid_metrics_steps_and_logging_after_finish(tmp_path):
    tracker = Tracker(log_dir=tmp_path, launch=False, quiet=True)

    with pytest.raises(ValueError, match="cannot be empty"):
        tracker.log({})
    with pytest.raises(ValueError, match="names cannot be empty"):
        tracker.log({"  ": 1})
    with pytest.raises(TypeError, match="must be a scalar"):
        tracker.log({"loss": [1, 2]})
    for step in (-1, True, 1.5):
        with pytest.raises(ValueError, match="non-negative integer"):
            tracker.log({"loss": 1}, step=step)

    tracker.finish("stopped")
    tracker.finish("completed")  # Repeated finish is deliberately idempotent.
    assert tracker.store.read_run(tracker.id)["status"] == "stopped"
    with pytest.raises(RuntimeError, match="finished run"):
        tracker.log({"loss": 1})


def test_tracker_context_manager_marks_failures_and_stops_dashboard(tmp_path):
    with pytest.raises(RuntimeError, match="training failed"):
        with Tracker(log_dir=tmp_path, launch=False, quiet=True) as tracker:
            tracker.log({"loss": 2.0})
            raise RuntimeError("training failed")

    metadata = tracker.store.read_run(tracker.id)
    assert metadata["status"] == "failed"
    assert metadata["summary"] == {"loss": 2.0}


def test_tracker_thread_safety_preserves_every_record_and_monotonic_sequences(tmp_path):
    tracker = Tracker(log_dir=tmp_path, launch=False, quiet=True)
    worker_count = 12
    records_per_worker = 50

    def write_worker(worker: int) -> None:
        for offset in range(records_per_worker):
            tracker.log({"worker": worker, "value": worker * records_per_worker + offset})

    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        list(pool.map(write_worker, range(worker_count)))
    tracker.finish()

    records = tracker.store.records(tracker.id)
    expected_count = worker_count * records_per_worker
    assert len(records) == expected_count
    assert [record["seq"] for record in records] == list(range(expected_count))
    assert [record["step"] for record in records] == list(range(expected_count))
    assert {record["metrics"]["value"] for record in records} == set(range(expected_count))


def test_store_state_selects_latest_run_and_evenly_samples_endpoints(tmp_path):
    store = RunStore(tmp_path)
    for run_id, started_at in (("old", "2026-01-01T00:00:00Z"), ("new", "2026-01-02T00:00:00Z")):
        store.create_run({"id": run_id, "started_at": started_at})
    for step in range(5):
        store.append("new", {"run_id": "new", "seq": step, "step": step, "metrics": {"loss": step}})

    state = store.state(max_records=3)
    assert state["selected_run_id"] == "new"
    assert [run["id"] for run in state["runs"]] == ["new", "old"]
    assert [record["step"] for record in state["records"]] == [0, 2, 4]
    assert state["server_time"].endswith("Z")

    fallback = store.state("does-not-exist", max_records=2)
    assert fallback["selected_run_id"] == "new"
    assert [record["step"] for record in fallback["records"]] == [0, 4]


def test_store_sampling_preserves_each_interleaved_metric_full_range(tmp_path):
    store = RunStore(tmp_path)
    store.create_run({"id": "interleaved"})
    for step in range(20):
        store.append("interleaved", {"step": step, "metrics": {"train/loss": step}})
        store.append("interleaved", {"step": step, "metrics": {"validation/loss": step + 0.5}})

    sampled = store.state("interleaved", max_records=20)["records"]
    steps_by_metric = {"train/loss": [], "validation/loss": []}
    for record in sampled:
        for name in record["metrics"]:
            steps_by_metric[name].append(record["step"])

    assert len(sampled) == 20
    assert steps_by_metric["train/loss"][0] == 0
    assert steps_by_metric["train/loss"][-1] == 19
    assert steps_by_metric["validation/loss"][0] == 0
    assert steps_by_metric["validation/loss"][-1] == 19


def test_store_records_keeps_valid_lines_around_a_corrupt_line(tmp_path):
    store = RunStore(tmp_path)
    store.create_run({"id": "run"})
    path = tmp_path / "run" / "metrics.jsonl"
    path.write_text(
        '{"seq":0,"step":0}\nnot-json\n{"seq":1,"step":1}\n',
        encoding="utf-8",
    )

    assert [record["seq"] for record in store.records("run")] == [0, 1]


@pytest.mark.parametrize(
    ("records", "limit", "expected"),
    [
        ([{"i": 0}, {"i": 1}], 1, [1]),
        ([{"i": 0}, {"i": 1}], 0, [1]),
        ([{"i": i} for i in range(10)], 4, [0, 3, 6, 9]),
    ],
)
def test_evenly_sample_is_deterministic_and_retains_endpoints(records, limit, expected):
    assert [record["i"] for record in _evenly_sample(records, limit)] == expected
