from __future__ import annotations

import http.client
import json
import queue
import threading
import urllib.error
import urllib.request
from types import SimpleNamespace
from urllib.parse import urlparse

import pytest

from scalarpulse.server import DashboardHandler, EventBroker, LogWatcher, start_dashboard
from scalarpulse.store import RunStore


def _read_json(url: str) -> tuple[int, str, dict]:
    try:
        response = urllib.request.urlopen(url, timeout=3)
    except urllib.error.HTTPError as error:
        response = error
    with response:
        return response.status, response.headers.get_content_type(), json.load(response)


def test_http_health_and_state_return_json_and_honor_record_limit(tmp_path):
    store = RunStore(tmp_path)
    store.create_run({"id": "run", "name": "HTTP test", "started_at": "2026-01-01T00:00:00Z"})
    for step in range(5):
        store.append("run", {"run_id": "run", "seq": step, "step": step, "metrics": {"loss": step}})

    handle = start_dashboard(tmp_path, port=0)
    try:
        status, content_type, health = _read_json(f"{handle.url}/api/health")
        assert status == 200
        assert content_type == "application/json"
        assert health == {"service": "scalarpulse", "log_dir": str(tmp_path.resolve())}

        status, content_type, state = _read_json(f"{handle.url}/api/state?run_id=run&max_records=3")
        assert status == 200
        assert content_type == "application/json"
        assert state["selected_run_id"] == "run"
        assert [record["step"] for record in state["records"]] == [0, 2, 4]

        status, _, error = _read_json(f"{handle.url}/api/state?max_records=abc")
        assert status == 400
        assert error == {"error": "max_records must be an integer"}

        status, _, error = _read_json(f"{handle.url}/missing")
        assert status == 404
        assert error == {"error": "not found"}
    finally:
        handle.stop()

    assert not handle.thread.is_alive()
    assert not handle.watcher.is_alive()


def test_event_broker_broadcasts_and_drops_oldest_for_slow_subscribers():
    broker = EventBroker()
    first = broker.subscribe()
    second = broker.subscribe()

    broker.publish("metric", {"seq": 0})
    assert first.get_nowait() == ("metric", {"seq": 0})
    assert second.get_nowait() == ("metric", {"seq": 0})

    for seq in range(1025):
        broker.publish("metric", {"seq": seq})

    assert first.qsize() == 1024
    assert first.get_nowait() == ("metric", {"seq": 1})
    last = None
    while not first.empty():
        last = first.get_nowait()
    assert last == ("metric", {"seq": 1024})

    broker.unsubscribe(second)
    while not second.empty():
        second.get_nowait()
    broker.publish("run", {"id": "new"})
    with pytest.raises(queue.Empty):
        second.get_nowait()


def test_sse_frames_unicode_json_and_unsubscribes_after_disconnect():
    class StubBroker:
        def __init__(self):
            self.events = queue.Queue()
            self.events.put(("metric", {"name": "验证/loss", "value": 0.5}))
            self.unsubscribed = None

        def subscribe(self):
            return self.events

        def unsubscribe(self, events):
            self.unsubscribed = events

    class DisconnectingWriter:
        def __init__(self):
            self.writes = []

        def write(self, data):
            self.writes.append(data)
            if len(self.writes) == 2:
                raise BrokenPipeError

        def flush(self):
            pass

    broker = StubBroker()
    writer = DisconnectingWriter()
    handler = object.__new__(DashboardHandler)
    handler.server = SimpleNamespace(broker=broker, stop_event=threading.Event())
    handler.wfile = writer
    handler.send_response = lambda status: None
    handler.send_header = lambda name, value: None
    handler.end_headers = lambda: None

    handler._serve_events()

    assert writer.writes[0] == b"retry: 1000\n\n"
    message = writer.writes[1].decode("utf-8")
    assert message.startswith("event: metric\ndata: ")
    assert message.endswith("\n\n")
    payload = json.loads(message.split("data: ", 1)[1])
    assert payload == {"name": "验证/loss", "value": 0.5}
    assert broker.unsubscribed is broker.events


def test_sse_unsubscribes_when_client_disconnects_during_headers():
    broker = EventBroker()
    handler = object.__new__(DashboardHandler)
    handler.server = SimpleNamespace(broker=broker, stop_event=threading.Event())

    def disconnect_during_headers(status):
        raise ConnectionResetError

    handler.send_response = disconnect_during_headers

    handler._serve_events()

    assert not broker._subscribers


def test_log_watcher_publishes_only_new_metrics_and_metadata(tmp_path):
    store = RunStore(tmp_path)
    store.create_run({"id": "run", "name": "before"})
    store.append("run", {"run_id": "run", "seq": 0, "step": 0, "metrics": {"loss": 2}})
    broker = EventBroker()
    events = broker.subscribe()
    watcher = LogWatcher(store, broker, threading.Event())
    watcher._scan_metrics()
    with pytest.raises(queue.Empty):
        events.get_nowait()  # Existing history is loaded through /api/state, not replayed over SSE.

    metric = {"run_id": "run", "seq": 1, "step": 1, "metrics": {"loss": 1}}
    store.append("run", metric)
    watcher._scan_metrics()
    assert events.get_nowait() == ("metric", metric)

    updated = store.update_run("run", status="completed")
    watcher._scan_metadata()
    assert events.get_nowait() == ("run", updated)
    with pytest.raises(queue.Empty):
        events.get_nowait()


def test_log_watcher_waits_for_a_complete_jsonl_line_before_advancing(tmp_path):
    store = RunStore(tmp_path)
    store.create_run({"id": "run"})
    broker = EventBroker()
    events = broker.subscribe()
    watcher = LogWatcher(store, broker, threading.Event())
    path = tmp_path / "run" / "metrics.jsonl"
    record = {"run_id": "run", "seq": 0, "step": 0, "metrics": {"loss": 1}}
    encoded = json.dumps(record, separators=(",", ":"))

    with path.open("a", encoding="utf-8", newline="\n") as stream:
        midpoint = len(encoded) // 2
        stream.write(encoded[:midpoint])
        stream.flush()
    watcher._scan_metrics()
    with pytest.raises(queue.Empty):
        events.get_nowait()

    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded[midpoint:] + "\n")
        stream.flush()
    watcher._scan_metrics()

    assert events.get_nowait() == ("metric", record)
    with pytest.raises(queue.Empty):
        events.get_nowait()


def test_log_watcher_recovers_after_metrics_file_is_truncated(tmp_path):
    store = RunStore(tmp_path)
    store.create_run({"id": "run"})
    store.append("run", {"run_id": "run", "seq": 0, "step": 0, "metrics": {"loss": 2}})
    broker = EventBroker()
    events = broker.subscribe()
    watcher = LogWatcher(store, broker, threading.Event())
    replacement = {"run_id": "run", "seq": 1, "step": 1, "metrics": {"loss": 1}}
    path = tmp_path / "run" / "metrics.jsonl"
    path.write_text(json.dumps(replacement) + "\n", encoding="utf-8")

    watcher._scan_metrics()
    assert events.get_nowait() == ("metric", replacement)


def test_dashboard_stop_closes_an_active_sse_handler(tmp_path):
    handle = start_dashboard(tmp_path, port=0)
    parsed = urlparse(handle.url)
    connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=2)
    connection.request("GET", "/api/events")
    response = connection.getresponse()
    assert response.status == 200
    assert response.getheader("Content-Type").startswith("text/event-stream")

    stopped = threading.Event()

    def stop_server():
        handle.stop()
        stopped.set()

    stop_thread = threading.Thread(target=stop_server)
    stop_thread.start()
    stop_thread.join(timeout=2)
    body = response.read()
    connection.close()

    assert stopped.is_set()
    assert body.startswith(b"retry: 1000\n\n")
    assert not handle.thread.is_alive()
    assert not handle.watcher.is_alive()
