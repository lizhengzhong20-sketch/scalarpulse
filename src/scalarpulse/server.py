from __future__ import annotations

import hashlib
import json
import queue
import threading
import time
import webbrowser
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .store import RunStore


class DashboardError(RuntimeError):
    pass


class EventBroker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: set[queue.Queue[tuple[str, dict[str, Any]]]] = set()

    def subscribe(self) -> queue.Queue[tuple[str, dict[str, Any]]]:
        events: queue.Queue[tuple[str, dict[str, Any]]] = queue.Queue(maxsize=1024)
        with self._lock:
            self._subscribers.add(events)
        return events

    def unsubscribe(self, events: queue.Queue[tuple[str, dict[str, Any]]]) -> None:
        with self._lock:
            self._subscribers.discard(events)

    def publish(self, event: str, data: dict[str, Any]) -> None:
        with self._lock:
            subscribers = tuple(self._subscribers)
        for events in subscribers:
            try:
                events.put_nowait((event, data))
            except queue.Full:
                try:
                    events.get_nowait()
                    events.put_nowait((event, data))
                except (queue.Empty, queue.Full):
                    pass

    def close(self) -> None:
        self.publish("__close__", {})


class LogWatcher(threading.Thread):
    def __init__(self, store: RunStore, broker: EventBroker, stop_event: threading.Event) -> None:
        super().__init__(name="scalarpulse-log-watcher", daemon=True)
        self.store = store
        self.broker = broker
        self.stop_event = stop_event
        self._positions: dict[Path, int] = {}
        self._metric_tails: dict[Path, bytes] = {}
        self._meta_signatures: dict[Path, bytes] = {}
        self._buffers: dict[Path, bytes] = {}
        self._snapshot_existing_files()

    def _snapshot_existing_files(self) -> None:
        for path in self.store.log_dir.glob("*/metrics.jsonl"):
            try:
                stat = path.stat()
                self._positions[path] = stat.st_size
                self._metric_tails[path] = self._tail_bytes(path, stat.st_size)
            except OSError:
                pass
        for path in self.store.log_dir.glob("*/meta.json"):
            try:
                self._meta_signatures[path] = self._file_signature(path)
            except OSError:
                pass

    def run(self) -> None:
        while not self.stop_event.wait(0.2):
            self._scan_metadata()
            self._scan_metrics()

    def _scan_metadata(self) -> None:
        for path in self.store.log_dir.glob("*/meta.json"):
            try:
                raw = path.read_bytes()
                signature = hashlib.blake2s(raw, digest_size=16).digest()
                if self._meta_signatures.get(path) == signature:
                    continue
                metadata = json.loads(raw.decode("utf-8"))
                self._meta_signatures[path] = signature
                self.broker.publish("run", metadata)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue

    def _scan_metrics(self) -> None:
        for path in self.store.log_dir.glob("*/metrics.jsonl"):
            try:
                position = self._positions.get(path, 0)
                stat = path.stat()
                size = stat.st_size
                current_tail = self._tail_bytes(path, size)
                previous_tail = self._metric_tails.get(path, b"")
                prefix_was_rewritten = (
                    position > 0
                    and size >= position
                    and self._tail_bytes(path, position) != previous_tail
                )
                if size < position or prefix_was_rewritten:
                    position = 0
                    self._buffers.pop(path, None)
                if size == position:
                    self._positions[path] = position
                    self._metric_tails[path] = current_tail
                    continue
                with path.open("rb") as stream:
                    stream.seek(position)
                    chunk = stream.read()
                    self._positions[path] = stream.tell()
                data = self._buffers.pop(path, b"") + chunk
                self._metric_tails[path] = current_tail
                lines = data.splitlines(keepends=True)
                if lines and not lines[-1].endswith((b"\n", b"\r")):
                    self._buffers[path] = lines.pop()
                for raw_line in lines:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        self.broker.publish("metric", json.loads(line.decode("utf-8")))
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        # One malformed line should not stop later valid live events.
                        continue
            except OSError:
                continue

    @staticmethod
    def _tail_bytes(path: Path, size: int, limit: int = 128) -> bytes:
        with path.open("rb") as stream:
            length = min(size, limit)
            stream.seek(size - length)
            return stream.read(length)

    @staticmethod
    def _file_signature(path: Path) -> bytes:
        return hashlib.blake2s(path.read_bytes(), digest_size=16).digest()


@dataclass
class DashboardHandle:
    server: _DashboardHTTPServer
    thread: threading.Thread
    watcher: LogWatcher
    stop_event: threading.Event
    url: str

    def stop(self) -> None:
        if self.stop_event.is_set():
            return
        self.stop_event.set()
        self.server.broker.close()
        self.server.shutdown()
        self.server.server_close()
        if self.thread.is_alive():
            self.thread.join(timeout=2)
        if self.watcher.is_alive():
            self.watcher.join(timeout=2)


class _DashboardHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        address: tuple[str, int],
        store: RunStore,
        broker: EventBroker,
        stop_event: threading.Event,
    ) -> None:
        self.store = store
        self.broker = broker
        self.stop_event = stop_event
        static_path = files("scalarpulse").joinpath("static/index.html")
        self.index_html = static_path.read_bytes()
        super().__init__(address, DashboardHandler)


class DashboardHandler(BaseHTTPRequestHandler):
    server: _DashboardHTTPServer
    protocol_version = "HTTP/1.1"

    def handle(self) -> None:
        try:
            super().handle()
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            # Normal when a browser closes an EventSource during shutdown.
            pass

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self._send_bytes(self.server.index_html, "text/html; charset=utf-8")
            return
        if parsed.path == "/api/health":
            self._send_json({"service": "scalarpulse", "log_dir": str(self.server.store.log_dir)})
            return
        if parsed.path == "/api/state":
            query = parse_qs(parsed.query)
            run_id = query.get("run_id", [None])[0]
            try:
                max_records = min(max(int(query.get("max_records", ["20000"])[0]), 1), 100_000)
            except ValueError:
                self._send_json({"error": "max_records must be an integer"}, HTTPStatus.BAD_REQUEST)
                return
            self._send_json(self.server.store.state(run_id, max_records=max_records))
            return
        if parsed.path == "/api/events":
            self._serve_events()
            return
        self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def _serve_events(self) -> None:
        events = self.server.broker.subscribe()
        self.close_connection = True
        try:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache, no-transform")
            # The response remains open while events stream, then closes on
            # shutdown instead of being reused for another HTTP request.
            self.send_header("Connection", "close")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            self.wfile.write(b"retry: 1000\n\n")
            self.wfile.flush()
            while not self.server.stop_event.is_set():
                try:
                    event, data = events.get(timeout=15)
                    if event == "__close__":
                        break
                    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
                    message = f"event: {event}\ndata: {payload}\n\n".encode("utf-8")
                except queue.Empty:
                    message = b": heartbeat\n\n"
                self.wfile.write(message)
                self.wfile.flush()
        except OSError:
            pass
        finally:
            self.close_connection = True
            self.server.broker.unsubscribe(events)

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self._send_bytes(body, "application/json; charset=utf-8", status)

    def _send_bytes(
        self,
        body: bytes,
        content_type: str,
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        self.close_connection = True
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Connection", "close")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
            "connect-src 'self'; img-src 'self' data:; object-src 'none'; base-uri 'none'; frame-ancestors 'none'",
        )
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        return


def start_dashboard(
    log_dir: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = False,
) -> DashboardHandle:
    store = RunStore(log_dir)
    broker = EventBroker()
    stop_event = threading.Event()
    try:
        server = _DashboardHTTPServer((host, port), store, broker, stop_event)
    except OSError as error:
        raise DashboardError(f"could not start dashboard on {host}:{port}: {error}") from error

    actual_host, actual_port = server.server_address[:2]
    browser_host = "127.0.0.1" if actual_host in ("0.0.0.0", "::") else actual_host
    url = f"http://{browser_host}:{actual_port}"
    watcher = LogWatcher(store, broker, stop_event)
    thread = threading.Thread(target=server.serve_forever, name="scalarpulse-http", daemon=True)
    thread.start()
    watcher.start()
    handle = DashboardHandle(server=server, thread=thread, watcher=watcher, stop_event=stop_event, url=url)
    if open_browser:
        threading.Thread(target=_open_browser, args=(url,), name="scalarpulse-browser", daemon=True).start()
    return handle


def _open_browser(url: str) -> None:
    time.sleep(0.25)
    webbrowser.open(url)
