from __future__ import annotations

import argparse
import math
import random
import signal
import sys
import time
from pathlib import Path

from . import __version__
from .server import DashboardError, start_dashboard
from .tracker import Tracker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="scalarpulse", description="Tiny live dashboards for model training")
    parser.add_argument("--version", action="version", version=f"scalarpulse {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve", help="serve an existing ScalarPulse log directory")
    _add_server_arguments(serve)

    demo = subparsers.add_parser("demo", help="stream a synthetic training run")
    _add_server_arguments(demo)
    demo.add_argument("--steps", type=int, default=160)
    demo.add_argument("--interval", type=float, default=0.08, help="seconds between steps")
    return parser


def _add_server_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--log-dir", type=Path, default=Path(".scalarpulse"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", dest="open_browser")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "serve":
        return _serve(args)
    if args.command == "demo":
        return _demo(args)
    return 2


def _serve(args: argparse.Namespace) -> int:
    try:
        handle = start_dashboard(
            args.log_dir,
            host=args.host,
            port=args.port,
            open_browser=args.open_browser,
        )
    except DashboardError as error:
        print(f"scalarpulse: {error}", file=sys.stderr)
        return 1
    print(f"ScalarPulse dashboard -> {handle.url}")
    print(f"Watching {Path(args.log_dir).expanduser().resolve()} (Ctrl+C to stop)")
    try:
        signal.pause() if hasattr(signal, "pause") else _wait_forever()
    except KeyboardInterrupt:
        pass
    finally:
        handle.stop()
    return 0


def _demo(args: argparse.Namespace) -> int:
    if args.steps <= 0 or args.interval < 0:
        print("scalarpulse: --steps must be positive and --interval cannot be negative", file=sys.stderr)
        return 2
    run = Tracker(
        project="demo",
        name="synthetic-classifier",
        config={"optimizer": "AdamW", "batch_size": 64, "learning_rate": 0.003},
        tags=["demo"],
        framework="pytorch",
        log_dir=args.log_dir,
        launch=True,
        host=args.host,
        port=args.port,
        open_browser=args.open_browser,
    )
    randomizer = random.Random(7)
    try:
        for step in range(args.steps):
            progress = step / max(args.steps - 1, 1)
            train_loss = 2.4 * math.exp(-4.2 * progress) + 0.08 + randomizer.uniform(-0.035, 0.035)
            val_loss = 2.5 * math.exp(-3.7 * progress) + 0.11 + randomizer.uniform(-0.025, 0.025)
            accuracy = min(0.985, 0.24 + 0.76 * (1 - math.exp(-4.5 * progress)) + randomizer.uniform(-0.008, 0.008))
            lr = 0.003 * 0.5 * (1 + math.cos(math.pi * progress))
            run.log(
                {
                    "train": {"loss": max(train_loss, 0.01), "accuracy": max(min(accuracy, 1), 0)},
                    "validation": {"loss": max(val_loss, 0.01)},
                    "learning_rate": lr,
                    "throughput": 760 + randomizer.uniform(-35, 35),
                },
                step=step,
            )
            time.sleep(args.interval)
        run.finish()
        print("Demo completed. Press Ctrl+C to stop the dashboard.")
        _wait_forever()
    except KeyboardInterrupt:
        run.finish("stopped")
    finally:
        run.stop_dashboard()
    return 0


def _wait_forever() -> None:
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    raise SystemExit(main())
