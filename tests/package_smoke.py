"""Run outside the source directory using a freshly installed wheel's Python."""
from importlib.metadata import version
from tempfile import TemporaryDirectory
import json
from urllib.request import urlopen

from scalarpulse import Tracker
from scalarpulse.server import start_dashboard


def main():
    with TemporaryDirectory(prefix="scalarpulse-package-smoke-") as root:
        with Tracker(log_dir=root, launch=False, quiet=True) as run:
            run.log({"loss": 0.5, "epoch": 1})
        handle = start_dashboard(root, port=0)
        try:
            with urlopen(handle.url + "/api/state", timeout=5) as response:
                state = json.load(response)
            assert state["records"][0]["metrics"] == {"loss": 0.5, "epoch": 1}
            assert state["run"]["status"] == "completed"
            with urlopen(handle.url, timeout=5) as response:
                page = response.read()
            assert b"ScalarPulse" in page
            print(f"PASS: scalarpulse {version('scalarpulse')}: logging, completion, HTTP state, bundled dashboard")
        finally:
            handle.stop()


if __name__ == "__main__":
    main()
