"""ScalarPulse: tiny, dependency-free live training dashboards."""

from .tracker import Tracker, init

__all__ = ["Tracker", "init", "__version__"]
__version__ = "0.1.0"
