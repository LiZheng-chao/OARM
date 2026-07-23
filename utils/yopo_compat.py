import os
import sys


def ensure_yopo_path() -> str:
    """Expose YOPO's config/policy modules without modifying the baseline tree."""

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    yopo_dir = os.path.join(repo_root, "YOPO")
    if yopo_dir in sys.path:
        sys.path.remove(yopo_dir)
    sys.path.insert(0, yopo_dir)
    return yopo_dir


ensure_yopo_path()

