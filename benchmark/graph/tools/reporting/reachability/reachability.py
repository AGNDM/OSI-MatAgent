"""Compatibility entry point; use graph_local.reporting.reachability."""
from pathlib import Path
import sys

_package = Path(__file__).resolve().parents[3]
_runtime = _package if (_package / 'graph_local').is_dir() else _package.parent
sys.path.insert(0, str(_runtime))
from graph_local.reporting import reachability as _report


def __getattr__(name):
    return getattr(_report, name)


if __name__ == '__main__':
    raise SystemExit(_report.main())
