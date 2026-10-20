"""Load unmodified, pinned ChemLab modules without an editable upstream checkout."""
import sys
from pathlib import Path

VENDOR_ROOT = Path(__file__).resolve().parents[1] / "vendor" / "chemlab"
sys.path.insert(0, str(VENDOR_ROOT / "scripts"))

from asset_runtime import binder_fd056 as oven_runtime  # noqa: E402
