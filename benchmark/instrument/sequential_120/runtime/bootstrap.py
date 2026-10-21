"""Activate the bundled modules without consulting the original checkout."""
from pathlib import Path
import sys


PACKAGES = (
    "instrument_bench", "stepwise_bench", "stepwise_expanded", "stepwise_choice",
    "asset_runtime", "basic40_runtime", "basic40_runtime_thermal",
)


def activate():
    """Use this release's exact native evaluator modules; reject mixed imports."""
    vendor = Path(__file__).resolve().parent / "vendor"
    if not vendor.is_dir():
        raise RuntimeError("The sequential runtime bundle is missing its vendor directory")
    for name, module in tuple(sys.modules.items()):
        if name.split(".", 1)[0] not in PACKAGES:
            continue
        location = getattr(module, "__file__", None)
        if location is not None and not Path(location).resolve().is_relative_to(vendor):
            raise RuntimeError(f"Module {name} was already loaded outside the bundled runtime: {location}")
    path = str(vendor)
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)
    return vendor
