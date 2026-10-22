#!/usr/bin/env python3
"""Check a copied release, or refresh its packaging inventory as a maintainer.

This utility uses only the standard library. Run it inside Slurm on SCITAS.
``refresh`` cannot bless changed benchmark data, answer keys, splits, graders,
or native runtime files: their hashes must match the previous root manifest.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile


DEFAULT_BUNDLE = Path(__file__).resolve().parents[1]
if str(DEFAULT_BUNDLE) not in sys.path:
    sys.path.insert(0, str(DEFAULT_BUNDLE))
from eval.validate import _publishable_files, _relative_path, validate_bundle


_CORE_GRADERS = frozenset({
    "eval/common.py", "eval/mcqa.py", "eval/prompts.py", "eval/sequential.py",
})
_SETUP_FILES = frozenset({
    "sequential_120/runtime/requirements.txt",
    "sequential_120/runtime/requirements-model.txt",
})
_RELOCATIONS = {
    "sequential_120/provenance/source-mcqa-bundle-manifest.json":
        "mcqa_388/provenance/source-bundle-manifest.json",
    "sequential_120/provenance/source-mcqa-bundle-validation.json":
        "mcqa_388/provenance/source-bundle-validation.json",
}


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")


def _record(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _inventory(bundle):
    return {name: _record(_relative_path(bundle, name))
            for name in sorted(_publishable_files(bundle))}


def _protected(name):
    if name in _SETUP_FILES:
        return False
    if name in _CORE_GRADERS:
        return True
    if name.startswith("mcqa_388/"):
        return not name.endswith(".md")
    if name.startswith("sequential_120/runtime/vendor/"):
        return True
    if name.startswith("sequential_120/"):
        return name != "sequential_120/manifest.json" and not name.endswith(".md")
    return False


def _verify_frozen(bundle, previous, current):
    entries = previous.get("files")
    if not isinstance(entries, dict) or not entries:
        raise ValueError("A previous, nonempty root manifest is required")
    protected = {}
    for old_name, expected in entries.items():
        _relative_path(bundle, old_name)  # Reject unsafe baseline paths too.
        if _protected(old_name):
            name = _RELOCATIONS.get(old_name, old_name)
            if name in protected:
                raise ValueError(f"Duplicate frozen asset after relocation: {name}")
            protected[name] = expected
    actual_names = {name for name in current if _protected(name)}
    if actual_names != set(protected):
        missing = sorted(set(protected) - actual_names)
        added = sorted(actual_names - set(protected))
        raise ValueError(f"Frozen asset inventory changed; missing={missing}, added={added}")
    for name, expected in protected.items():
        if current[name] != expected:
            raise ValueError(f"Frozen asset changed: {name}; create a separately reviewed dataset/runtime release")
    return len(protected)


def _write_atomic(path, payload):
    mode = path.stat().st_mode & 0o777 if path.exists() else 0o644
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".release-", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
        temporary.chmod(mode)
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def check_bundle(bundle):
    """Authenticate both inventories and run the shipped static validators."""
    bundle = Path(bundle).resolve()
    if (bundle / "manifest.json").is_symlink():
        raise ValueError("Root manifest must be a regular file, not a symlink")
    report = validate_bundle(bundle)
    nested = bundle / "sequential_120/manifest.json"
    if nested.is_file() and _json(nested).get("files") != _inventory(nested.parent):
        raise ValueError("Sequential nested file inventory differs from its manifest")
    return {**report, "manifest_sha256": _record(bundle / "manifest.json")["sha256"],
            "nested_inventory_checked": nested.is_file()}


def refresh_bundle(bundle, *, version=None):
    """Refresh docs/tooling inventories while retaining frozen asset pins.

    Both manifest writes are rolled back if static validation fails. Historical
    validation receipts are left untouched; write current receipts under output/.
    """
    bundle = Path(bundle).resolve()
    manifest_path = bundle / "manifest.json"
    if manifest_path.is_symlink():
        raise ValueError("Root manifest must be a regular file, not a symlink")
    original = manifest_path.read_bytes()
    previous = json.loads(original)
    if version is not None and re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version) is None:
        raise ValueError("--version must have the form MAJOR.MINOR.PATCH")
    current = _inventory(bundle)
    frozen_count = _verify_frozen(bundle, previous, current)
    nested = bundle / "sequential_120/manifest.json"
    nested_original = None
    if nested.is_file():
        nested_original = nested.read_bytes()
        expected = previous["files"].get("sequential_120/manifest.json")
        if expected != _record(nested):
            raise ValueError("Nested manifest was edited; refresh accepts its previous authenticated contents only")
    try:
        if nested_original is not None:
            metadata = json.loads(nested_original)
            metadata["files"] = _inventory(nested.parent)
            _write_atomic(nested, _bytes(metadata))
        metadata = dict(previous)
        if version is not None:
            metadata["bundle_version"] = version
        metadata["created_at_utc"] = datetime.now(timezone.utc).isoformat()
        metadata["files"] = _inventory(bundle)
        _write_atomic(manifest_path, _bytes(metadata))
        report = check_bundle(bundle)
    except BaseException:
        if nested_original is not None:
            _write_atomic(nested, nested_original)
        _write_atomic(manifest_path, original)
        raise
    return {**report, "frozen_assets_verified": frozen_count,
            "operation": "packaging_inventory_refresh", "native_replay_performed": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "refresh"))
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--version", help="Maintainer packaging version; refresh only")
    parser.add_argument("--output", type=Path, help="Optional JSON receipt; use output/ inside the bundle")
    args = parser.parse_args(argv)
    try:
        if not os.environ.get("SLURMD_NODENAME") or not os.environ.get("SLURM_JOB_ID"):
            raise ValueError("Run inside a real Slurm compute allocation")
        if args.command == "check" and args.version is not None:
            raise ValueError("--version is only supported by refresh")
        if args.output is not None:
            target = args.output.resolve()
            bundle = args.bundle.resolve()
            if target.is_relative_to(bundle) and not target.is_relative_to(bundle / "output"):
                raise ValueError("Receipts inside the bundle must be written under output/")
        report = (check_bundle(args.bundle) if args.command == "check"
                  else refresh_bundle(args.bundle, version=args.version))
        report.update({"slurm_job_id": os.environ["SLURM_JOB_ID"],
                       "host": os.environ["SLURMD_NODENAME"]})
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            _write_atomic(args.output, _bytes(report))
        print(json.dumps(report, sort_keys=True))
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
