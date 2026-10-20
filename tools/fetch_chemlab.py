"""Fetch only pinned native runtimes, their tests, quickstart and license notices."""
from hashlib import sha256
import json
from pathlib import Path
from urllib.request import urlopen

REVISION = "123cf78593469871e41b8dbd63a11dfb7c5ca251"
DATA_REVISION = "4ca4a42dab2ca0a884e449d202e9ad96625d5648"
ROOT = Path(__file__).resolve().parents[1] / "vendor" / "chemlab"
FILES = [
    "scripts/asset_runtime/__init__.py",
    "scripts/asset_runtime/_annotations.py",
    "scripts/asset_runtime/binder_fd056.py",
    "scripts/asset_runtime/esco_efa5uduvw.py",
    "tests/test_asset_binder_fd056.py",
    "tests/test_asset_esco_efa5uduvw.py",
    "examples/quickstart_fume_hood.py",
    "LICENSE", "LICENSE-ASSETS.md", "NOTICE.md",
    "skills/chemistry-lab-assets/references/protocol.md",
]
DATA_FILES = [
    "assets/expansion_batch05/t4_measure/binder_fd056/instrument_model.json",
    "assets/expansion_batch05/t4_measure/binder_fd056/load_templates.json",
    "assets/expansion_batch05/t2_vacuum/esco_efa5uduvw/instrument_model.json",
    "assets/expansion_batch05/t2_vacuum/esco_efa5uduvw/vapour_templates.json",
]


def main():
    hashes = {}
    for name in FILES + DATA_FILES:
        if name in FILES:
            url = f"https://raw.githubusercontent.com/mbzuai-nlp/ChemLab-Assets/{REVISION}/{name}"
        else:
            url = f"https://huggingface.co/datasets/MBZUAI/ChemLab-Assets/resolve/{DATA_REVISION}/{name}"
        with urlopen(url, timeout=30) as response:
            data = response.read()
        path = ROOT / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        hashes[name] = sha256(data).hexdigest()
        print(name)
    (ROOT / "manifest.json").write_text(json.dumps({
        "repository": "https://github.com/mbzuai-nlp/ChemLab-Assets",
        "revision": REVISION, "sha256": hashes,
        "dataset": "https://huggingface.co/datasets/MBZUAI/ChemLab-Assets",
        "dataset_revision": DATA_REVISION,
        "scope": "Selected unmodified Python runtimes and four instrument JSON files; no geometry",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
