"""Install the pinned official llama.cpp Vulkan binary without system package changes."""
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path.home() / ".local/share/osi-hack"
ARCHIVE_SHA = "0a29372770497ec5450be009a09f21284c578944bb366979d7636ae54a2e4871"
URL = "https://github.com/ggml-org/llama.cpp/releases/download/b11147/llama-b11147-bin-ubuntu-vulkan-x64.tar.gz"


def main():
    archive = ROOT / "downloads/llama-b11147-vulkan.tar.gz"
    destination = ROOT / "llama-b11147"
    archive.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        subprocess.run(["curl", "-L", "--fail", "--retry", "2", "--max-time", "300",
                        URL, "-o", str(archive)], check=True)
    if sha256(archive.read_bytes()).hexdigest() != ARCHIVE_SHA:
        raise RuntimeError("Official runtime archive checksum mismatch")
    binary = destination / "llama-b11147/llama-server"
    if not binary.exists():
        with tarfile.open(archive) as bundle:
            bundle.extractall(destination, filter="data")
    result = subprocess.run([str(binary), "--version"], capture_output=True, text=True, check=True)
    manifest = {"release": "b11147", "commit": "fee39dd92", "source": URL,
        "archive_sha256": ARCHIVE_SHA, "binary": str(binary), "version": (result.stdout + result.stderr).strip(),
        "backends": ["Vulkan", "CPU"], "network_binding": "127.0.0.1 only"}
    (destination / "deployment-runtime.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
