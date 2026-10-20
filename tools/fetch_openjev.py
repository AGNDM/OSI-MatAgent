"""Download one pinned, checksummed OpenJev Flash 9B Q4 artifact (no full HF cache copy)."""
from hashlib import sha256
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import json
from pathlib import Path
import shutil
import subprocess
from urllib.request import urlopen

REPOSITORY = "openjev/OpenJev-Flash-9B-GGUF"
REVISION = "a3acd96966b06c2f161209f888b66235029f9ac1"
FILENAME = "OpenJev-Flash-9B-Q4_K_M.gguf"
EXPECTED_SHA256 = "6209cf51aec005b00cb80fdfef1b961adde57d5a8f181a377bd3882a4668b91d"
EXPECTED_BYTES = 5629109120
MODEL_DIR = Path.home() / ".local/share/osi-hack/models/openjev-flash-9b-q4"


def checksum(path):
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_ranges(url, partial):
    """Bound each response to its requested size; retain verified chunks on restart."""
    plan_path = partial.with_suffix(".ranges.json")
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
    else:
        prefix = partial.stat().st_size if partial.exists() else 0
        block = 64 * 1024**2
        plan = {"prefix_bytes": prefix, "ranges": [
            [start, min(start + block, EXPECTED_BYTES) - 1]
            for start in range(prefix, EXPECTED_BYTES, block)]}
        plan_path.write_text(json.dumps(plan))

    def fetch(item):
        index, (start, end) = item
        chunk = partial.with_suffix(f".chunk-{index:03}")
        size = end - start + 1
        if chunk.exists() and chunk.stat().st_size == size:
            return index, chunk
        subprocess.run(["curl", "--location", "--fail", "--range", f"{start}-{end}",
            "--max-filesize", str(size), "--retry", "3", "--connect-timeout", "20",
            "--max-time", "300", "--silent", "--show-error", url, "--output", str(chunk)], check=True)
        if chunk.stat().st_size != size:
            raise RuntimeError(f"Range {index} has wrong size")
        print(f"Downloaded range {index + 1}/{len(plan['ranges'])}", flush=True)
        return index, chunk

    if not partial.exists():
        partial.touch()
    # An interrupted merge is rolled back to its original prefix, then replayed.
    with partial.open("r+b") as stream:
        stream.truncate(plan["prefix_bytes"])
    chunks = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for future in as_completed([pool.submit(fetch, item) for item in enumerate(plan["ranges"])]):
            index, chunk = future.result()
            chunks[index] = chunk
    with partial.open("ab") as output:
        for index in sorted(chunks):
            with chunks[index].open("rb") as source:
                shutil.copyfileobj(source, output, 8 * 1024**2)
    if checksum(partial) != EXPECTED_SHA256:
        raise RuntimeError("Merged download failed SHA-256; source chunks retained")
    for chunk in chunks.values():
        chunk.unlink()
    plan_path.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xet", action="store_true", help="Use installed huggingface_hub/hf_xet")
    args = parser.parse_args()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / FILENAME
    partial = path.with_suffix(".gguf.part")
    if not path.exists():
        existing = partial.stat().st_size if partial.exists() else 0
        if shutil.disk_usage(MODEL_DIR).free < 2 * EXPECTED_BYTES - existing + 2 * 1024**3:
            raise RuntimeError("Insufficient disk space including a 2 GiB reserve")
        url = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{FILENAME}?download=true"
        if args.xet:
            from huggingface_hub import hf_hub_download
            hf_hub_download(REPOSITORY, FILENAME, revision=REVISION, local_dir=str(MODEL_DIR), token=False)
            if path.stat().st_size != EXPECTED_BYTES or checksum(path) != EXPECTED_SHA256:
                raise RuntimeError("Downloaded Xet model failed integrity check")
        else:
            download_ranges(url, partial)
            if partial.stat().st_size != EXPECTED_BYTES or checksum(partial) != EXPECTED_SHA256:
                raise RuntimeError("Model size/checksum mismatch; partial artifact retained for inspection")
            partial.rename(path)
    elif path.stat().st_size != EXPECTED_BYTES or checksum(path) != EXPECTED_SHA256:
        raise RuntimeError("Existing model failed integrity check")
    # Only remove this installer's alternate transport files after final integrity verification.
    for pattern in [FILENAME + ".part", FILENAME + ".chunk-*", "OpenJev-Flash-9B-Q4_K_M.gguf.ranges.json"]:
        for extra in MODEL_DIR.glob(pattern):
            extra.unlink()
    for name in ["README.md", "LICENSE", "LICENSE-APACHE-2.0", "NOTICE", "MANIFEST.json", "SHA256SUMS"]:
        with urlopen(f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{name}", timeout=30) as response:
            (MODEL_DIR / name).write_bytes(response.read())
    manifest = {"repository": REPOSITORY, "revision": REVISION, "file": FILENAME,
        "bytes": EXPECTED_BYTES, "sha256": EXPECTED_SHA256, "quantization": "Q4_K_M",
        "license": "CC-BY-NC-4.0", "source_model": "openjev/OpenJev-Flash-9B",
        "different_from": "ejhshen/OpenJev-4B", "path": str(path),
        "integrity_verified": True}
    (MODEL_DIR / "deployment-model.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
