"""Run the pinned llama.cpp server on loopback; Ctrl+C releases model memory."""
import argparse
from hashlib import sha256
import os
from pathlib import Path
import signal

ROOT = Path.home() / ".local/share/osi-hack"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--port", type=int, default=8792)
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--stop", action="store_true", help="Stop only the server started by this helper")
    args = parser.parse_args()
    binary = ROOT / "llama-b11147/llama-b11147/llama-server"
    pid_file = ROOT / "openjev-server.pid"
    if pid_file.exists():
        pid = int(pid_file.read_text())
        process = Path(f"/proc/{pid}")
        try:
            active = (process / "exe").resolve() == binary.resolve()
        except OSError:
            active = False
        if active:
            if args.stop:
                os.kill(pid, signal.SIGTERM)
                pid_file.unlink()
                print(f"Stopped OpenJev server PID {pid}")
                return
            parser.error(f"OpenJev is already running as PID {pid}; stop it before loading another copy")
        pid_file.unlink()
    if args.stop:
        print("No server started by this helper is running")
        return
    model = ROOT / "models/openjev-flash-9b-q4/OpenJev-Flash-9B-Q4_K_M.gguf"
    if not binary.is_file() or not model.is_file():
        parser.error("Pinned runtime or verified model is missing; complete setup first")
    digest = sha256()
    with model.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024**2), b""):
            digest.update(block)
    if digest.hexdigest() != "6209cf51aec005b00cb80fdfef1b961adde57d5a8f181a377bd3882a4668b91d":
        parser.error("Model SHA-256 differs from pinned OpenJev Flash 9B Q4 artifact")
    command = [str(binary), "--model", str(model), "--host", "127.0.0.1", "--port", str(args.port),
        "--alias", "openjev-flash-9b-q4", "--ctx-size", str(args.context), "--parallel", "1",
        "--threads", "8", "--threads-batch", "8", "--batch-size", "256", "--ubatch-size", "128",
        "--cache-ram", "256", "--reasoning", "off", "--chat-template-kwargs", '{"enable_thinking":false}',
        "--metrics", "--n-gpu-layers", "0" if args.cpu else "99",
        "--device", "none" if args.cpu else "Vulkan0"]
    print("Starting " + ("CPU" if args.cpu else "Radeon Vulkan") + " OpenJev at http://127.0.0.1:" + str(args.port), flush=True)
    pid_file.write_text(str(os.getpid()))
    os.execv(str(binary), command)


if __name__ == "__main__":
    main()
