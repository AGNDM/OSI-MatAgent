"""Portable, standard-library-only server for saved instrument histories."""
from __future__ import annotations

import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import socket
import sys
import tempfile
import urllib.error
import urllib.request
import webbrowser
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # Also works with Windows' isolated embedded Python.
sys.path.insert(0, str(ROOT / 'tools'))
from app import replay_viewer, result_viewer
import fetch_replay_assets

SERVICE = 'matagent-readonly-replay-v1'
SAMPLE_NAME = 'example-rc1mx-hydrolysis'
SAMPLE_FILES = {'result.json', 'manifest.json', 'monitor.jsonl',
                'selection-trace.jsonl', 'execution/trace.jsonl'}


class ReplayHTTPServer(ThreadingHTTPServer):
    # Windows SO_REUSEADDR can let two servers silently claim the same port.
    # Exclusive binding makes the fallback/reuse behavior deterministic.
    allow_reuse_address = False
    allow_reuse_port = False

    def server_bind(self):
        if hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def install_example(results: Path) -> bool:
    """Install a bundled example once, only when no existing histories are present."""
    if result_viewer.list_runs(results):
        return False
    archive = ROOT / 'examples/replay/rc1mx-hydrolysis.zip'
    if not archive.is_file():
        print('No bundled example found. Copy saved run folders into results/.')
        return False
    results.mkdir(parents=True, exist_ok=True)
    target = results / SAMPLE_NAME
    if target.exists():
        raise ValueError(f'Example directory already exists but is incomplete: {target}')
    with zipfile.ZipFile(archive) as bundle:
        entries = bundle.infolist()
        if (len(entries) != len(SAMPLE_FILES)
                or {item.filename for item in entries} != SAMPLE_FILES
                or sum(item.file_size for item in entries) > 80 * 1024 * 1024):
            raise ValueError('Unexpected example archive contents.')
        # Extract only known regular files into a temporary directory, then rename.
        with tempfile.TemporaryDirectory(prefix='.replay-example-', dir=results) as temp:
            staging = Path(temp) / SAMPLE_NAME
            staging.mkdir()
            for item in entries:
                destination = staging / item.filename
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(bundle.read(item))
            json.loads((staging / 'result.json').read_text(encoding='utf-8'))
            staging.rename(target)
    print('Installed the bundled, sanitized example recording.')
    return True


def prepare(results: Path):
    required = ['app/result_viewer.py', 'web/replay.html', 'web/replay.js',
                'web/replay-scene.js', 'web/replay.css', 'web/result-viewer.html',
                'web/result-viewer.js', 'web/result-viewer.css']
    required += ['web/vendor/three/' + name for name in replay_viewer.VENDOR_FILES]
    missing = [name for name in required if not (ROOT / name).is_file()]
    if missing:
        raise ValueError('Extract the entire project folder. Missing: ' + ', '.join(missing))
    fetch_replay_assets.main()
    install_example(results)
    results.mkdir(parents=True, exist_ok=True)


def instance_id(results: Path) -> str:
    return hashlib.sha256((str(ROOT.resolve()) + '\n' + str(results.resolve())).encode()).hexdigest()


def existing_server(port: int, identity: str) -> bool:
    try:
        # Loopback must not be sent through a machine's configured proxy.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(f'http://127.0.0.1:{port}/api/replay/health', timeout=0.4) as response:
            health = json.loads(response.read(4096))
        return health.get('service') == SERVICE and health.get('instance') == identity
    except (OSError, ValueError, urllib.error.URLError):
        return False


def handler_for(results: Path):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/api/replay/health':
                body = json.dumps({'service': SERVICE, 'instance': instance_id(results)}).encode()
                response = (200, 'application/json; charset=utf-8', body)
            else:
                response = replay_viewer.response_for(self.path, results, ROOT / 'web', ROOT)
                if response is None:
                    response = result_viewer.response_for(self.path, results, ROOT / 'web')
                if response is None:
                    response = (404, 'text/plain; charset=utf-8', b'Not found')
            status, mime, body = response
            self.send_response(status)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, format, *args):
            if len(args) >= 2 and str(args[1]) not in ('200', '304'):
                super().log_message(format, *args)

    return Handler


def open_page(port: int, no_browser: bool):
    url = f'http://127.0.0.1:{port}/replay'
    print(f'\nInstrument replay: {url}', flush=True)
    if not no_browser:
        try:
            if not webbrowser.open(url):
                print('Open the URL above in your browser.')
        except webbrowser.Error:
            print('Open the URL above in your browser.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8793)
    parser.add_argument('--no-browser', action='store_true')
    parser.add_argument('--check', action='store_true', help='Prepare files and exit without serving.')
    parser.add_argument('--results', type=Path, default=ROOT / 'results')
    args = parser.parse_args()
    if not 1024 <= args.port <= 65515:
        parser.error('--port must be between 1024 and 65515')
    results = args.results.resolve()
    prepare(results)
    print(f'Recordings: {results}')
    if args.check:
        print('Replay setup is ready. No API key or simulator environment is required.')
        return 0
    identity = instance_id(results)
    # Repeated double-clicks reuse this folder's server. Other services are untouched.
    for port in range(args.port, args.port + 21):
        if existing_server(port, identity):
            print('Reusing the running replay server.')
            open_page(port, args.no_browser)
            return 0
    for port in range(args.port, args.port + 21):
        try:
            server = ReplayHTTPServer(('127.0.0.1', port), handler_for(results))
            break
        except OSError:
            if existing_server(port, identity):
                open_page(port, args.no_browser)
                return 0
            continue
    else:
        raise RuntimeError('No available local port. Try --port 8893.')
    with server:
        open_page(port, args.no_browser)
        print('Keep this window open. Press Ctrl+C or close it to stop replay.\n'
              'This server reads saved files; it does not run experiments.', flush=True)
        try:
            server.serve_forever(poll_interval=0.3)
        except KeyboardInterrupt:
            print('\nReplay server stopped.')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as exc:
        print(f'\nCould not start replay: {exc}\n'
              'For first-time downloads, check your internet connection and try again.', file=sys.stderr)
        raise SystemExit(1)
