"""Create an isolated graph layer; keep the inference environment read-only."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess


def main():
    if not os.environ.get('SLURMD_NODENAME') or not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Installation requires a real Slurm compute allocation')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-python', type=Path, required=True)
    parser.add_argument('--environment', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--app-source', type=Path, required=True)
    parser.add_argument('--comparison-source', type=Path, required=True)
    parser.add_argument('--uv', default=shutil.which('uv'))
    args = parser.parse_args()
    if not args.uv:
        raise ValueError('Pass --uv with the local uv executable')
    base = args.base_python.expanduser().absolute()
    environment = args.environment.expanduser().absolute()
    if environment.exists():
        raise FileExistsError('Refusing to mutate an existing environment: ' + str(environment))
    package = Path(__file__).resolve().parents[1]
    site = subprocess.check_output([str(base), '-c',
        'import sysconfig; print(sysconfig.get_paths()["purelib"])'], text=True).strip()
    subprocess.run([str(base), '-m', 'venv', '--without-pip', str(environment)], check=True)
    python = environment / 'bin/python'
    own_site = Path(subprocess.check_output([str(python), '-c',
        'import sysconfig; print(sysconfig.get_paths()["purelib"])'], text=True).strip())
    (own_site / 'frozen-inference-stack.pth').write_text(site + '\n')
    subprocess.run([str(args.uv), 'pip', 'install', '--python', str(python), '--no-deps',
        '-r', str(package / 'requirements-graph.txt')], check=True)
    subprocess.run([str(args.uv), 'pip', 'check', '--python', str(python)], check=True)
    subprocess.run([str(python), str(package / 'tools/environment.py'),
        '--requirements', str(package / 'requirements-graph.txt'),
        '--app-source', str(args.app_source), '--comparison-source', str(args.comparison_source),
        '--output', str(args.receipt), '--base-site-packages', site], check=True)


if __name__ == '__main__':
    main()
