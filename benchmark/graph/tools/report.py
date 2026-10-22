"""Render the separately validated matrix reporter from explicit report inputs."""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys


def reporting_directory():
    """Locate source in the checkout or a prepared snapshot/package launcher."""
    package = Path(__file__).resolve().parents[1]
    for runtime in (package, package.parent):
        directory = runtime / 'graph_local/reporting'
        if directory.is_dir():
            return directory
    raise FileNotFoundError('The graph_local/reporting package is missing')


def validate_protocols(studies):
    """Keep separate prompt/policy protocols out of a pooled matrix report."""
    common = None
    paths = ('graph_local/profile.py', 'graph_local/policy.py',
             'graph_local/backends.py', 'upstream/app/graph.py')
    for study in studies:
        signature = {name: hashlib.sha256((Path(study) / 'snapshot' / name).read_bytes()).hexdigest()
                     for name in paths}
        if common is None:
            common = signature
        elif signature != common:
            raise ValueError('Do not pool different graph protocols: profile/policy/backend/application bytes differ. '
                             'Render study-v2 and prompt-v3 as separate reports.')
    return common


def main():
    if not os.environ.get('SLURMD_NODENAME') or not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Report analysis and rendering require Slurm')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, action='append', required=True)
    parser.add_argument('--combined-report', type=Path, required=True,
        help='JSON produced by comparison.study_report from the full declared plans and raw runs')
    parser.add_argument('--report-script', type=Path,
        default=reporting_directory() / 'matrix_report.py')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    script = args.report_script.expanduser().resolve()
    if not script.is_file():
        raise FileNotFoundError('Pass --report-script with the validated matrix_report.py')
    validate_protocols([study.resolve() for study in args.study])
    command = [sys.executable, str(script), '--report', str(args.combined_report.resolve()),
        '--output', str(args.output.resolve())]
    for study in args.study:
        command += ['--study', str(study.resolve())]
    subprocess.run(command, check=True)
    appendix = reporting_directory() / 'reachability.py'
    if appendix.is_file():
        subprocess.run([sys.executable, str(appendix), '--matrix',
            str(args.output.resolve() / 'matrix.json'), '--output',
            str(args.output.resolve() / 'reachability')], check=True)


if __name__ == '__main__':
    main()
