"""Run package checks after activating the real immutable native benchmark."""
import argparse
import importlib
import os
from pathlib import Path
import sys


def main():
    if not os.environ.get('SLURMD_NODENAME') or not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Tests require a genuine Slurm allocation')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--app-source', type=Path, required=True)
    parser.add_argument('--comparison-source', type=Path, required=True)
    parser.add_argument('--junitxml', type=Path)
    parser.add_argument('--portability-only', action='store_true',
        help='Run portability and bundled reporter checks; omit previously tested graph-core checks')
    args = parser.parse_args()
    package = Path(__file__).resolve().parents[1]
    runtime = package if (package / 'graph_local').is_dir() else package.parent
    sys.path[:0] = [str(runtime), str(runtime / 'tests'),
        str(args.comparison_source.resolve()), str(args.app_source.resolve())]
    os.environ['GRAPH_PACKAGE_TEST_COMPARISON'] = str(args.comparison_source.resolve())
    os.environ['GRAPH_LOCAL_BUNDLE'] = str(args.comparison_source.resolve() / 'instrument')
    from comparison.policies import load_bundle
    helpers = load_bundle(args.comparison_source.resolve() / 'instrument')
    helpers['validate'].validate_bundle(args.comparison_source.resolve() / 'instrument')
    helpers['sequential']._runtime()
    for name in ('asset_runtime', 'asset_runtime.binder_fd056', 'basic40_runtime', 'basic40_runtime_thermal'):
        importlib.import_module(name)
    import pytest
    target = runtime / 'tests/test_portability.py' if args.portability_only else runtime / 'tests'
    command = ['-q', '--tb=short', '-p', 'no:cacheprovider', str(target)]
    if args.portability_only:
        command.extend(str(runtime / 'tests' / name) for name in
            ('test_matrix_report.py', 'test_reachability.py'))
    if args.junitxml:
        args.junitxml.parent.mkdir(parents=True, exist_ok=True)
        command += ['--junitxml', str(args.junitxml)]
    raise SystemExit(pytest.main(command))


if __name__ == '__main__':
    main()
