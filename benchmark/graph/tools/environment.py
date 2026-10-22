"""Verify the full import-visible dependency closure and actual graph import."""
import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys


def main():
    if not os.environ.get('SLURMD_NODENAME') or not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Environment validation requires Slurm')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--requirements', type=Path, required=True)
    parser.add_argument('--app-source', type=Path, required=True)
    parser.add_argument('--comparison-source', type=Path, required=True)
    parser.add_argument('--base-site-packages', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
    packages = {}
    for distribution in importlib.metadata.distributions():
        packages.setdefault(canonicalize_name(distribution.metadata['Name']), distribution)
    failures = []
    for name, distribution in packages.items():
        for value in distribution.requires or []:
            requirement = Requirement(value)
            if requirement.marker and not requirement.marker.evaluate({'extra': ''}):
                continue
            installed = packages.get(canonicalize_name(requirement.name))
            if installed is None or (requirement.specifier and installed.version not in requirement.specifier):
                failures.append((name, value, installed.version if installed else None))
    if failures:
        raise ValueError('Incompatible import-visible dependencies: ' + repr(failures))
    for line in args.requirements.read_text().splitlines():
        if line and not line.startswith('#'):
            name, version = line.split('==')
            if importlib.metadata.version(name) != version:
                raise ValueError('Pinned graph dependency differs: ' + line)
    expected = {'torch': '2.8.0', 'transformers': '4.57.1', 'sentence-transformers': '5.1.1',
        'scikit-learn': '1.7.2', 'numpy': '2.3.5'}
    for name, version in expected.items():
        actual = importlib.metadata.version(name)
        if actual.split('+')[0] != version:
            raise ValueError(f'Frozen model stack requires {name}=={version}; got {actual}')
    comparison_source = args.comparison_source.resolve()
    app_source = args.app_source.resolve()
    sys.path[:0] = [str(comparison_source), str(app_source)]
    from comparison.policies import load_bundle
    helpers = load_bundle(comparison_source / 'instrument')
    helpers['validate'].validate_bundle(comparison_source / 'instrument')
    helpers['sequential']._runtime()
    for name in ('asset_runtime', 'asset_runtime.binder_fd056', 'basic40_runtime', 'basic40_runtime_thermal'):
        importlib.import_module(name)
    from app.graph import build_graph
    from app.trace import Trace
    graph = build_graph(object(), Trace('portable-environment-import'))
    importlib.import_module('sequential_120.runtime.bootstrap').activate()
    receipt = {'status': 'validated', 'slurm_job_id': os.environ['SLURM_JOB_ID'],
        'compute_node': os.environ['SLURMD_NODENAME'], 'python': sys.executable,
        'frozen_base_site_packages': str(args.base_site_packages.resolve()),
        'full_import_visible_dependency_check_passed': True,
        'packages': {name: distribution.version for name, distribution in sorted(packages.items())},
        'graph_requirements_sha256': hashlib.sha256(args.requirements.read_bytes()).hexdigest(),
        'application_graph_sha256': hashlib.sha256((app_source / 'app/graph.py').read_bytes()).hexdigest(),
        'real_application_graph_constructed': graph is not None, 'inference_performed': False,
        'hosted_inference_calls': 0, 'base_environment_modified': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'status': receipt['status'], 'output': str(args.output)}), flush=True)


if __name__ == '__main__':
    main()
