"""Resolve only explicit ${NAME} path bindings in a portable experiment recipe."""
import argparse
import json
import os
from pathlib import Path
import re

PLACEHOLDER = re.compile(r'^\$\{([A-Z][A-Z0-9_]*)\}$')


def render(value, bindings):
    if isinstance(value, dict):
        return {key: render(item, bindings) for key, item in value.items()}
    if isinstance(value, list):
        return [render(item, bindings) for item in value]
    if isinstance(value, str) and (match := PLACEHOLDER.fullmatch(value)):
        supplied = bindings[match[1]]
        if not isinstance(supplied, str) or not Path(supplied).is_absolute() or '${' in supplied:
            raise ValueError('Binding must be an explicit absolute path: ' + match[1])
        return supplied
    if isinstance(value, str) and '${' in value:
        raise ValueError('Only complete explicit path placeholders are supported')
    return value


def main():
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURMD_NODENAME'):
        raise RuntimeError('A genuine Slurm compute allocation is required')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--bindings', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = render(json.loads(args.plan.read_text()), json.loads(args.bindings.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
