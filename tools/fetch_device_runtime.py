"""Verify or restore the exact official subset recorded by the expansion manifests.

No geometry, model inference, main-branch discovery or automatic package install.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
from urllib.request import urlopen

ROOT=Path(__file__).resolve().parents[1]/'vendor/chemlab'
CODE_REVISION='123cf78593469871e41b8dbd63a11dfb7c5ca251'
DATA_REVISION='4ca4a42dab2ca0a884e449d202e9ad96625d5648'


def entries():
    merged={}
    for name in ['manifest.json','expansion-code-manifest.json','expansion-data-manifest.json']:
        record=json.loads((ROOT/name).read_text())
        for rel,digest in record['sha256'].items():
            path=PurePosixPath(rel)
            if path.is_absolute() or '..' in path.parts:
                raise ValueError('Unsafe manifest path')
            if rel in merged and merged[rel]!=digest:
                raise ValueError('Conflicting frozen hashes for '+rel)
            merged[rel]=digest
    return merged


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--download',action='store_true',help='Restore missing or mismatched files; default verify only')
    args=p.parse_args()
    def verify(item):
        rel,digest=item;path=ROOT/rel
        if path.exists() and sha256(path.read_bytes()).hexdigest()==digest:return None
        if not args.download:return rel
        if rel.startswith('assets/'):
            url=f'https://huggingface.co/datasets/MBZUAI/ChemLab-Assets/resolve/{DATA_REVISION}/{rel}'
        else:
            url=f'https://raw.githubusercontent.com/mbzuai-nlp/ChemLab-Assets/{CODE_REVISION}/{rel}'
        with urlopen(url,timeout=60) as response:data=response.read()
        if sha256(data).hexdigest()!=digest:raise ValueError('Frozen hash mismatch: '+rel)
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        return None
    frozen=entries()
    with ThreadPoolExecutor(max_workers=6) as pool:
        failures=[r for r in pool.map(verify,frozen.items()) if r]
    print(json.dumps({'files':len(frozen),'mismatched':failures},indent=2))
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
