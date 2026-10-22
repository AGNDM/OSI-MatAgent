"""Fetch the pinned, small RC1mx GLB used by the read-only animation page."""
import hashlib
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
RELATIVE = 'vendor/chemlab/assets/expansion_batch04/t1_thermal/mt_rc1mxhfcal/model.glb'
URL = ('https://huggingface.co/datasets/MBZUAI/ChemLab-Assets/resolve/'
       '4ca4a42dab2ca0a884e449d202e9ad96625d5648/' + RELATIVE.removeprefix('vendor/chemlab/'))
SHA256 = 'c98306ecd292289224e1508c00af741595b51e52ea6c917b6616a1f8a8952335'


def main():
    target = ROOT / RELATIVE
    if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == SHA256:
        print('RC1mx replay model is present and verified.')
        return
    print('Downloading pinned RC1mx model (2.6 MB)…')
    with urllib.request.urlopen(URL, timeout=60) as response:
        data = response.read(10 * 1024 * 1024)
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise ValueError('Model checksum mismatch; existing file was not changed.')
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix('.download')
    temporary.write_bytes(data)
    temporary.replace(target)
    print('Verified and saved:', target)


if __name__ == '__main__':
    main()
