#!/usr/bin/env python3
"""Restore the optional whole-brain benchmark payload from pinned upstream data."""
import argparse
import hashlib
import shutil
from pathlib import Path
import tempfile
import urllib.request

URL = ('https://raw.githubusercontent.com/pnlbwh/ukftractography/'
       '2d2b661539700aa825c1949823119601e11e1823/UKFTractography/Data/Input/dwi.raw.gz')
SHA256 = '9c89aaeb8984b08d025edd229057e18ea91f8205dceafe52f05fa549aa8f04f3'


def download(output):
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        digest = hashlib.sha256()
        with urllib.request.urlopen(URL, timeout=60) as response:
            with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as target:
                temporary = Path(target.name)
                while chunk := response.read(1024 * 1024):
                    digest.update(chunk)
                    target.write(chunk)
        if digest.hexdigest() != SHA256:
            raise ValueError('Benchmark SHA-256 verification failed')
        # Exclusive creation also protects against an output created during download.
        with output.open('xb') as target, temporary.open('rb') as source:
            shutil.copyfileobj(source, target)
        print('Verified benchmark data:', output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).resolve().parents[1] /
                        'UKFTractography/Data/Input/dwi.raw.gz')
    download(parser.parse_args().output)
