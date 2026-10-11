#!/usr/bin/env python3
"""Require a nonempty GPU branch pass, finite recordings, and CPU agreement."""
import argparse
from pathlib import Path
import re
import subprocess

import numpy as np
from assert_fiber_data import read
from strict_geometry import compare

parser = argparse.ArgumentParser()
parser.add_argument('--binary', default='../build-cpu/UKFTractography')
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
data = Path('UKFTractography/Data/Input')
common = ['--dwiFile', str(data/'two_tensor.nhdr'), '--maskFile', str(data/'mask.nhdr'),
          '--seedsFile', str(data/'seed.nhdr'), '--numTensor', '2', '--seedsPerVoxel', '1',
          '--numThreads', '4', '--minBranchingAngle', '1', '--maxBranchingAngle', '80',
          '--recordFA', '--recordTrace', '--recordState', '--recordCovariance']
results = {}
for mode in ('cpu', 'gpu'):
    extra = [] if mode == 'cpu' else ['--useGPU', '--gpuDevice', '0', '--gpuPrecision', '0']
    target = args.output/f'branches-{mode}.vtk'
    run = subprocess.run([args.binary]+common+extra+['--tracts', str(target)],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120)
    (args.output/f'branches-{mode}.log').write_text(run.stdout)
    assert run.returncode == 0, f'{mode}: CLI failed; see log'
    if mode == 'gpu':
        generated = re.search(r'\[GPU branches\] generated=(\d+)', run.stdout)
        assert generated and int(generated[1]) > 0, 'no GPU branches exercised'
        assert run.stdout.count('[GPU worker]') == 2 and 'falling back' not in run.stdout
    results[mode] = read(str(target))
geometry = compare(results['cpu'], results['gpu'])
assert geometry['passed'], f'strict geometry failed: {geometry}'
assert results['cpu'].keys() == results['gpu'].keys()
for key, expected in results['cpu'].items():
    actual = results['gpu'][key]
    assert actual.shape == expected.shape, f'{key}: shape differs'
    assert np.isfinite(actual).all(), f'{key}: nonfinite output'
    if key == 'lines':
        assert np.array_equal(actual, expected), 'connectivity differs'
    else:
        assert np.allclose(actual, expected, atol=0.001, rtol=0.0002), f'{key}: CPU/GPU mismatch'
print(f'PASS branches: GPU generated={generated[1]}, all geometry and recordings verified')
