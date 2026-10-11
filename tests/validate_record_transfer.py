#!/usr/bin/env python3
"""Require unchanged GPU output after record-download optimizations."""
import argparse
from pathlib import Path
import re
import subprocess

import numpy as np
from assert_fiber_data import read

parser = argparse.ArgumentParser()
parser.add_argument('--reference', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--binary', default='../build-cpu/UKFTractography')
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
data = Path('UKFTractography/Data/Input')
references = sorted(args.reference.glob('*-fp64.vtk'))
assert len(references) == 12, 'reference must contain all 12 GPU model fixtures'
for path in references:
    name = path.name.removesuffix('-fp64.vtk')
    kind, number, fw = re.fullmatch(r'(simple|full|noddi)([123])(-fw)?', name).groups()
    n = int(number)
    source = ('single_tensor' if n == 1 else 'two_tensor')+('_fw' if fw else '')+'.nhdr'
    flags = ['--recordVic', '--recordKappa', '--recordViso', '--noddi'] if kind == 'noddi' else ['--recordFA', '--recordTrace', '--recordTensors']
    if kind == 'full':
        flags += ['--fullTensorModel']
    if fw:
        flags += ['--freeWater', '--recordFreeWater']
    common = ['--dwiFile', str(data/source), '--maskFile', str(data/'mask.nhdr'),
              '--seedsFile', str(data/'seed.nhdr'), '--seedsPerVoxel', '1', '--numTensor', str(n),
              '--numThreads', '4', '--minBranchingAngle', '0', '--maxBranchingAngle', '0',
              '--recordNMSE', '--recordState', '--recordCovariance', '--useGPU', '--gpuDevice', '0']+flags
    for mode, precision in [('fp64', 0), ('mixed', 1)]:
        target = args.output/f'{name}-{mode}.vtk'
        run = subprocess.run([args.binary]+common+['--gpuPrecision', str(precision), '--tracts', str(target)],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=180)
        (args.output/f'{name}-{mode}.log').write_text(run.stdout)
        assert run.returncode == 0 and 'tractography done on device' in run.stdout and 'falling back' not in run.stdout, f'{name}/{mode}: GPU execution failed'
        expected = read(str(args.reference/target.name))
        actual = read(str(target))
        assert expected.keys() == actual.keys(), f'{name}/{mode}: arrays differ'
        for key, values in expected.items():
            assert np.isfinite(actual[key]).all(), f'{name}/{mode}/{key}: nonfinite'
            assert np.array_equal(values, actual[key]), f'{name}/{mode}/{key}: output changed'
        print(f'PASS {name}/{mode}: geometry, connectivity and all recorded arrays unchanged', flush=True)
