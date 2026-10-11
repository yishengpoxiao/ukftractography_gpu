#!/usr/bin/env python3
"""Run every supported model against its CPU path; require actual GPU execution."""
import argparse
from pathlib import Path
import subprocess
import time

import numpy as np

from assert_fiber_data import read
from strict_geometry import compare


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', type=Path, default=Path('../build-cpu/UKFTractography'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--models', nargs='*')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = Path('UKFTractography/Data/Input')
    cases = []
    for n in (1, 2, 3):
        for full in (False, True):
            for fw in (False, True) if n < 3 else (False,):
                name = ('full' if full else 'simple') + str(n) + ('-fw' if fw else '')
                flags = (['--fullTensorModel'] if full else []) + (['--freeWater', '--recordFreeWater'] if fw else [])
                cases.append((name, n, fw, flags, ['--recordFA', '--recordTrace', '--recordTensors']))
    for n in (1, 2):
        cases.append(('noddi'+str(n), n, False, ['--noddi'], ['--recordVic', '--recordKappa', '--recordViso']))
    for name, n, fw, flags, records in cases:
        if args.models and name not in args.models:
            continue
        source = ('single_tensor' if n == 1 else 'two_tensor') + ('_fw' if fw else '') + '.nhdr'
        common = ['--dwiFile', str(data/source), '--maskFile', str(data/'mask.nhdr'),
                  '--seedsFile', str(data/'seed.nhdr'), '--seedsPerVoxel', '1', '--numTensor', str(n),
                  '--numThreads', '4', '--minBranchingAngle', '0', '--maxBranchingAngle', '0',
                  '--recordNMSE', '--recordState', '--recordCovariance'] + flags + records
        timings = {}
        arrays = {}
        for mode in ('cpu', 'fp64', 'mixed'):
            output = args.output/f'{name}-{mode}.vtk'
            extra = [] if mode == 'cpu' else ['--useGPU', '--gpuDevice', '0', '--gpuPrecision', str(int(mode == 'mixed'))]
            start = time.monotonic()
            run = subprocess.run([str(args.binary)]+common+extra+['--tracts', str(output)],
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=180)
            timings[mode] = time.monotonic()-start
            (args.output/f'{name}-{mode}.log').write_text(run.stdout)
            if run.returncode != 0:
                raise AssertionError(f'{name}/{mode}: CLI failed ({run.returncode}); see log')
            if mode != 'cpu' and ('tractography done on device' not in run.stdout or 'falling back' in run.stdout):
                raise AssertionError(f'{name}/{mode}: did not run completely on GPU')
            arrays[mode] = read(str(output))
        reference = arrays['cpu']
        if len(reference['points']) == 0:
            raise AssertionError(f'{name}: empty reference is insufficient coverage')
        errors = {}
        for mode in ('fp64', 'mixed'):
            actual = arrays[mode]
            geometry = compare(reference, actual)
            assert geometry['passed'], f'{name}/{mode}: strict geometry failed: {geometry}'
            assert actual.keys() == reference.keys(), f'{name}/{mode}: recorded arrays differ'
            for key, expected in reference.items():
                got = actual[key]
                assert got.shape == expected.shape, f'{name}/{mode}/{key}: shape {got.shape} != {expected.shape}'
                assert np.isfinite(got).all(), f'{name}/{mode}/{key}: nonfinite'
                if key == 'lines':
                    assert np.array_equal(got, expected), f'{name}/{mode}: connectivity differs'
                elif key == 'points':
                    error = float(np.linalg.norm(got.astype(np.float64)-expected.astype(np.float64), axis=1).max())
                    errors[mode] = error
                    assert error <= 0.001, f'{name}/{mode}: position error {error} mm'
                else:
                    # State/covariance include packed eigenvalues. Scale per value,
                    # while still reporting all maxima for audit.
                    atol, rtol = (0.002, 0.0002) if mode == 'mixed' else (0.00002, 0.000002)
                    assert np.allclose(got, expected, atol=atol, rtol=rtol), f'{name}/{mode}/{key}: scalar error {np.max(np.abs(got-expected))}'
        print(f'PASS {name}: points={len(reference["points"])} position_mm={errors} seconds={timings}', flush=True)


if __name__ == '__main__':
    main()
