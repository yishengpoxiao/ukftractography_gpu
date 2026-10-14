"""Native Windows/Linux CPU-vs-GPU checks; collect failures without weakening tolerance."""
import json
import gzip
import hashlib
import os
from pathlib import Path
import subprocess
import re
import sys
import time
import traceback

import numpy as np
import slicer

try:
    repository = Path(os.environ['UKF_TEST_REPOSITORY'])
    sys.path.insert(0, str(repository/'tests'))
    sys.path.insert(0, str(repository/'slicer-extension/UKFGPU'))
    from assert_fiber_data import read
    from strict_geometry import compare
    from UKFGPULib import MODELS, command, process_environment
    binary = os.environ['UKF_NATIVE_BINARY']
    output = Path(os.environ['UKF_NATIVE_OUTPUT'])
    data = repository/'UKFTractography/Data/Input'
    environment = process_environment(slicer.util.startupEnvironment(), binary)
    real_seeds = int(os.environ.get('UKF_NATIVE_REAL_SEEDS', '0'))
    seed_file = data/'seed.nhdr'
    mask_file = data/'mask.nhdr'
    if real_seeds:
        if not (data/'dwi.raw.gz').is_file():
            raise FileNotFoundError('Run scripts/fetch_benchmark_data.py before whole-brain acceptance.')
        text = (data/'dwi-mask.nhdr').read_text()
        shape = tuple(map(int, re.search(r'^sizes: (.+)$', text, re.M)[1].split()))
        mask = np.frombuffer(gzip.decompress((data/'dwi-mask.raw.gz').read_bytes()), dtype='<i2').reshape(shape[::-1]) > 0
        interior = mask.copy()
        interior[[0,-1],:,:] = False
        interior[:,[0,-1],:] = False
        interior[:,:,[0,-1]] = False
        for axis in range(3):
            interior &= np.roll(mask, 1, axis) & np.roll(mask, -1, axis)
        seed = np.zeros(mask.size, dtype='<i2')
        seed[np.random.default_rng(20261014).choice(np.flatnonzero(interior), real_seeds, replace=False)] = 1
        (output/'seeds.raw.gz').write_bytes(gzip.compress(seed.tobytes()))
        seed_file = output/'seeds.nhdr'
        seed_file.write_text(re.sub(r'^data file: .+$', 'data file: seeds.raw.gz', text, flags=re.M))
        mask_file = data/'dwi-mask.nhdr'
    selected = set(filter(None, os.environ.get('UKF_NATIVE_MODELS', '').split(',')))
    cases = {}
    for model, (n, flags) in MODELS.items():
        name = model.lower().replace('t', '').replace('f', '') if model.startswith('NODDI') else model.lower().replace('t', '')
        if selected and name not in selected:
            continue
        fw = '--freeWater' in flags
        source = ('single_tensor' if n == 1 else 'two_tensor') + ('_fw' if fw else '') + '.nhdr'
        if real_seeds:
            source = 'dwi.nhdr'
        options = dict(numThreads=16 if real_seeds else 4, seedsPerVoxel=1, minBranchingAngle=0, maxBranchingAngle=0,
                       recordNMSE=True, recordState=True, recordCovariance=True)
        if real_seeds:
            options['maxHalfFiberLength'] = 80
        options.update(dict(recordVic=True, recordKappa=True, recordViso=True) if '--noddi' in flags else
                       dict(recordFA=True, recordTrace=True, recordTensors=True))
        if fw:
            options['recordFreeWater'] = True
        arrays = {}
        timings = {}
        execution_errors = {}
        commands = {}
        for mode, precision in [('cpu', 0), ('fp64', 0), ('mixed', 1)]:
            target = output/(name+'-'+mode+'.vtk')
            argv = command(binary, data/source, mask_file, target, model, precision,
                           seeds=seed_file, options=options)
            if mode == 'cpu':
                argv.remove('--useGPU')
                for flag in ('--gpuDevice', '--gpuPrecision'):
                    index = argv.index(flag)
                    del argv[index:index+2]
            start = time.monotonic()
            commands[mode] = argv
            try:
                run = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, encoding='utf-8', errors='replace', env=environment, timeout=180)
            except subprocess.TimeoutExpired as exc:
                timings[mode] = time.monotonic()-start
                (output/(name+'-'+mode+'.log')).write_bytes(exc.stdout or b'')
                execution_errors[mode] = 'Backend timed out after 180 seconds.'
                continue
            timings[mode] = time.monotonic()-start
            (output/(name+'-'+mode+'.log')).write_text(run.stdout, encoding='utf-8')
            if run.returncode or (mode != 'cpu' and ('falling back' in run.stdout or 'tractography done on device' not in run.stdout)):
                execution_errors[mode] = next((line for line in run.stdout.splitlines() if 'falling back' in line),
                                               'Backend failed: exit code '+str(run.returncode))
                continue
            try:
                arrays[mode] = read(str(target))
            except Exception as exc:
                execution_errors[mode] = 'Could not read fiber output: '+str(exc)
        cases[name] = dict(seconds=timings, commands=commands, execution_errors=execution_errors)
        for mode in ('fp64', 'mixed'):
            if mode not in arrays or 'cpu' not in arrays:
                cases[name][mode] = dict(passed=False, execution_error=execution_errors.get(mode, execution_errors.get('cpu')))
                print('FAIL '+name+'/'+mode+': '+cases[name][mode]['execution_error'], flush=True)
                continue
            actual, expected = arrays[mode], arrays['cpu']
            geometry = compare(expected, actual)
            same_arrays = actual.keys() == expected.keys()
            effective_mixed = mode == 'mixed' and model.startswith('Simple') and not fw
            atol, rtol = (0.002, 0.0002) if effective_mixed else (0.00002, 0.000002)
            record_errors = {}
            for key in expected.keys() & actual.keys():
                if key in ('points', 'lines'):
                    continue
                a, b = expected[key], actual[key]
                finite = np.isfinite(a).all() and np.isfinite(b).all()
                passed = a.shape == b.shape and finite and np.allclose(a, b, atol=atol, rtol=rtol)
                record_errors[key] = dict(passed=bool(passed), cpu_shape=list(a.shape), gpu_shape=list(b.shape),
                                         max_absolute_error=float(np.max(np.abs(a-b))) if a.shape == b.shape and finite and a.size else None)
                same_arrays = same_arrays and passed
            cases[name][mode] = {'passed': bool(geometry['passed'] and same_arrays),
                                 'geometry': geometry, 'record_arrays': record_errors}
            print(('PASS ' if cases[name][mode]['passed'] else 'FAIL ') + name+'/'+mode, flush=True)
    if not cases or (selected and selected != cases.keys()):
        raise ValueError('No cases matched, or an unknown model name was selected')
    report = {'platform': sys.platform, 'slicer': slicer.app.applicationVersion,
              'input': 'whole-brain' if real_seeds else 'synthetic', 'real_seed_voxels': real_seeds,
              'position_tolerance_mm': 0.001, 'cases': cases,
              'passed': all(case[mode]['passed'] for case in cases.values() for mode in ('fp64', 'mixed'))}
    native = Path(binary)
    artifacts = {native, *native.parent.glob('UKFTractography_gpu*'), *native.parent.glob('ukf_gpu.dll'),
                 *(native.parent.parent/'lib').glob('libukf_gpu.so*')}
    report['native_sha256'] = {file.name: hashlib.sha256(file.read_bytes()).hexdigest() for file in sorted(artifacts) if file.is_file()}
    (output/'acceptance.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    slicer.util.exit(0 if report['passed'] else 1)
except Exception:
    traceback.print_exc()
    slicer.util.exit(1)
