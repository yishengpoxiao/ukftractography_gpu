"""Native Windows/Linux CPU-vs-GPU checks; collect failures without weakening tolerance."""
import json
import os
from pathlib import Path
import subprocess
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
    selected = set(filter(None, os.environ.get('UKF_NATIVE_MODELS', '').split(',')))
    cases = {}
    for model, (n, flags) in MODELS.items():
        name = model.lower().replace('t', '').replace('f', '') if model.startswith('NODDI') else model.lower().replace('t', '')
        if selected and name not in selected:
            continue
        fw = '--freeWater' in flags
        source = ('single_tensor' if n == 1 else 'two_tensor') + ('_fw' if fw else '') + '.nhdr'
        options = dict(numThreads=4, seedsPerVoxel=1, minBranchingAngle=0, maxBranchingAngle=0,
                       recordNMSE=True, recordState=True, recordCovariance=True)
        options.update(dict(recordVic=True, recordKappa=True, recordViso=True) if '--noddi' in flags else
                       dict(recordFA=True, recordTrace=True, recordTensors=True))
        if fw:
            options['recordFreeWater'] = True
        arrays = {}
        timings = {}
        for mode, precision in [('cpu', 0), ('fp64', 0), ('mixed', 1)]:
            target = output/(name+'-'+mode+'.vtk')
            argv = command(binary, data/source, data/'mask.nhdr', target, model, precision,
                           seeds=data/'seed.nhdr', options=options)
            if mode == 'cpu':
                argv.remove('--useGPU')
                for flag in ('--gpuDevice', '--gpuPrecision'):
                    index = argv.index(flag)
                    del argv[index:index+2]
            start = time.monotonic()
            run = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, encoding='utf-8', errors='replace', env=environment, timeout=180)
            timings[mode] = time.monotonic()-start
            (output/(name+'-'+mode+'.log')).write_text(run.stdout, encoding='utf-8')
            if run.returncode or (mode != 'cpu' and ('falling back' in run.stdout or 'tractography done on device' not in run.stdout)):
                raise RuntimeError(name+'/'+mode+': backend execution failed; see log')
            arrays[mode] = read(str(target))
        cases[name] = {'seconds': timings}
        for mode in ('fp64', 'mixed'):
            actual, expected = arrays[mode], arrays['cpu']
            geometry = compare(expected, actual)
            same_arrays = actual.keys() == expected.keys()
            atol, rtol = (0.002, 0.0002) if mode == 'mixed' else (0.00002, 0.000002)
            record_errors = {}
            for key in expected.keys() & actual.keys():
                if key in ('points', 'lines'):
                    continue
                a, b = expected[key], actual[key]
                passed = a.shape == b.shape and np.isfinite(b).all() and np.allclose(a, b, atol=atol, rtol=rtol)
                record_errors[key] = {'passed': bool(passed)}
                same_arrays = same_arrays and passed
            cases[name][mode] = {'passed': bool(geometry['passed'] and same_arrays),
                                 'geometry': geometry, 'record_arrays': record_errors}
            print(('PASS ' if cases[name][mode]['passed'] else 'FAIL ') + name+'/'+mode, flush=True)
    if not cases or (selected and selected != cases.keys()):
        raise ValueError('No cases matched, or an unknown model name was selected')
    report = {'platform': sys.platform, 'slicer': slicer.app.applicationVersion,
              'position_tolerance_mm': 0.001, 'cases': cases,
              'passed': all(case[mode]['passed'] for case in cases.values() for mode in ('fp64', 'mixed'))}
    (output/'acceptance.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    slicer.util.exit(0 if report['passed'] else 1)
except Exception:
    traceback.print_exc()
    slicer.util.exit(1)
