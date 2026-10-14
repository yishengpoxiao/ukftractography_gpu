#!/usr/bin/env python3
"""Install and reload a real portable bundle in two isolated Slicer processes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import zipfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--slicer', type=Path, required=True)
parser.add_argument('--bundle', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
args.output = args.output.resolve()
args.output.mkdir(parents=True, exist_ok=False)
with zipfile.ZipFile(args.bundle) as archive:
    archive.extractall(args.output/'bundle')
tests = Path(__file__).resolve().parents[1]/'slicer-extension/UKFGPU/Testing'
env = dict(os.environ, UKF_INSTALL_BUNDLE=str(args.output/'bundle/UKFGPU-Installer'),
           UKF_INSTALL_TEST_OUTPUT=str(args.output))
for name in ('PYTHONPATH', 'PYTHONHOME', 'QT_PLUGIN_PATH', 'QT_QPA_PLATFORM_PLUGIN_PATH', 'UKF_GPU_EXECUTABLE'):
    env.pop(name, None)
base = [str(args.slicer.resolve(strict=True)), '--testing', '--no-splash', '--no-main-window']
for name in ('install_bundle', 'slicer_loader'):
    extra = []
    if name == 'slicer_loader':
        installed = json.loads((args.output/'installed.json').read_text(encoding='utf-8'))
        extra = ['--additional-module-paths', *installed['paths']]
    with (args.output/(name+'.log')).open('w', encoding='utf-8') as log:
        result = subprocess.run(base+extra+['--python-script', str(tests/(name+'.py'))],
                                env=env, stdout=log, stderr=subprocess.STDOUT, timeout=180)
    if result.returncode:
        raise RuntimeError(name+' failed; inspect '+str(args.output/(name+'.log')))
print('PASS real installation and registered-module loading:', args.output/'installed.json')
