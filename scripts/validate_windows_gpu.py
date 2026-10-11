#!/usr/bin/env python3
"""Run native backend CPU/GPU precision checks using Slicer's bundled VTK/Numpy."""
import argparse
import os
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--slicer', type=Path, required=True)
parser.add_argument('--binary', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--models', nargs='+', help='e.g. simple1 simple2; default all 12')
args = parser.parse_args()
repository = Path(__file__).resolve().parents[1]
if args.output.exists():
    parser.error('--output must be a new directory')
args.output.mkdir(parents=True)
env = dict(os.environ, UKF_TEST_REPOSITORY=str(repository),
           UKF_NATIVE_BINARY=str(args.binary.resolve(strict=True)),
           UKF_NATIVE_OUTPUT=str(args.output.resolve()),
           UKF_NATIVE_MODELS=','.join(args.models or []))
with (args.output/'slicer-validation.log').open('w', encoding='utf-8') as log:
    run = subprocess.run([str(args.slicer.resolve(strict=True)), '--no-splash', '--no-main-window',
                          '--python-script', str(repository/'slicer-extension/UKFGPU/Testing/native_acceptance.py')],
                         stdout=log, stderr=subprocess.STDOUT, env=env, timeout=1200)
print('Results:', args.output/'acceptance.json')
raise SystemExit(run.returncode)
