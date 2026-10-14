#!/usr/bin/env python3
"""Package a Linux or Windows x64 Slicer extension with its native CUDA backend."""
import argparse
from datetime import date
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import tarfile
import tempfile


def check_windows_x64(path):
    """Reject Linux, 32-bit, ARM and malformed files in a Windows package."""
    with path.open('rb') as stream:
        header = stream.read(64)
        if len(header) != 64 or header[:2] != b'MZ':
            raise ValueError('Not a Windows PE binary: ' + str(path))
        stream.seek(struct.unpack_from('<I', header, 60)[0])
        signature = stream.read(6)
        if signature != b'PE\0\0\x64\x86':
            raise ValueError('Not a Windows x64 PE binary: ' + str(path))


def copy_runtime(args, runtime):
    binary_name = 'UKFTractography_gpu.exe' if args.platform == 'windows' else 'UKFTractography_gpu'
    gpu_name = 'ukf_gpu.dll' if args.platform == 'windows' else 'libukf_gpu.so'
    cuda_name = args.cuda_runtime.name
    if args.platform == 'linux':
        dynamic = subprocess.check_output(['readelf', '-d', str(args.cuda_runtime.resolve())], text=True)
        soname = re.search(r'\(SONAME\).*?\[(libcudart\.so\.[\d.]+)\]', dynamic)
        if not soname:
            raise ValueError('Expected a shared Linux CUDA runtime with a libcudart.so.<major> SONAME.')
        cuda_name = soname.group(1)
    binary = args.binary or args.repository.parent/'bin'/binary_name
    gpu_library = args.gpu_library or args.repository.parent/'build-gpu/lib'/gpu_name
    library_dir = runtime/'bin' if args.platform == 'windows' else runtime/'lib'
    (runtime/'bin').mkdir(parents=True)
    library_dir.mkdir(exist_ok=True)
    sources = [(binary, runtime/'bin'/binary_name),
               (gpu_library, library_dir/gpu_name),
               (args.cuda_runtime, library_dir/cuda_name)]
    for directory in args.runtime_dir:
        sources += [(file, library_dir/file.name) for file in sorted(directory.glob('*.dll'))]
    for source, destination in sources:
        source = source.resolve(strict=True)
        if args.platform == 'windows':
            check_windows_x64(source)
        if destination.exists():
            if source.read_bytes() != destination.read_bytes():
                raise ValueError('Conflicting runtime DLL: ' + destination.name)
            continue
        shutil.copy2(source, destination)
        if args.platform == 'linux':
            subprocess.run(['strip', '--strip-unneeded', str(destination)], check=True)
            with tempfile.TemporaryDirectory() as tmp:
                script = Path(tmp)/'remove-rpath.cmake'
                script.write_text('file(RPATH_REMOVE FILE "${UKF_RUNTIME_FILE}")\n')
                subprocess.run([args.cmake, '-DUKF_RUNTIME_FILE=' + str(destination.resolve()),
                                '-P', str(script)], check=True)
            dynamic = subprocess.check_output(['readelf', '-d', str(destination)], text=True)
            if re.search(r'\((?:RPATH|RUNPATH)\)', dynamic):
                raise RuntimeError('Unexpected runtime search path: ' + str(destination))
    if args.platform == 'windows':
        for name in ('vcruntime140.dll', 'vcruntime140_1.dll', 'msvcp140.dll'):
            if not (library_dir/name).is_file():
                raise ValueError('Missing MSVC runtime ' + name + '; provide --runtime-dir pointing to the x64 CRT directory')
    else:
        launcher = runtime/'bin/ukf-gpu'
        launcher.write_text('''#!/bin/sh
set -eu
runtime_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
unset LD_PRELOAD PYTHONHOME PYTHONPATH QT_PLUGIN_PATH QT_QPA_PLATFORM_PLUGIN_PATH
export LD_LIBRARY_PATH="$runtime_dir/lib"
exec "$runtime_dir/bin/UKFTractography_gpu" "$@"
''')
        launcher.chmod(0o755)
    for candidate in [args.cuda_runtime.resolve().parent/'LICENSE',
                      args.cuda_runtime.parent.parent/'LICENSE',
                      args.cuda_runtime.parent.parent/'EULA.txt']:
        if candidate.is_file():
            shutil.copy2(candidate, runtime/'CUDA-LICENSE.txt')
            break


def package(args):
    args.output.mkdir(parents=True, exist_ok=False)
    stage = args.output/'UKFGPU'
    shutil.copytree(args.repository/'slicer-extension', stage,
                    ignore=shutil.ignore_patterns('__pycache__', 'Runtime'))
    module = stage/'UKFGPU'
    copy_runtime(args, module/'Runtime')
    shutil.copy2(args.repository/'LICENSE.txt', stage/'LICENSE.txt')
    suffix = args.platform + '-x86_64-' + date.today().isoformat() + '.tar.gz'
    archive = args.output/('UKFGPU-' + suffix)
    with tarfile.open(archive, 'w:gz') as tar:
        tar.add(stage, arcname=stage.name)
    installed = args.output/'install-tree/UKFGPU'
    scripts = installed/('lib/Slicer-' + args.slicer_version)/'qt-scripted-modules'
    scripts.mkdir(parents=True)
    shutil.copy2(module/'UKFGPU.py', scripts/'UKFGPU.py')
    for name in ('UKFGPULib', 'Runtime'):
        shutil.copytree(module/name, scripts/name, ignore=shutil.ignore_patterns('__pycache__'))
    share = installed/('share/Slicer-' + args.slicer_version)
    share.mkdir(parents=True)
    (share/'UKFGPU.s4ext').write_text('''scm local
scmurl https://github.com/pnlbwh/ukftractography
scmrevision NA
depends NA
homepage https://github.com/pnlbwh/ukftractography
category Diffusion.Tractography
contributors UKF GPU contributors; original UKFTractography authors
description Experimental GPU UKF tractography: Simple, Full Tensor, FreeWater and NODDI.
status Experimental
enabled 1
''')
    for name in ('README.md', 'WINDOWS-2026-10-11.md', 'LICENSE.txt'):
        if (stage/name).is_file():
            shutil.copy2(stage/name, share/name)
    install_archive = args.output/('UKFGPU-Slicer' + args.slicer_version + '-' + suffix)
    with tarfile.open(install_archive, 'w:gz') as tar:
        tar.add(installed, arcname='UKFGPU')
    print('Module directory:', module)
    print('Source/runtime archive:', archive)
    print('Extensions Manager archive:', install_archive)
    return install_archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--platform', choices=['linux', 'windows'], default='windows' if os.name == 'nt' else 'linux')
    parser.add_argument('--binary', type=Path)
    parser.add_argument('--gpu-library', type=Path)
    parser.add_argument('--cuda-runtime', type=Path, required=True, help='Matching shared CUDA runtime (libcudart.so.* or cudart64_*.dll)')
    parser.add_argument('--runtime-dir', type=Path, action='append', default=[], help='Windows x64 DLL directory; repeat for additional dependencies')
    parser.add_argument('--output', type=Path, required=True, help='New output directory (must not exist)')
    parser.add_argument('--cmake', default='cmake')
    parser.add_argument('--slicer-version', default='5.12', help='Slicer major.minor, e.g. 5.6 for Slicer 5.6.2')
    args = parser.parse_args()
    if not re.fullmatch(r'\d+\.\d+', args.slicer_version):
        parser.error('--slicer-version must be major.minor')
    if args.platform == 'windows' and not re.fullmatch(r'cudart64_\d+\.dll', args.cuda_runtime.name, re.I):
        parser.error('Windows requires a native cudart64_*.dll runtime')
    args.repository = Path(__file__).resolve().parents[1]
    package(args)


if __name__ == '__main__':
    main()
