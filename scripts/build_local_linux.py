#!/usr/bin/env python3
"""Build UKFGPU locally on Linux and install it into an existing Slicer."""
import argparse
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile


REPOSITORY = Path(__file__).resolve().parents[1]


def executable(path):
    path = Path(path).expanduser()
    if path.is_dir():
        path = path/'Slicer'
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError('Slicer must be an executable file or its installation directory: ' + str(path))
    return path.resolve()


def cuda_compiler(argument):
    candidates = []
    if argument:
        path = Path(argument).expanduser()
        compiler = path/'bin/nvcc' if path.is_dir() else path
        if not compiler.is_file() or not os.access(compiler, os.X_OK):
            raise ValueError('--cuda-path does not identify an executable nvcc or Toolkit root: ' + str(path))
        return compiler.resolve()
    for name in ('CUDA_PATH', 'CUDACXX'):
        value = os.environ.get(name)
        if value:
            path = Path(value.strip('"')).expanduser()
            candidates.append(path/'bin/nvcc' if name == 'CUDA_PATH' and path.is_dir()
                              else path/'nvcc' if name == 'CUDACXX' and path.is_dir()
                              else path)
    found = shutil.which('nvcc')
    if found:
        candidates.append(Path(found))
    candidates.extend(sorted(Path('/usr/local').glob('cuda*/bin/nvcc')))
    for path in candidates:
        if path.is_file() and os.access(path, os.X_OK):
            return path.resolve()
    raise ValueError('Could not find nvcc. Use --cuda-path with the Toolkit root or nvcc executable.')


def cuda_architectures(nvcc):
    try:
        codes = subprocess.check_output([str(nvcc), '--list-gpu-code'], text=True, stderr=subprocess.DEVNULL)
        supported = sorted({int(value) for value in re.findall(r'\bsm_(\d+)\b', codes) if int(value) >= 60})
        devices = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'],
            text=True, stderr=subprocess.DEVNULL, timeout=10)
        local = {int(major)*10+int(minor) for major, minor in
                 re.findall(r'^\s*(\d+)\.(\d+)\s*$', devices, re.M)}
        real = sorted(local & set(supported))
        if not real or not supported:
            return None
        return ';'.join([*(str(value)+'-real' for value in real), str(supported[0])+'-virtual'])
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def runtime_path(cuda_root, nvcc):
    candidates = [cuda_root/'lib64', cuda_root/'targets/x86_64-linux/lib',
                  cuda_root/'lib', nvcc.parent.parent/'lib64', nvcc.parent.parent/'lib',
                  nvcc.parent.parent/'targets/x86_64-linux/lib']
    for directory in candidates:
        runtimes = sorted(directory.glob('libcudart.so.*')) if directory.is_dir() else []
        if runtimes:
            return runtimes[-1].resolve()
    raise ValueError('Could not find versioned libcudart.so under the CUDA Toolkit. Checked: '
                     + ', '.join(str(path) for path in candidates))


def preflight(args):
    if platform.system() != 'Linux' or platform.machine() not in ('x86_64', 'amd64'):
        raise ValueError('Local Linux builds require Linux x86_64.')
    slicer = executable(args.slicer)
    lib_dirs = (slicer.parent/'lib', slicer.parent.parent/'lib')
    slicer_versions = [match for directory in lib_dirs if directory.is_dir()
                       for path in directory.glob('Slicer-*')
                       if path.is_dir() and (match := re.fullmatch(r'Slicer-(\d+)\.(\d+)', path.name))]
    if not slicer_versions or max(tuple(map(int, match.groups())) for match in slicer_versions) < (5, 2):
        raise ValueError('Slicer 5.2 or newer is required (version not found under lib/Slicer-X.Y).')
    tools = {}
    for name in ('cmake', 'git', 'gcc', 'g++', 'make'):
        found = shutil.which(name)
        if not found:
            raise ValueError('Required build tool is missing from PATH: ' + name)
        tools[name] = Path(found).resolve()
    version = subprocess.check_output([str(tools['cmake']), '--version'], text=True)
    match = re.search(r'cmake version (\d+)\.(\d+)', version)
    if not match or tuple(map(int, match.groups())) < (3, 22):
        raise ValueError('CMake 3.22 or newer is required.')
    nvcc = cuda_compiler(args.cuda_path)
    root = nvcc.parent.parent
    runtime = runtime_path(root, nvcc)
    return slicer, tools, nvcc, root, runtime


def run(command, **kwargs):
    print('+', ' '.join(map(str, command)), flush=True)
    subprocess.run(command, check=True, **kwargs)


def install_bundle(slicer, archive, build_dir):
    # Keep upgrade archives available after the installer closes.
    destination = build_dir/'UKFGPU-Installer'
    if destination.exists():
        if not all((destination/name).is_file() for name in ('manifest.json', 'install.py')):
            raise ValueError('Existing directory is not a recognized UKFGPU installer: ' + str(destination))
    with tempfile.TemporaryDirectory(prefix='UKFGPU-install-', dir=build_dir) as temp:
        root = Path(temp)
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(root)
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(root/'UKFGPU-Installer', destination)
    env = os.environ.copy()
    for name in ('PYTHONHOME', 'PYTHONPATH', 'QT_PLUGIN_PATH', 'QT_QPA_PLATFORM_PLUGIN_PATH',
                 'UKF_GPU_EXECUTABLE'):
        env.pop(name, None)
    subprocess.Popen([str(slicer), '--no-splash', '--python-script', str(destination/'install.py')], env=env)
    print('Slicer installer launched. Follow its installation/upgrade instructions and restart Slicer.')
    print('Installer files and upgrade archives remain in:', destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--slicer', required=True, help='Slicer executable or installation directory')
    parser.add_argument('--build-dir', type=Path,
                        default=REPOSITORY.parent/'build-local-linux', help='Build/cache directory (preserved between runs)')
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--cuda-path', help='CUDA Toolkit root or nvcc executable; otherwise autodetected')
    parser.add_argument('--check-only', action='store_true', help='Run preflight without configuring or building')
    parser.add_argument('--no-install', action='store_true', help='Build and package, but do not launch Slicer')
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error('--jobs must be at least 1')
    try:
        slicer, tools, nvcc, cuda_root, cudart = preflight(args)
        architectures = cuda_architectures(nvcc)
        print('Slicer:', slicer)
        print('Build tools:', ', '.join(name+'='+str(path) for name, path in tools.items()))
        print('CUDA Toolkit:', cuda_root, '(nvcc:', nvcc, ')')
        print('CUDA runtime:', cudart)
        if architectures:
            print('CUDA targets:', architectures)
        else:
            print('CUDA device/target probing unavailable; CMake will use its all-supported-architecture default.')
        if args.check_only:
            return 0
        if not args.build_dir.is_absolute():
            args.build_dir = (Path.cwd()/args.build_dir).resolve()
        args.build_dir.mkdir(parents=True, exist_ok=True)
        configure = [str(tools['cmake']), '-S', str(REPOSITORY/'platforms/superbuild'),
                     '-B', str(args.build_dir), '-DCMAKE_BUILD_TYPE=Release',
                     '-DCMAKE_CUDA_COMPILER='+str(nvcc),
                     '-DUKF_CUDA_ARCHITECTURES='+(architectures or '')]
        run(configure)
        build_env = os.environ.copy()
        build_env['CMAKE_BUILD_PARALLEL_LEVEL'] = str(args.jobs)
        run([str(tools['cmake']), '--build', str(args.build_dir), '--parallel', str(args.jobs)], env=build_env)
        # ExternalProject stamps do not track edits in our local backend sources.
        run([str(tools['cmake']), '-S', str(REPOSITORY/'platforms/standalone'),
             '-B', str(args.build_dir/'backend')])
        run([str(tools['cmake']), '--build', str(args.build_dir/'backend'),
             '--parallel', str(args.jobs)], env=build_env)
        backend = args.build_dir/'backend/bin'
        binary = backend/'UKFTractography_gpu'
        gpu_library = backend/'libukf_gpu.so'
        if not binary.is_file() or not gpu_library.is_file():
            raise ValueError('Build finished without the expected backend files in ' + str(backend))
        with tempfile.TemporaryDirectory(prefix='UKFGPU-package-', dir=args.build_dir) as temp:
            staging = Path(temp)
            package_dir = staging/'package'
            run([sys.executable, str(REPOSITORY/'scripts/package_slicer_extension.py'),
                 '--platform', 'linux', '--binary', str(binary), '--gpu-library', str(gpu_library),
                 '--cuda-runtime', str(cudart), '--output', str(package_dir)])
            candidate = staging/'UKFGPU-Installer-linux-x86_64.zip'
            run([sys.executable, str(REPOSITORY/'scripts/package_portable_installer.py'),
                 '--runtime', 'linux-x86_64='+str(package_dir/'UKFGPU/UKFGPU/Runtime'),
                 '--output', str(candidate)])
            final = args.build_dir/'UKFGPU-Installer-linux-x86_64.zip'
            os.replace(candidate, final)
        print('Installer bundle:', final)
        if not args.no_install:
            install_bundle(slicer, final, args.build_dir)
        return 0
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print('Build/install failed:', exc, file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
