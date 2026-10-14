#!/usr/bin/env python3
"""Bundle native runtimes into an offline installer adapting to the running Slicer version."""
import argparse
from datetime import date
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tempfile
import zipfile

from package_slicer_extension import check_windows_x64


def complete_linux_runtime(runtime):
    """Bundle the dependency closure except the system glibc and GPU driver."""
    library = runtime/'lib'
    env = dict(os.environ, LD_LIBRARY_PATH=str(library.resolve()))
    env.pop('LD_PRELOAD', None)
    baseline = re.compile(r'(?:lib(?:c|m|pthread|dl|rt|resolv|util)\.so\.|ld-linux|libcuda\.|libnvidia-)')
    pending = [runtime/'bin/UKFTractography_gpu', *library.glob('*.so*')]
    checked = set()
    while pending:
        path = pending.pop()
        if path.name in checked:
            continue
        checked.add(path.name)
        output = subprocess.check_output(['ldd', str(path)], env=env, text=True)
        if 'not found' in output:
            raise RuntimeError('Unresolved native dependency:\n' + output)
        for name, resolved in re.findall(r'^\s*(\S+) => (/.+?) \(0x', output, re.M):
            if baseline.match(name):
                continue
            destination = library/name
            if not destination.exists():
                shutil.copy2(Path(resolved).resolve(), destination)
                pending.append(destination)
    requirements = []
    for path in [runtime/'bin/UKFTractography_gpu', *library.glob('*.so*')]:
        header = subprocess.check_output(['readelf', '-h', str(path)], text=True)
        if 'Advanced Micro Devices X86-64' not in header:
            raise ValueError('Expected Linux x86_64 ELF: ' + str(path))
        versions = subprocess.check_output(['readelf', '--version-info', str(path)], text=True)
        requirements.extend(tuple(map(int, value.split('.'))) for value in re.findall(r'Name: GLIBC_([\d.]+)', versions))
    for package in ('libstdc++6', 'libgcc-s1'):
        license_file = Path('/usr/share/doc')/package/'copyright'
        if license_file.is_file():
            shutil.copy2(license_file, runtime/(package+'-copyright.txt'))
    for name in ('GPL-3', 'LGPL-3'):
        source = Path('/usr/share/common-licenses')/name
        if source.is_file():
            shutil.copy2(source, runtime/(name+'.txt'))
    return '.'.join(map(str, max(requirements)))


def create_bundle(repository, runtimes, output):
    with tempfile.TemporaryDirectory(prefix='UKFGPU-bundle-') as tmp:
        bundle = Path(tmp)/'UKFGPU-Installer'
        module = bundle/'module'
        module.mkdir(parents=True)
        shutil.copy2(repository/'slicer-extension/UKFGPU/UKFGPU.py', module/'UKFGPU.py')
        shutil.copytree(repository/'slicer-extension/UKFGPU/UKFGPULib', module/'UKFGPULib',
                        ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copy2(repository/'slicer-extension/install.py', bundle/'install.py')
        docs = bundle/'docs'
        docs.mkdir()
        for source in [repository/'LICENSE.txt', *[repository/'slicer-extension'/name for name in
                       ('README.md', 'INSTALLER.md', 'WINDOWS.md')]]:
            shutil.copy2(source, docs/source.name)
        manifest = dict(format=1, source_commit=subprocess.check_output(
            ['git', '-C', str(repository), 'rev-parse', 'HEAD'], text=True).strip(), runtimes={})
        manifest['source_dirty'] = bool(subprocess.check_output(
            ['git', '-C', str(repository), 'status', '--porcelain'], text=True).strip())
        for key, source in runtimes.items():
            if key not in ('linux-x86_64', 'windows-x86_64'):
                raise ValueError('Unsupported native runtime: ' + key)
            runtime = bundle/'runtimes'/key
            shutil.copytree(source, runtime, ignore=shutil.ignore_patterns('__pycache__'))
            if key == 'linux-x86_64':
                if not (runtime/'bin/ukf-gpu').is_file():
                    raise ValueError('Linux runtime requires the isolated ukf-gpu launcher.')
                manifest['runtimes'][key] = dict(glibc_min=complete_linux_runtime(runtime))
            else:
                for name in ('UKFTractography_gpu.exe', 'ukf_gpu.dll', 'vcruntime140.dll', 'vcruntime140_1.dll', 'msvcp140.dll'):
                    check_windows_x64(runtime/'bin'/name)
                if not list((runtime/'bin').glob('cudart64_*.dll')):
                    raise ValueError('Missing native Windows CUDA runtime.')
                for path in (runtime/'bin').glob('*.dll'):
                    check_windows_x64(path)
                manifest['runtimes'][key] = {}
            # Native loader/ABI failures must be caught before publishing a bundle.
            host = {'Linux': 'linux-x86_64', 'Windows': 'windows-x86_64'}.get(platform.system())
            if key == host:
                executable = runtime/'bin'/('ukf-gpu' if key.startswith('linux') else 'UKFTractography_gpu.exe')
                env = dict(os.environ)
                for name in ('LD_PRELOAD', 'PYTHONHOME', 'PYTHONPATH', 'QT_PLUGIN_PATH', 'QT_QPA_PLATFORM_PLUGIN_PATH'):
                    env.pop(name, None)
                if key.startswith('windows'):
                    windows = next(value for name, value in env.items() if name.upper() == 'SYSTEMROOT')
                    env['PATH'] = ';'.join((str(executable.parent), windows+'\\System32', windows))
                result = subprocess.run([str(executable), '--help'], env=env, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, timeout=30)
                if result.returncode:
                    raise RuntimeError('Packaged native backend failed to load:\n' + result.stdout.decode(errors='replace'))
        manifest['sha256'] = {path.relative_to(bundle).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in sorted(bundle.rglob('*')) if path.is_file()}
        (bundle/('manifest-' + date.today().isoformat()+'.json')).write_text(
            json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
        with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(bundle.rglob('*')):
                if path.is_file():
                    archive.write(path, 'UKFGPU-Installer/'+path.relative_to(bundle).as_posix())
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', action='append', required=True, metavar='PLATFORM=PATH',
                        help='linux-x86_64 or windows-x86_64 with a packaged Runtime directory; repeat to combine')
    parser.add_argument('--output', type=Path, required=True, help='New installer ZIP')
    args = parser.parse_args()
    runtimes = {}
    for entry in args.runtime:
        key, path = entry.split('=', 1)
        if key in runtimes:
            parser.error('Duplicate platform: ' + key)
        runtimes[key] = Path(path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print('Offline, version-adaptive installer:', create_bundle(Path(__file__).resolve().parents[1], runtimes, args.output))


if __name__ == '__main__':
    main()
