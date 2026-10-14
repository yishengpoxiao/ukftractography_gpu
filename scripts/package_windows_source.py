#!/usr/bin/env python3
"""Create a Windows source/build kit; never label source as a native runtime."""
import argparse
from datetime import date
from pathlib import Path
import zipfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True, help='New .zip file')
args = parser.parse_args()
repository = Path(__file__).resolve().parents[1]
args.output.parent.mkdir(parents=True, exist_ok=True)
source_root = 'UKFGPU-Windows-source-' + date.today().isoformat()
files = set()
for name in ('common', 'ukf', 'platforms', 'slicer-extension', 'third-party'):
    for path in (repository/name).rglob('*'):
        if (path.is_file() and not {'__pycache__', 'Runtime', 'Testing'} & set(path.parts)
                and not any(word in path.name.upper() for word in ('VALIDATION', 'ACCURACY'))):
            files.add(path)
for name in ('package_windows_source.py', 'package_slicer_extension.py', 'package_portable_installer.py', 'build_windows_gpu.ps1', 'install_cuda_toolkit.ps1'):
    files.add(repository/'scripts'/name)
for name in ('single_tensor', 'single_tensor_fw', 'two_tensor', 'two_tensor_fw', 'mask', 'seed'):
    for extension in ('.nhdr', '.raw.gz'):
        files.add(repository/'UKFTractography/Data/Input'/(name+extension))
files.update([repository/'LICENSE.txt', repository/'UKFTractography/UKFTractography.cxx',
              repository/'UKFTractography/UKFTractography-2026-10-10.xml'])
with zipfile.ZipFile(args.output, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(files):
        archive.write(path, source_root+'/'+path.relative_to(repository).as_posix())
    archive.writestr(source_root+'/README-WINDOWS.md',
                     (repository/'slicer-extension/WINDOWS.md').read_text(encoding='utf-8'))
print('Windows source/build kit (no native executable):', args.output)
