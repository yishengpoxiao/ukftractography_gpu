"""Structural Windows packaging test; no native Windows execution."""
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

repository = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repository/'scripts'))
from package_portable_installer import create_bundle


class PortablePackagerTest(unittest.TestCase):
    def test_uppercase_windows_environment_and_bundled_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = root/'Runtime/bin'
            runtime.mkdir(parents=True)
            data = bytearray(128)
            data[:2] = b'MZ'
            struct.pack_into('<I', data, 60, 64)
            data[64:70] = b'PE\0\0\x64\x86'
            for name in ('UKFTractography_gpu.exe', 'ukf_gpu.dll', 'cudart64_12.dll',
                         'vcruntime140.dll', 'vcruntime140_1.dll', 'msvcp140.dll'):
                (runtime/name).write_bytes(data)
            environments = []
            real_run = subprocess.run

            def run(argv, **kwargs):
                if str(argv[0]).endswith('.exe'):
                    environments.append(kwargs['env'])
                    return subprocess.CompletedProcess(argv, 0, stdout=b'fixture')
                return real_run(argv, **kwargs)

            with patch('package_portable_installer.platform.system', return_value='Windows'), \
                 patch.dict(os.environ, {'SYSTEMROOT': r'C:\Windows', 'PYTHONPATH': 'SlicerPython'}), \
                 patch('package_portable_installer.subprocess.run', side_effect=run):
                archive = create_bundle(repository, {'windows-x86_64': runtime.parent}, root/'installer.zip')
            self.assertEqual(len(environments), 1)
            self.assertNotIn('PYTHONPATH', environments[0])
            self.assertTrue(environments[0]['PATH'].endswith(r';C:\Windows\System32;C:\Windows'))
            with zipfile.ZipFile(archive) as bundle:
                names = bundle.namelist()
                self.assertIn('UKFGPU-Installer/install.py', names)
                self.assertIn('UKFGPU-Installer/manifest.json', names)
                self.assertFalse(any('/manifest-' in name for name in names))
                self.assertIn('UKFGPU-Installer/runtimes/windows-x86_64/bin/UKFTractography_gpu.exe', names)
                self.assertFalse(any('/linux-x86_64/' in name for name in names))


if __name__ == '__main__':
    unittest.main()
