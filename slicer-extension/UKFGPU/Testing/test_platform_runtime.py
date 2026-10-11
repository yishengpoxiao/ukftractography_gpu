"""Platform-specific packaging and DLL isolation contracts (no Windows execution)."""
import importlib.util
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

repository = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repository/'slicer-extension/UKFGPU'))
from UKFGPULib import command, default_executable, process_environment
spec = importlib.util.spec_from_file_location('packager', repository/'scripts/package_slicer_extension.py')
packager = importlib.util.module_from_spec(spec)
spec.loader.exec_module(packager)


def pe_fixture(path, machine=0x8664):
    # Structural test fixture only: never executed or delivered as a backend.
    data = bytearray(128)
    data[:2] = b'MZ'
    struct.pack_into('<I', data, 60, 64)
    data[64:68] = b'PE\0\0'
    struct.pack_into('<H', data, 68, machine)
    path.write_bytes(data)


class PlatformRuntimeTest(unittest.TestCase):
    def test_windows_environment(self):
        env = process_environment({'SystemRoot': r'C:\Windows', 'Path': r'C:\Slicer\bin',
                                   'Qt_Plugin_Path': 'SlicerQt', 'PYTHONPATH': 'SlicerPython',
                                   'CUDA_VISIBLE_DEVICES': '1'}, r'C:\UKF GPU\Runtime\bin\UKFTractography_gpu.exe', 'win32')
        self.assertNotIn('Path', env)
        self.assertNotIn('Qt_Plugin_Path', env)
        self.assertNotIn('PYTHONPATH', env)
        self.assertNotIn('Slicer', env['PATH'])
        self.assertTrue(env['PATH'].startswith(r'C:\UKF GPU\Runtime\bin;'))
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '1')

    def test_windows_command_and_discovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            module = Path(tmp)
            binary = module/'Runtime/bin/UKFTractography_gpu.exe'
            binary.parent.mkdir(parents=True)
            binary.touch()
            with patch.dict('os.environ', {}, clear=True):
                self.assertEqual(default_executable(module, 'win32'), str(binary))
        argv = command(r'C:\GPU 模块\UKF.exe', 'DWI 含空格.nrrd', 'mask.nrrd', 'result.vtk', 'Full2T-FW')
        self.assertEqual(argv[2], 'DWI 含空格.nrrd')
        self.assertIn('--fullTensorModel', argv)
        self.assertIn('--freeWater', argv)

    def test_windows_package_layout_and_reject_foreign_binary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            crt = root/'CRT'
            crt.mkdir()
            binary, gpu, cuda = [root/name for name in ('UKFTractography_gpu.exe', 'ukf_gpu.dll', 'cudart64_12.dll')]
            for path in [binary, gpu, cuda] + [crt/name for name in ('vcruntime140.dll', 'vcruntime140_1.dll', 'msvcp140.dll')]:
                pe_fixture(path)
            args = SimpleNamespace(platform='windows', repository=repository, output=root/'package',
                                   binary=binary, gpu_library=gpu, cuda_runtime=cuda,
                                   runtime_dir=[crt], slicer_version='5.6')
            archive = packager.package(args)
            self.assertTrue(archive.is_file())
            installed = args.output/'install-tree/UKFGPU'
            runtime = installed/'lib/Slicer-5.6/qt-scripted-modules/Runtime/bin'
            self.assertTrue((runtime/'UKFTractography_gpu.exe').is_file())
            self.assertTrue((runtime/'ukf_gpu.dll').is_file())
            self.assertFalse((runtime/'ukf-gpu').exists())
            self.assertTrue((installed/'share/Slicer-5.6/UKFGPU.s4ext').is_file())
            binary.write_bytes(b'\x7fELF')
            with self.assertRaisesRegex(ValueError, 'Not a Windows PE'):
                packager.check_windows_x64(binary)
            pe_fixture(binary, machine=0x14c)
            with self.assertRaisesRegex(ValueError, 'Not a Windows x64'):
                packager.check_windows_x64(binary)
            (runtime/'msvcp140.dll').unlink()
            # Missing CRT is detected at packaging time, not after installation.
            (crt/'msvcp140.dll').unlink()
            pe_fixture(binary)
            with self.assertRaisesRegex(ValueError, 'Missing MSVC runtime'):
                packager.copy_runtime(args, root/'missing-crt')


if __name__ == '__main__':
    unittest.main()
