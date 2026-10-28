"""Installer integrity, platform selection and version-specific archive contracts."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

extension = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('installer', extension/'install.py')
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


def fixture(root):
    for name in ('install.py', 'module/UKFGPU.py', 'module/UKFGPULib/__init__.py',
                 'docs/LICENSE.txt', 'runtimes/linux-x86_64/bin/ukf-gpu',
                 'runtimes/linux-x86_64/bin/UKFTractography_gpu',
                 'runtimes/windows-x86_64/bin/UKFTractography_gpu.exe'):
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('fixture')
    manifest = dict(format=1, source_commit='fixture',
                    runtimes={'linux-x86_64': {'glibc_min': '2.29'}, 'windows-x86_64': {}},
                    sha256={p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in root.rglob('*') if p.is_file()})
    (root/'manifest.json').write_text(json.dumps(manifest))


class InstallerTest(unittest.TestCase):
    @patch.object(installer.platform, 'libc_ver', return_value=('glibc', '2.31'))
    def test_version_layout_platform_selection_and_exclusive_output(self, unused):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            for version in ((5, 2), (5, 6), (5, 12), (6, 0)):
                for system, binary in (('Linux', 'ukf-gpu'), ('Windows', 'UKFTractography_gpu.exe')):
                    output = root/('%s-%s.tar.gz' % (version[1], system))
                    installer.prepare_archive(root, output, version, 'test-revision', system, 'AMD64')
                    base = 'UKFGPU/lib/Slicer-%d.%d/qt-scripted-modules/' % version
                    with tarfile.open(output) as archive:
                        self.assertIn(base+'UKFGPU.py', archive.getnames())
                        self.assertIn(base+'Runtime/bin/'+binary, archive.getnames())
                        other = 'UKFTractography_gpu.exe' if system == 'Linux' else 'ukf-gpu'
                        self.assertNotIn(base+'Runtime/bin/'+other, archive.getnames())
                        if system == 'Linux':
                            self.assertEqual(archive.getmember(base+'Runtime/bin/ukf-gpu').mode & 0o111, 0o111)
                    with self.assertRaises(FileExistsError):
                        installer.prepare_archive(root, output, version, 'test-revision', system, 'AMD64')
            with self.assertRaisesRegex(RuntimeError, 'Slicer 5.2'):
                installer.prepare_archive(root, root/'unsupported.tar.gz', (5, 1), 'r')

    def test_tampered_missing_and_unlisted_payload_rejected(self):
        for mode in ('modified', 'missing', 'unlisted'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                fixture(root)
                if mode == 'modified':
                    (root/'module/UKFGPU.py').write_text('changed')
                elif mode == 'missing':
                    (root/'module/UKFGPU.py').unlink()
                else:
                    (root/'module/unlisted.py').touch()
                with self.assertRaises(RuntimeError):
                    installer.verified_manifest(root)

    def test_generated_python_cache_does_not_invalidate_installer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            cache = root/'module/UKFGPULib/__pycache__'
            cache.mkdir()
            (cache/'__init__.cpython-39.pyc').write_bytes(b'generated-cache')
            installer.verified_manifest(root)

    def test_missing_native_backend_unsupported_architecture_and_glibc(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            manifest = installer.verified_manifest(root)
            del manifest['runtimes']['windows-x86_64']
            with self.assertRaisesRegex(RuntimeError, 'no native windows'):
                installer.select_runtime(root, manifest, 'Windows', 'AMD64')
            with patch.object(installer.platform, 'libc_ver', return_value=('glibc', '2.28')):
                with self.assertRaisesRegex(RuntimeError, 'glibc >= 2.29'):
                    installer.select_runtime(root, manifest, 'Linux', 'x86_64')
        for system, machine in (('Darwin', 'x86_64'), ('Linux', 'aarch64')):
            with self.assertRaises(RuntimeError):
                installer.platform_key(system, machine)


if __name__ == '__main__':
    unittest.main()
