"""Install the actual portable bundle through an isolated Extensions Manager."""
from datetime import date
import importlib.util
import json
import os
from pathlib import Path
import traceback
import slicer

try:
    root = Path(os.environ['UKF_INSTALL_TEST_OUTPUT'])
    bundle = Path(os.environ['UKF_INSTALL_BUNDLE'])
    spec = importlib.util.spec_from_file_location('installer', bundle/'install.py')
    installer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(installer)
    manifest = installer.verified_manifest(bundle)
    runtime = installer.select_runtime(bundle, manifest)
    if os.name != 'nt':
        for file in (runtime/'bin').iterdir():
            file.chmod(0o755)
    status = installer.preflight(bundle, runtime, slicer)
    settings = root/('extensions-'+date.today().isoformat()+'.ini')
    with settings.open('x', encoding='utf-8') as file:
        file.write('[Extensions]\nInstallPath='+(root/'extensions').as_posix()+'\n')
    manager = slicer.qSlicerExtensionsManagerModel(slicer.app)
    manager.extensionsSettingsFilePath = str(settings)
    original = slicer.app.extensionsManagerModel()
    for name in ('slicerRevision', 'slicerOs', 'slicerArch'):
        setattr(manager, name, getattr(original, name))
    manager.interactive = False
    archive = root/'UKFGPU.tar.gz'
    installer.prepare_archive(bundle, archive, (slicer.app.majorVersion, slicer.app.minorVersion), manager.slicerRevision)
    assert manager.installExtension(str(archive), False, True)
    assert manager.isExtensionInstalled('UKFGPU')
    paths = list(manager.extensionModulePaths('UKFGPU'))
    assert any((Path(path)/'UKFGPU.py').is_file() for path in paths)
    (root/'installed.json').write_text(json.dumps(dict(slicer=slicer.app.applicationVersion,
                                                     paths=paths, preflight=status)), encoding='utf-8')
    print('PASS portable bundle installation:', paths, status, flush=True)
    slicer.util.exit(0)
except Exception:
    traceback.print_exc()
    slicer.util.exit(1)
