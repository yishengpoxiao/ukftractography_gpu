"""Install an archive using Slicer's real manager, into isolated test settings."""
import os
from pathlib import Path
import traceback
import slicer

try:
    root = Path(os.environ['UKF_INSTALL_TEST_OUTPUT']).resolve()
    root.mkdir(parents=True, exist_ok=False)
    settings = root/'extensions-2026-10-11.ini'
    settings.write_text('[Extensions]\nInstallPath=' + str(root/'extensions') + '\n')
    manager = slicer.qSlicerExtensionsManagerModel(slicer.app)
    manager.extensionsSettingsFilePath = str(settings)
    app_manager = slicer.app.extensionsManagerModel()
    manager.slicerRevision = app_manager.slicerRevision
    manager.slicerOs = app_manager.slicerOs
    manager.slicerArch = app_manager.slicerArch
    manager.interactive = False
    assert manager.installExtension(os.environ['UKF_INSTALL_ARCHIVE'], False, True)
    assert manager.isExtensionInstalled('UKFGPU')
    paths = manager.extensionModulePaths('UKFGPU')
    assert any((Path(path)/'UKFGPU.py').is_file() for path in paths), paths
    print('PASS Extensions Manager install from archive:', paths, flush=True)
    slicer.util.exit(0)
except Exception:
    traceback.print_exc()
    slicer.util.exit(1)
