"""Verify a packaged module's actual registration, widget and native loader."""
import os
from pathlib import Path
import subprocess
import time
import traceback
import slicer

try:
    assert slicer.app.moduleManager().module('UKFGPU') is not None
    widget = slicer.modules.ukfgpu.widgetRepresentation().self()
    assert Path(widget.backend.text).is_file()
    from UKFGPULib import process_environment
    result = subprocess.run([widget.backend.text, '--help'],
                            env=process_environment(slicer.util.startupEnvironment(), widget.backend.text),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
    assert result.returncode == 0, result.stdout.decode(errors='replace')
    deadline = time.monotonic()+40
    while widget.gpuProbe is not None:
        slicer.app.processEvents()
        assert time.monotonic() < deadline, 'CUDA discovery did not finish'
        time.sleep(.01)
    print('PASS installed module registered, widget created, native dependencies loaded:',
          slicer.app.applicationVersion, widget.gpuStatus.text, flush=True)
    slicer.util.exit(0)
except Exception:
    traceback.print_exc()
    slicer.util.exit(1)
