"""Run with Slicer --no-main-window --python-script (see README)."""
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import qt
import slicer
import vtk
from vtk.util.numpy_support import vtk_to_numpy

from UKFGPU import UKFGPULogic


def wait(logic, timeout=90):
    deadline = time.monotonic() + timeout
    while logic.process is not None and time.monotonic() < deadline:
        slicer.app.processEvents()
        time.sleep(0.01)
    assert logic.process is None, "Backend timed out"


def polydata(path):
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(str(path))
    reader.ReadAllScalarsOn()
    reader.ReadAllVectorsOn()
    reader.ReadAllTensorsOn()
    reader.Update()
    return reader.GetOutput()


try:
    repository = Path(os.environ["UKF_TEST_REPOSITORY"])
    output = Path(os.environ["UKF_TEST_OUTPUT"])
    output.mkdir(parents=True, exist_ok=True)
    backend = os.environ["UKF_GPU_EXECUTABLE"]
    data = repository/"UKFTractography/Data/Input"
    reference = repository.parent/"validation/precision-2026-10-11/all-models"
    # Create the actual registered widget, exercising selectors, controls and signals.
    widget = slicer.modules.ukfgpu.widgetRepresentation().self()
    assert widget.backend.text == backend
    assert widget.model.count == 12
    assert not hasattr(widget, "output")
    for control in [widget.dwi, widget.mask, widget.seeds, widget.seedLabels,
                    widget.model, widget.precision, widget.device,
                    *widget.parameters.values(), *widget.records.values()]:
        assert control.toolTip, control.className()
    print("PASS scene-first UI and hover help", flush=True)
    mask = slicer.util.loadLabelVolume(str(data/"mask.nhdr"))
    seeds = slicer.util.loadLabelVolume(str(data/"seed.nhdr"))
    for model, source, name, precision in [
            ("Simple2T", "two_tensor.nhdr", "simple2-mixed", 1),
            ("Simple1T-FW", "single_tensor_fw.nhdr", "simple1-fw-fp64", 0),
            ("NODDI1F", "single_tensor.nhdr", "noddi1-fp64", 0)]:
        dwi = slicer.util.loadVolume(str(data/source))
        assert dwi.IsA("vtkMRMLDiffusionWeightedVolumeNode"), dwi.GetClassName()
        results = []
        logic = widget.logic if model == "Simple2T" else UKFGPULogic()
        def completed(node, error):
            results.append((node, error))
            if model == "Simple2T":
                widget.onFinished(node, error)
        logic.onFinished = completed
        options = dict(numThreads=4, seedsPerVoxel=1, minBranchingAngle=0,
                       maxBranchingAngle=0, recordNMSE=True, recordState=True,
                       recordCovariance=True)
        options.update(dict(recordVic=True, recordKappa=True, recordViso=True)
                       if model.startswith("NODDI") else
                       dict(recordFA=True, recordTrace=True, recordTensors=True))
        if "FW" in model:
            options["recordFreeWater"] = True
        target = output/(name+".vtk")
        storage_path = dwi.GetStorageNode().GetFileName()
        if model == "Simple2T":
            widget.dwi.setCurrentNode(dwi)
            widget.mask.setCurrentNode(mask)
            widget.seeds.setCurrentNode(seeds)
            widget.parameters["numThreads"].value = 4
            for key, control in widget.records.items():
                control.checked = options.get(key, False)
            widget.run()
            assert not widget.runButton.enabled and widget.cancelButton.enabled
        else:
            logic.start(backend, dwi, mask, model, precision, 0, seeds, "1", options)
        temporary = Path(logic.jobDirectory)
        wait(logic)
        assert not temporary.exists() and not Path(logic.output).exists()
        assert dwi.GetStorageNode().GetFileName() == storage_path
        assert len(results) == 1 and results[0][1] is None, results
        node = results[0][0]
        expected = polydata(reference/(name+".vtk"))
        actual = node.GetPolyData()
        assert actual.GetNumberOfLines() == expected.GetNumberOfLines()
        assert np.array_equal(vtk_to_numpy(actual.GetLines().GetData()),
                              vtk_to_numpy(expected.GetLines().GetData()))
        delta = vtk_to_numpy(actual.GetPoints().GetData()).astype(float)-vtk_to_numpy(expected.GetPoints().GetData())
        assert np.linalg.norm(delta, axis=1).max(initial=0) <= 0.001
        for i in range(expected.GetPointData().GetNumberOfArrays()):
            array = expected.GetPointData().GetArray(i)
            values = actual.GetPointData().GetArray(array.GetName())
            assert values is not None, array.GetName()
            assert np.isfinite(vtk_to_numpy(values)).all()
        assert logic.jobDirectory is None
        assert node.GetAttribute("UKFGPU.Model") == model
        assert node.GetDisplayNode().GetVisibility()
        assert node.GetModifiedSinceRead()
        # Exercise Slicer's Save after all temporary tracking files are gone.
        assert slicer.util.saveNode(node, str(target))
        restored = slicer.util.loadNodeFromFile(str(target),
            "FiberBundleFile" if node.IsA("vtkMRMLFiberBundleNode") else "ModelFile")
        assert restored.GetPolyData().GetNumberOfLines() == actual.GetNumberOfLines()
        assert np.allclose(vtk_to_numpy(restored.GetPolyData().GetPoints().GetData()),
                           vtk_to_numpy(actual.GetPoints().GetData()), atol=0.001, rtol=0)
        assert restored.GetPolyData().GetPointData().GetNumberOfArrays() == actual.GetPointData().GetNumberOfArrays()
        if model == "Simple2T":
            assert widget.runButton.enabled and not widget.cancelButton.enabled
            assert "tractography done on device" in widget.log.toPlainText()
            if os.environ.get("UKF_TEST_SCREENSHOT"):
                slicer.util.selectModule("UKFGPU")
                slicer.app.processEvents()
                slicer.util.mainWindow().grab().save(os.environ["UKF_TEST_SCREENSHOT"])
        print("PASS Slicer export / GPU / scene / Save / reload:", model, node.GetClassName(), flush=True)
    # Cancel an actual QProcess before it can complete tracking.
    results = []
    logic = UKFGPULogic()
    logic.onFinished = lambda node, error: results.append((node, error))
    logic.start(backend, dwi, mask, seeds=seeds)
    target = Path(logic.output)
    temporary = Path(logic.jobDirectory)
    logic.cancel()
    wait(logic)
    assert results == [(None, "Cancelled")] and not target.exists(), results
    assert logic.jobDirectory is None and not temporary.exists()
    print("PASS cancellation and temporary-file cleanup", flush=True)
    # A failed executable must reset the UI lifecycle and preserve its log.
    bad = output/"invalid-backend"
    bad.write_text("invalid executable\n")
    bad.chmod(0o755)
    results = []
    logic = UKFGPULogic()
    logic.onFinished = lambda node, error: results.append((node, error))
    logic.start(str(bad), dwi, mask, seeds=seeds)
    temporary = Path(logic.jobDirectory)
    wait(logic)
    assert len(results) == 1 and results[0][0] is None and results[0][1]
    assert logic.jobDirectory is None and not temporary.exists()
    print("PASS failed process cleanup", flush=True)
    slicer.util.exit(0)
except Exception:
    traceback.print_exc()
    slicer.util.exit(1)
