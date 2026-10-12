import os
import json
import sys
from pathlib import Path
import shutil
import tempfile

import ctk
import qt
import slicer
import vtk
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule, ScriptedLoadableModuleWidget, ScriptedLoadableModuleLogic,
)
# Prefer this module's library over another installed UKFGPU version.
sys.path.insert(0, str(Path(__file__).parent))
from UKFGPULib import (MODELS, NUMERIC_OPTIONS, RECORD_OPTIONS, command,
                       process_environment, default_executable, check_result,
                       PRESETS, preset_values, model_options)


PARAMETER_HELP = {
    "seedsPerVoxel": "Number of seed positions per eligible voxel. More seeds increase tract density, runtime and memory use.",
    "numThreads": "CPU worker threads used by the backend; this does not select the number of GPUs.",
    "stepLength": "Tracking step size in mm. Smaller steps increase computation. Must be greater than zero.",
    "recordLength": "Distance between recorded fiber points in mm. Smaller spacing increases output size. Must be at least the tracking step size.",
    "maxHalfFiberLength": "Maximum tracking length in mm in each direction from a seed; a complete fiber can span both directions.",
    "seedingThreshold": "Minimum normalized mean DWI signal for eligible seeds. Higher thresholds restrict seeding; 0 requests the backend default.",
    "stoppingFA": "Stop tensor tracking when fractional anisotropy falls below this threshold (0 to 1). Lower values allow tracking into less anisotropic tissue. Not used by NODDI.",
    "stoppingThreshold": "Stop tracking when normalized mean DWI signal falls below this threshold (0 to 1). Lower values allow tracking through weaker signal.",
    "Qm": "Orientation process noise. Larger values allow directions to change more between steps. 0 selects the model-specific backend default.",
    "Ql": "Tensor eigenvalue process noise. Larger values allow diffusion eigenvalues to change more between steps. 0 selects the backend default. Not used by NODDI.",
    "Qw": "FreeWater fraction process noise. Used by FreeWater tensor models. 0 selects the backend default.",
    "Rs": "Expected measurement noise in the DWI signal. Larger values give the measurements less weight in the filter. 0 selects the model-specific backend default.",
    "Qkappa": "NODDI kappa process noise: controls how much the orientation concentration parameter may change between steps. Used by NODDI models only. 0 selects the model-specific backend default.",
    "Qvic": "NODDI compartment-fraction process noise for intracellular and isotropic fractions. Used by NODDI models only. 0 selects the model-specific backend default.",
    "minBranchingAngle": "Minimum angle in degrees between tensor directions for creating a branch. Branching requires a positive maximum angle and a compatible multi-tensor model.",
    "maxBranchingAngle": "Maximum branching angle in degrees (0 to 90). 0 disables branching. Branching increases work and fiber count; strict CPU/GPU agreement is not yet established.",
}
RECORD_HELP = {
    "recordFA": "Record tensor fractional anisotropy along fibers. Tensor models only.",
    "recordTrace": "Record the sum of tensor eigenvalues along fibers. Tensor models only.",
    "recordTensors": "Record fitted diffusion tensors, enabling tensor-based fiber coloring and analysis. Tensor models only.",
    "recordFreeWater": "Record the fitted free-water fraction along fibers. FreeWater tensor models only.",
    "recordVic": "Record the intracellular volume fraction along fibers. NODDI models only.",
    "recordKappa": "Record NODDI orientation concentration (kappa). Higher kappa indicates less orientation dispersion. NODDI models only.",
    "recordViso": "Record the isotropic compartment volume fraction along fibers. NODDI models only.",
    "recordNMSE": "Record normalized mean squared signal-fitting error along fibers.",
    "recordState": "Record the complete filter state at each fiber point. Intended for analysis/debugging; increases memory and saved scene size.",
    "recordCovariance": "Record filter covariance matrices at fiber points. Intended for debugging; can substantially increase memory and saved scene size.",
}


class UKFGPU(ScriptedLoadableModule):
    def __init__(self, parent):
        super().__init__(parent)
        parent.title = "UKF GPU Tractography"
        parent.categories = ["Diffusion.Tractography"]
        parent.dependencies = []
        parent.contributors = ["UKF GPU contributors", "Original UKFTractography authors"]
        parent.helpText = (
            "GPU UKF tracking with Simple, Full Tensor, FreeWater and NODDI models. "
            "Select a DWI volume with gradients, a brain mask and an optional seed labelmap. "
            "The backend requires Linux or Windows x86_64 and an NVIDIA CUDA GPU. "
            "Windows requires a separately built native backend; see the README. "
            "GPU support is experimental: some models have not passed strict CPU equivalence. "
            "See the extension README and precision acceptance report.")
        parent.acknowledgementText = "Based on pnlbwh/ukftractography."


class UKFGPUWidget(ScriptedLoadableModuleWidget):
    def setup(self):
        super().setup()
        self.logic = UKFGPULogic()
        self.logic.onLog = self.appendLog
        self.logic.onFinished = self.onFinished
        panel = qt.QWidget()
        form = qt.QFormLayout(panel)
        self.layout.addWidget(panel)
        self.backend = qt.QLineEdit(default_executable(Path(__file__).parent))
        row = qt.QHBoxLayout()
        row.addWidget(self.backend)
        browse = qt.QPushButton("Browse…")
        browse.connect("clicked()", self.browseBackend)
        row.addWidget(browse)
        self.addRow(form, "GPU executable", row, "Path to the native GPU backend. Packaged extensions select it automatically; use Browse to choose another executable.")
        self.backend.setToolTip("Path to the native GPU backend. Packaged extensions select it automatically.")
        browse.setToolTip("Select a compiled GPU backend executable.")
        self.dwi = self.selector(["vtkMRMLDiffusionWeightedVolumeNode"])
        self.mask = self.selector(["vtkMRMLLabelMapVolumeNode", "vtkMRMLScalarVolumeNode"])
        self.seeds = self.selector(["vtkMRMLLabelMapVolumeNode"], optional=True)
        self.addRow(form, "DWI volume", self.dwi, "Diffusion-weighted volume with gradient directions and b-values. Load a DWI NRRD, not a scalar volume.")
        self.addRow(form, "Brain mask", self.mask, "Brain mask aligned with the DWI. Tracking is restricted to nonzero mask voxels. Inputs must have no parent transforms.")
        self.addRow(form, "Seed labelmap (optional)", self.seeds, "Optional labelmap aligned with the DWI. When omitted, the backend seeds eligible voxels inside the brain mask.")
        self.seedLabels = qt.QLineEdit("1")
        self.addRow(form, "Seed labels (comma separated)", self.seedLabels, "Voxel labels to seed in the selected seed labelmap, for example 1,2. Default: 1. Ignored when no seed labelmap is selected.")
        self.model = qt.QComboBox()
        self.model.addItems(list(MODELS))
        self.model.setCurrentIndex(1)
        self.addRow(form, "Model", self.model, "1T/2T/3T indicate the number of tensor components. Simple uses axially symmetric tensors; Full allows three independent eigenvalues. FW adds free water; NODDI models neurite compartments. Default: Simple2T. Some models have not passed strict CPU/GPU acceptance.")
        self.precision = qt.QComboBox()
        self.precision.addItems(["FP64", "Mixed FP32 / FP64"])
        self.precision.setCurrentIndex(1)
        self.addRow(form, "Precision", self.precision, "Default: mixed FP32 / FP64. Mixed precision accelerates selected Simple-model signal calculations; sensitive calculations and Full Tensor, FreeWater and NODDI retain FP64.")
        self.device = qt.QComboBox()
        self.device.setEditable(True)
        self.device.addItem("0")
        self.gpuDetected = False
        self.gpuProbe = None
        self.gpuAvailable = True
        self.gpuTimer = qt.QTimer(self.parent)
        self.gpuTimer.setSingleShot(True)
        self.gpuTimer.connect("timeout()", self.gpuTimeout)
        gpuRow = qt.QHBoxLayout()
        gpuRow.addWidget(self.device)
        self.refreshGpuButton = qt.QPushButton("Refresh")
        self.refreshGpuButton.connect("clicked()", self.refreshGpu)
        self.refreshGpuButton.setToolTip("Query GPUs visible to the backend. CUDA_VISIBLE_DEVICES is respected.")
        gpuRow.addWidget(self.refreshGpuButton)
        gpuHelp = "Detected CUDA ordinal, GPU name and total memory. If detection fails, enter a CUDA device index manually. Refresh after changing the backend."
        self.addRow(form, "GPU device", gpuRow, gpuHelp)
        self.device.setToolTip(gpuHelp)
        self.gpuStatus = qt.QLabel()
        self.gpuStatus.wordWrap = True
        form.addRow(self.gpuStatus)
        self.fiberName = qt.QLineEdit()
        self.fiberName.setPlaceholderText("UKF_Simple2T")
        self.addRow(form, "Fiber name", self.fiberName, "Name of the result in the current scene. Leave empty to name it by model. Slicer automatically distinguishes duplicate names. This is not a save path.")
        self.preset = qt.QComboBox()
        self.preset.addItems(list(PRESETS))
        self.applyPresetButton = qt.QPushButton("Apply")
        self.applyPresetButton.setToolTip("Replace tracking parameters and clear recording options using the selected preset. Does not change model, GPU, input volumes or fiber name.")
        self.applyPresetButton.connect("clicked()", self.applyPreset)
        presetRow = qt.QHBoxLayout()
        presetRow.addWidget(self.preset)
        presetRow.addWidget(self.applyPresetButton)
        self.addRow(form, "Tracking preset", presetRow, "Choose a preset, review its description, then click Apply. Manual edits remain until you apply another preset.")
        self.preset.setToolTip("Presets replace tracking parameters only when Apply is clicked.")
        self.presetDescription = qt.QLabel(PRESETS['Standard'])
        self.presetDescription.wordWrap = True
        form.addRow(self.presetDescription)
        self.preset.connect("currentIndexChanged(int)", self.updatePresetDescription)
        resultHint = qt.QLabel("Fibers appear in the scene after tracking. Use Slicer's Save to keep them.")
        resultHint.wordWrap = True
        resultHint.setToolTip("No output path is required. Temporary tracking files are deleted after import; fibers remain in the current scene until saved or removed.")
        form.addRow(resultHint)

        advanced = ctk.ctkCollapsibleButton()
        advanced.text = "Tracking and recording"
        advanced.collapsed = True
        self.layout.addWidget(advanced)
        parameters = qt.QFormLayout(advanced)
        # Zero noise values request the backend's model-specific defaults.
        defaults = preset_values("Standard", self.model.currentText)
        self.parameters = {}
        for name in NUMERIC_OPTIONS:
            control = qt.QDoubleSpinBox()
            control.setDecimals(6)
            control.setRange(0, 1000000)
            if name in ("seedsPerVoxel", "numThreads", "maxHalfFiberLength"):
                control.setDecimals(0)
                control.setMinimum(1)
            control.setValue(defaults[name])
            self.parameters[name] = control
            self.addRow(parameters, name, control, PARAMETER_HELP[name] + " Default: " + str(defaults[name]) + ".")
        self.records = {}
        for name in RECORD_OPTIONS:
            control = qt.QCheckBox()
            self.records[name] = control
            self.addRow(parameters, name, control, RECORD_HELP[name] + " Default: off. Recording adds memory use.")

        row = qt.QHBoxLayout()
        self.runButton = qt.QPushButton("Run GPU tracking")
        self.cancelButton = qt.QPushButton("Cancel")
        self.cancelButton.enabled = False
        self.runButton.setToolTip("Run GPU tracking and add the resulting fibers to the current Slicer scene.")
        self.cancelButton.setToolTip("Stop the current tracking job and delete its temporary files.")
        self.runButton.connect("clicked()", self.run)
        self.cancelButton.connect("clicked()", self.logic.cancel)
        row.addWidget(self.runButton)
        row.addWidget(self.cancelButton)
        self.layout.addLayout(row)
        self.status = qt.QLabel("Ready")
        self.layout.addWidget(self.status)
        self.log = qt.QPlainTextEdit()
        self.log.readOnly = True
        self.log.setMaximumBlockCount(4000)
        self.layout.addWidget(self.log)
        self.layout.addStretch(1)
        self.model.connect("currentIndexChanged(int)", self.updateModelOptions)
        self.seeds.connect("currentNodeChanged(vtkMRMLNode*)", self.updateModelOptions)
        self.backend.connect("editingFinished()", self.refreshGpu)
        self.updateModelOptions()
        qt.QTimer.singleShot(0, self.refreshGpu)

    def addRow(self, form, title, field, help_text):
        label = qt.QLabel(title)
        label.setToolTip(help_text)
        if hasattr(field, "setToolTip"):
            field.setToolTip(help_text)
        form.addRow(label, field)

    def updateModelOptions(self, *unused):
        numeric, records = model_options(self.model.currentText, self.seeds.currentNode() is not None)
        for name, enabled in numeric.items():
            self.parameters[name].enabled = enabled
            if not enabled and name in ("minBranchingAngle", "maxBranchingAngle"):
                self.parameters[name].value = 0
        for name, enabled in records.items():
            self.records[name].enabled = enabled
            if not enabled:
                self.records[name].checked = False
        simple = self.model.currentText in ("Simple1T", "Simple2T", "Simple3T")
        self.precision.enabled = simple
        if not simple:
            self.precision.currentIndex = 0
        self.seedLabels.enabled = self.seeds.currentNode() is not None
        self.fiberName.setPlaceholderText("UKF_" + self.model.currentText)

    def updatePresetDescription(self, *unused):
        self.presetDescription.text = PRESETS[self.preset.currentText]

    def applyPreset(self):
        for name, value in preset_values(self.preset.currentText, self.model.currentText).items():
            self.parameters[name].value = value
        for control in self.records.values():
            control.checked = False
        self.updateModelOptions()
        self.status.text = "Applied preset: " + self.preset.currentText

    def gpuDevice(self):
        if self.gpuDetected:
            return int(self.device.itemData(self.device.currentIndex))
        device = int(self.device.currentText)
        if device < 0:
            raise ValueError("GPU device index must be nonnegative.")
        return device

    def refreshGpu(self):
        if self.gpuProbe is not None or self.logic.process is not None:
            return
        self.gpuStatus.text = "Detecting CUDA GPUs..."
        self.refreshGpuButton.enabled = False
        self.runButton.enabled = False
        self.gpuProbe = qt.QProcess(self.parent)
        environment = qt.QProcessEnvironment()
        for key, value in process_environment(slicer.util.startupEnvironment()).items():
            environment.insert(key, value)
        self.gpuProbe.setProcessEnvironment(environment)
        self.gpuProbe.setProcessChannelMode(qt.QProcess.MergedChannels)
        self.gpuProbe.connect("finished(int,QProcess::ExitStatus)", self.gpuFinished)
        self.gpuProbe.connect("errorOccurred(QProcess::ProcessError)", self.gpuError)
        python = Path(slicer.app.slicerHome)/"bin"/("PythonSlicer.exe" if os.name == "nt" else "PythonSlicer")
        library = Path(__file__).parent/"UKFGPULib/__init__.py"
        self.gpuProbe.start(str(python), [str(library), self.backend.text])
        self.gpuTimer.start(10000)

    def gpuTimeout(self):
        if self.gpuProbe is not None:
            self.gpuProbe.kill()
            self.finishGpuDiscovery(None, "GPU detection timed out; enter a CUDA index manually.")

    def gpuError(self, error):
        if error == qt.QProcess.FailedToStart:
            qt.QTimer.singleShot(0, lambda: self.finishGpuDiscovery(None, "GPU detection could not start; enter a CUDA index manually."))

    def gpuFinished(self, exit_code, exit_status):
        if self.gpuProbe is None:
            return
        text = self.gpuProbe.readAllStandardOutput().data().decode("utf-8", errors="replace")
        try:
            if exit_code:
                raise RuntimeError(text.strip().splitlines()[-1] if text.strip() else "GPU detection failed.")
            self.finishGpuDiscovery(json.loads(text), "")
        except (ValueError, RuntimeError) as exc:
            self.finishGpuDiscovery(None, str(exc))

    def finishGpuDiscovery(self, devices, message):
        if self.gpuProbe is None:
            return
        self.gpuTimer.stop()
        self.gpuProbe.deleteLater()
        self.gpuProbe = None
        selected = self.device.itemData(self.device.currentIndex) if self.gpuDetected else 0
        self.device.clear()
        self.gpuDetected = devices is not None and bool(devices)
        self.gpuAvailable = devices is None or bool(devices)
        self.device.setEditable(devices is None)
        if devices is None:
            self.device.addItem("0")
            self.gpuStatus.text = message + " Manual CUDA index is available."
        elif not devices:
            self.gpuStatus.text = "No CUDA GPU is visible. Check the driver and CUDA_VISIBLE_DEVICES, then Refresh."
        else:
            for device in devices:
                self.device.addItem("CUDA %d: %s (%.1f GiB)" % (device['index'], device['name'], device['memory']/2**30), device['index'])
            index = self.device.findData(selected)
            self.device.currentIndex = max(0, index)
            self.gpuStatus.text = "%d CUDA GPU(s) detected." % len(devices)
        self.device.enabled = self.gpuAvailable
        self.refreshGpuButton.enabled = True
        self.runButton.enabled = self.gpuAvailable and self.logic.process is None

    def selector(self, types, optional=False):
        selector = slicer.qMRMLNodeComboBox()
        selector.nodeTypes = types
        selector.noneEnabled = optional
        selector.addEnabled = False
        selector.removeEnabled = False
        selector.setMRMLScene(slicer.mrmlScene)
        return selector

    def browseBackend(self):
        path = qt.QFileDialog.getOpenFileName(self.parent, "Select GPU executable", self.backend.text)
        if path:
            self.backend.text = path
            self.refreshGpu()

    def appendLog(self, text):
        cursor = self.log.textCursor()
        cursor.movePosition(qt.QTextCursor.End)
        cursor.insertText(text)
        self.log.setTextCursor(cursor)
        self.log.ensureCursorVisible()

    def run(self):
        self.log.clear()
        options = {name: control.value for name, control in self.parameters.items() if control.enabled}
        for name in ("seedsPerVoxel", "numThreads", "maxHalfFiberLength"):
            options[name] = int(options[name])
        options.update({name: control.checked for name, control in self.records.items() if control.enabled})
        try:
            self.logic.start(self.backend.text, self.dwi.currentNode(), self.mask.currentNode(),
                             self.model.currentText, self.precision.currentIndex,
                             self.gpuDevice(), self.seeds.currentNode(), self.seedLabels.text, options,
                             result_name=self.fiberName.text)
            self.runButton.enabled = False
            self.refreshGpuButton.enabled = False
            self.applyPresetButton.enabled = False
            self.cancelButton.enabled = True
            self.status.text = "Tracking on GPU…"
        except Exception as exc:
            slicer.util.errorDisplay(str(exc))

    def onFinished(self, node, error):
        self.runButton.enabled = self.gpuAvailable and self.gpuProbe is None
        self.refreshGpuButton.enabled = True
        self.applyPresetButton.enabled = True
        self.cancelButton.enabled = False
        self.status.text = error or "Added to scene: " + node.GetName() + ". Use Save to keep fibers."
        if error and error != "Cancelled":
            slicer.util.errorDisplay(error)

    def cleanup(self):
        self.gpuTimer.stop()
        if self.gpuProbe is not None:
            self.gpuProbe.kill()
            self.gpuProbe.deleteLater()
            self.gpuProbe = None
        self.logic.onLog = None
        self.logic.onFinished = None
        self.logic.cancel()


class UKFGPULogic(ScriptedLoadableModuleLogic):
    def __init__(self):
        super().__init__()
        self.process = None
        self.jobDirectory = None
        self.onLog = None
        self.onFinished = None
        self.lastLog = ""
        self.cancelled = False

    def start(self, executable, dwi, mask, model="Simple2T", precision=1,
              device=0, seeds=None, seed_labels="1", options=None, result_name=""):
        if self.process is not None:
            raise RuntimeError("Tracking is already running.")
        if not os.path.isfile(executable) or not os.access(executable, os.X_OK):
            raise ValueError("Select an executable UKF GPU backend.")
        executable = str(Path(executable).expanduser().resolve())
        if dwi is None or not dwi.IsA("vtkMRMLDiffusionWeightedVolumeNode"):
            raise ValueError("Select a diffusion-weighted volume with gradient metadata.")
        if dwi.GetNumberOfGradients() == 0:
            raise ValueError("DWI gradient metadata is missing.")
        if mask is None:
            raise ValueError("Select a brain mask.")
        for node in (dwi, mask, seeds):
            if node and node.GetParentTransformNode():
                raise ValueError("Inputs with parent transforms are not supported; use aligned volumes without parent transforms.")
        # Build argv before exporting so invalid model/options leave no temporary files.
        args = command(executable, "dwi.nrrd", "mask.nrrd", "fibers.vtk", model,
                       precision, device, "seeds.nrrd" if seeds else None, seed_labels, options)
        self.jobDirectory = tempfile.mkdtemp(prefix="UKFGPU-", dir=slicer.app.temporaryPath)
        self.output = str(Path(self.jobDirectory) / "fibers.vtk")
        self.model = model
        self.resultName = result_name.strip() or "UKF_" + model
        self.precision = precision
        try:
            for node, name in ((dwi, "dwi"), (mask, "mask"), (seeds, "seeds")):
                if node and not slicer.util.exportNode(node, os.path.join(self.jobDirectory, name + ".nrrd")):
                    raise RuntimeError("Could not export " + name)
            self.lastLog = ""
            self.cancelled = False
            self.process = qt.QProcess(slicer.app)
            environment = qt.QProcessEnvironment()
            for key, value in process_environment(slicer.util.startupEnvironment(), executable).items():
                environment.insert(key, value)
            self.process.setProcessEnvironment(environment)
            self.process.setWorkingDirectory(self.jobDirectory)
            self.process.setProcessChannelMode(qt.QProcess.MergedChannels)
            self.process.connect("readyReadStandardOutput()", self.readLog)
            self.process.connect("finished(int,QProcess::ExitStatus)", self.finished)
            self.process.connect("errorOccurred(QProcess::ProcessError)", self.processError)
            self.process.start(args[0], args[1:])
        except Exception:
            self.release()
            raise

    def readLog(self):
        text = self.process.readAllStandardOutput().data().decode("utf-8", errors="replace")
        self.lastLog += text
        if self.onLog:
            self.onLog(text)

    def processError(self, error):
        if error == qt.QProcess.FailedToStart:
            message = "Could not start backend: " + self.process.errorString()
            # FailedToStart may be emitted inside QProcess.start(). Release it
            # only after that C++ call has returned to the event loop.
            qt.QTimer.singleShot(0, lambda: self.complete(None, message))

    def finished(self, exit_code, exit_status):
        self.readLog()
        node = None
        error = None
        try:
            if self.cancelled:
                error = "Cancelled"
            else:
                check_result(exit_code, self.lastLog, self.output)
                # UKF writes RAS coordinates without a VTK coordinate-system tag.
                # Read directly: Slicer's generic model reader defaults to LPS.
                reader = vtk.vtkPolyDataReader()
                reader.SetFileName(self.output)
                reader.ReadAllScalarsOn()
                reader.ReadAllVectorsOn()
                reader.ReadAllTensorsOn()
                reader.Update()
                if reader.GetErrorCode():
                    raise RuntimeError("Could not read the generated fiber bundle.")
                node_class = ("vtkMRMLFiberBundleNode" if
                              slicer.mrmlScene.IsNodeClassRegistered("vtkMRMLFiberBundleNode")
                              else "vtkMRMLModelNode")
                node = slicer.mrmlScene.AddNewNodeByClass(node_class,
                    slicer.mrmlScene.GenerateUniqueName(self.resultName))
                node.SetAndObservePolyData(reader.GetOutput())
                node.CreateDefaultDisplayNodes()
                node.SetAttribute("UKFGPU.Model", self.model)
                node.SetAttribute("UKFGPU.Precision", "FP64" if self.precision == 0 else "Mixed FP32 / FP64")
                if slicer.app.layoutManager():
                    slicer.util.resetThreeDViews()
        except Exception as exc:
            error = str(exc)
        self.complete(node, error)

    def complete(self, node, error):
        self.release()
        if self.onFinished:
            self.onFinished(node, error)

    def release(self):
        if self.process is not None:
            self.process.deleteLater()
            self.process = None
        if self.jobDirectory:
            shutil.rmtree(self.jobDirectory)
            self.jobDirectory = None

    def cancel(self):
        if self.process is not None:
            self.cancelled = True
            self.process.kill()
