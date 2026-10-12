import os
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
from UKFGPULib import (MODELS, NUMERIC_OPTIONS, RECORD_OPTIONS, command,
                       process_environment, default_executable, check_result)


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
    "Qkappa": "NODDI kappa process noise: controls how much the orientation concentration parameter may change between steps. Used by NODDI models only.",
    "Qvic": "NODDI compartment-fraction process noise for intracellular and isotropic fractions. Used by NODDI models only.",
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
        self.device = qt.QSpinBox()
        self.device.setRange(0, 63)
        self.addRow(form, "GPU device", self.device, "Zero-based GPU index. Default: 0. Indices follow CUDA_VISIBLE_DEVICES when set.")
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
        defaults = dict(seedsPerVoxel=1, numThreads=16, stepLength=0.3, recordLength=0.9,
                        maxHalfFiberLength=250, seedingThreshold=0.18, stoppingFA=0.15,
                        stoppingThreshold=0.1, Qm=0, Ql=0, Qw=0, Rs=0,
                        Qkappa=0.01, Qvic=0.004, minBranchingAngle=0, maxBranchingAngle=0)
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

    def addRow(self, form, title, field, help_text):
        label = qt.QLabel(title)
        label.setToolTip(help_text)
        if hasattr(field, "setToolTip"):
            field.setToolTip(help_text)
        form.addRow(label, field)

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

    def appendLog(self, text):
        cursor = self.log.textCursor()
        cursor.movePosition(qt.QTextCursor.End)
        cursor.insertText(text)
        self.log.setTextCursor(cursor)
        self.log.ensureCursorVisible()

    def run(self):
        self.log.clear()
        options = {name: control.value for name, control in self.parameters.items()}
        for name in ("seedsPerVoxel", "numThreads", "maxHalfFiberLength"):
            options[name] = int(options[name])
        options.update({name: control.checked for name, control in self.records.items()})
        try:
            self.logic.start(self.backend.text, self.dwi.currentNode(), self.mask.currentNode(),
                             self.model.currentText, self.precision.currentIndex,
                             self.device.value, self.seeds.currentNode(), self.seedLabels.text, options)
            self.runButton.enabled = False
            self.cancelButton.enabled = True
            self.status.text = "Tracking on GPU…"
        except Exception as exc:
            slicer.util.errorDisplay(str(exc))

    def onFinished(self, node, error):
        self.runButton.enabled = True
        self.cancelButton.enabled = False
        self.status.text = error or "Added to scene: " + node.GetName() + ". Use Save to keep fibers."
        if error and error != "Cancelled":
            slicer.util.errorDisplay(error)

    def cleanup(self):
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
              device=0, seeds=None, seed_labels="1", options=None):
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
                node = slicer.mrmlScene.AddNewNodeByClass(node_class, "UKF_" + self.model)
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
