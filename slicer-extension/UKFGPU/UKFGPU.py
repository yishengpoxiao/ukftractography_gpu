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
        form.addRow("GPU executable", row)
        self.dwi = self.selector(["vtkMRMLDiffusionWeightedVolumeNode"])
        self.mask = self.selector(["vtkMRMLLabelMapVolumeNode", "vtkMRMLScalarVolumeNode"])
        self.seeds = self.selector(["vtkMRMLLabelMapVolumeNode"], optional=True)
        form.addRow("DWI volume", self.dwi)
        form.addRow("Brain mask", self.mask)
        form.addRow("Seed labelmap (optional)", self.seeds)
        self.seedLabels = qt.QLineEdit("1")
        form.addRow("Seed labels (comma separated)", self.seedLabels)
        self.model = qt.QComboBox()
        self.model.addItems(list(MODELS))
        self.model.setCurrentIndex(1)
        form.addRow("Model", self.model)
        self.precision = qt.QComboBox()
        self.precision.addItems(["FP64", "Mixed FP32 / FP64"])
        self.precision.setCurrentIndex(1)
        self.precision.setToolTip("Mixed precision accelerates Simple models. Full Tensor, FreeWater and NODDI retain FP64.")
        form.addRow("Precision", self.precision)
        self.device = qt.QSpinBox()
        self.device.setRange(0, 63)
        self.device.setToolTip("Device index follows CUDA_VISIBLE_DEVICES if it is set.")
        form.addRow("GPU device", self.device)
        self.output = qt.QLineEdit()
        row = qt.QHBoxLayout()
        row.addWidget(self.output)
        browseOutput = qt.QPushButton("Save as…")
        browseOutput.connect("clicked()", self.browseOutput)
        row.addWidget(browseOutput)
        form.addRow("Output fibers (.vtk)", row)

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
            parameters.addRow(name, control)
        self.records = {}
        for name in RECORD_OPTIONS:
            control = qt.QCheckBox()
            self.records[name] = control
            parameters.addRow(name, control)

        row = qt.QHBoxLayout()
        self.runButton = qt.QPushButton("Run GPU tracking")
        self.cancelButton = qt.QPushButton("Cancel")
        self.cancelButton.enabled = False
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

    def browseOutput(self):
        path = qt.QFileDialog.getSaveFileName(self.parent, "Save fiber bundle", "UKFGPU.vtk", "VTK fibers (*.vtk)")
        if path:
            self.output.text = path if path.lower().endswith(".vtk") else path + ".vtk"

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
                             self.output.text, self.model.currentText, self.precision.currentIndex,
                             self.device.value, self.seeds.currentNode(), self.seedLabels.text, options)
            self.runButton.enabled = False
            self.cancelButton.enabled = True
            self.status.text = "Tracking on GPU…"
        except Exception as exc:
            slicer.util.errorDisplay(str(exc))

    def onFinished(self, node, error):
        self.runButton.enabled = True
        self.cancelButton.enabled = False
        self.status.text = error or "Completed: " + node.GetName()
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

    def start(self, executable, dwi, mask, output, model="Simple2T", precision=1,
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
        if not output or Path(output).suffix.lower() != ".vtk":
            raise ValueError("Select a new .vtk output file.")
        output = Path(output).expanduser().resolve()
        if output.exists() or Path(str(output) + ".log").exists():
            raise ValueError("Output or log already exists. Choose a new filename.")
        if not output.parent.is_dir():
            raise ValueError("The output directory does not exist.")
        for node in (dwi, mask, seeds):
            if node and node.GetParentTransformNode():
                raise ValueError("Inputs with parent transforms are not supported; use aligned volumes without parent transforms.")
        # Build argv before exporting so invalid model/options leave no temporary files.
        args = command(executable, "dwi.nrrd", "mask.nrrd", output, model,
                       precision, device, "seeds.nrrd" if seeds else None, seed_labels, options)
        self.jobDirectory = tempfile.mkdtemp(prefix="UKFGPU-", dir=slicer.app.temporaryPath)
        try:
            for node, name in ((dwi, "dwi"), (mask, "mask"), (seeds, "seeds")):
                if node and not slicer.util.exportNode(node, os.path.join(self.jobDirectory, name + ".nrrd")):
                    raise RuntimeError("Could not export " + name)
            self.output = str(output)
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
                Path(self.output).unlink(missing_ok=True)
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
                node = slicer.mrmlScene.AddNewNodeByClass(node_class, Path(self.output).stem)
                node.SetAndObservePolyData(reader.GetOutput())
                node.CreateDefaultDisplayNodes()
                node.SetAttribute("UKFGPU.LogFile", self.output + ".log")
                node.SetAttribute("UKFGPU.OutputFile", self.output)
                if slicer.app.layoutManager():
                    slicer.util.resetThreeDViews()
        except Exception as exc:
            error = str(exc)
        self.complete(node, error)

    def complete(self, node, error):
        Path(self.output + ".log").write_text(self.lastLog, encoding="utf-8")
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
