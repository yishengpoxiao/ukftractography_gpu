import os
import json
import sys
from pathlib import Path
import shutil
import tempfile
import time
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

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
                       PRESETS, preset_values, model_options, TrackingProgress)


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
            "The offline installer bundles native dependencies for its included platforms; "
            "a compatible NVIDIA driver is required. "
            "GPU support is experimental and numerical validation is workload-specific. "
            "See the extension README for installation and the repository README for current limitations.")
        parent.acknowledgementText = "Based on pnlbwh/ukftractography."


class UKFGPUWidget(ScriptedLoadableModuleWidget):
    def setup(self):
        super().setup()
        self.logic = UKFGPULogic()
        self.logic.onLog = self.appendLog
        self.logic.onFinished = self.onFinished
        self.logic.onProgress = self.onProgress
        self.updatingParameters = False
        self.sceneRestoring = False
        self.cleanedUp = False
        self.startedAt = None
        self.pendingGpuPci = []
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
        self.precision.addItems(["FP64", "Mixed (guarded FP32)"])
        self.precision.setCurrentIndex(0)
        self.addRow(form, "Precision", self.precision, "Default: FP64. Mixed mode uses FP32 to select ranges in compatible signal exponentials, with FP64 fallback near rounding boundaries. Interpolation, exponential polynomials, states, covariance, and filter updates stay FP64. Unsupported host math policies retain FP64. DWI storage is FP32 in both modes.")
        self.allGpus = qt.QCheckBox("Show all accessible CUDA GPUs")
        self.allGpus.checked = True
        self.addRow(form, "GPU visibility", self.allGpus, "Enabled: ignore CUDA_VISIBLE_DEVICES in discovery and tracking child processes. Disabled: use Slicer's startup CUDA visibility and ordering. Container and driver access restrictions still apply. Only NVIDIA CUDA GPUs can run this backend.")
        self.allGpus.connect("toggled(bool)", self.refreshGpu)
        self.device = qt.QComboBox()
        self.device.setEditable(True)
        self.device.addItem("0")
        self.gpuDetected = False
        self.gpuDevices = {}
        self.gpuProbe = None
        self.gpuAvailable = True
        self.gpuTimer = qt.QTimer(self.parent)
        self.gpuTimer.setSingleShot(True)
        self.gpuTimer.connect("timeout()", self.gpuTimeout)
        gpuRow = qt.QHBoxLayout()
        gpuRow.addWidget(self.device)
        self.refreshGpuButton = qt.QPushButton("Refresh")
        self.refreshGpuButton.connect("clicked()", self.refreshGpu)
        self.refreshGpuButton.setToolTip("Detect GPUs using the selected visibility mode and preserve the selected physical GPU when available.")
        gpuRow.addWidget(self.refreshGpuButton)
        gpuHelp = "Choose the single GPU used for tracking. Each entry shows CUDA index, model, total memory and PCI bus address, so identical cards can be distinguished. If discovery fails, enter a CUDA index manually."
        self.addRow(form, "GPU device", gpuRow, gpuHelp)
        self.device.setToolTip(gpuHelp)
        self.recommendGpuButton = qt.QPushButton("Recommend GPU")
        self.recommendGpuButton.setToolTip("Select the detected GPU with the most free memory. Values are a snapshot; other applications may allocate memory later.")
        self.recommendGpuButton.connect("clicked()", self.recommendGpu)
        gpuRow.addWidget(self.recommendGpuButton)
        self.multiGpu = qt.QCheckBox("Use selected GPUs together")
        self.addRow(form, "Multiple GPUs", self.multiGpu, "Enable to distribute seeds across the checked GPUs below. Each device holds its own input copy; memory is not pooled.")
        self.gpuList = qt.QListWidget()
        self.gpuList.maximumHeight = 140
        self.gpuList.visible = False
        form.addRow(self.gpuList)
        self.multiGpu.connect("toggled(bool)", self.updateGpuMode)
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
            if name in ('stepLength', 'recordLength'):
                control.setMinimum(0.000001)
            if name in ('stoppingFA', 'stoppingThreshold', 'seedingThreshold'):
                control.setMaximum(1)
            if name.endswith('BranchingAngle'):
                control.setMaximum(90)
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
        self.progressBar = qt.QProgressBar()
        self.progressBar.setRange(0, 100)
        self.progressBar.setValue(0)
        self.layout.addWidget(self.progressBar)
        self.status = qt.QLabel("Ready")
        self.layout.addWidget(self.status)
        self.elapsed = qt.QLabel()
        self.layout.addWidget(self.elapsed)
        self.elapsedTimer = qt.QTimer(self.parent)
        self.elapsedTimer.connect('timeout()', self.updateElapsed)
        displayPanel = ctk.ctkCollapsibleButton()
        displayPanel.text = 'Fiber display'
        self.layout.addWidget(displayPanel)
        displayForm = qt.QFormLayout(displayPanel)
        self.result = self.selector(['vtkMRMLModelNode'], optional=True)
        self.addRow(displayForm, 'Fiber result', self.result, 'Select the result whose display you want to change.')
        self.showFibers = qt.QCheckBox('Visible')
        self.showFibers.checked = True
        self.addRow(displayForm, 'Visibility', self.showFibers, 'Show or hide the selected fibers.')
        self.opacity = qt.QDoubleSpinBox()
        self.opacity.setRange(0, 1)
        self.opacity.setSingleStep(0.05)
        self.opacity.value = 1
        self.addRow(displayForm, 'Opacity', self.opacity, '0 is transparent; 1 is opaque.')
        self.lineWidth = qt.QDoubleSpinBox()
        self.lineWidth.setRange(0.1, 10)
        self.lineWidth.value = 1
        self.addRow(displayForm, 'Line width', self.lineWidth, 'Width of line representations; tube radius is controlled by SlicerDMRI.')
        self.colorMode = qt.QComboBox()
        self.colorMode.addItems(['Solid', 'Direction', 'FA', 'Length'])
        self.addRow(displayForm, 'Color by', self.colorMode, 'Direction: absolute RAS tangent RGB. FA: recorded FA1 values (requires RecordFA). Length: whole-fiber length in mm.')
        self.fiberColor = ctk.ctkColorPickerButton()
        self.fiberColor.color = qt.QColor.fromRgbF(1, 0.8, 0.2)
        self.addRow(displayForm, 'Solid color', self.fiberColor, 'Color used in Solid mode.')
        for control, signal in ((self.result, 'currentNodeChanged(vtkMRMLNode*)'),
                                (self.showFibers, 'toggled(bool)'), (self.opacity, 'valueChanged(double)'),
                                (self.lineWidth, 'valueChanged(double)'), (self.colorMode, 'currentIndexChanged(int)'),
                                (self.fiberColor, 'colorChanged(QColor)')):
            control.connect(signal, self.applyDisplay)
        self.log = qt.QPlainTextEdit()
        self.log.readOnly = True
        self.log.setMaximumBlockCount(4000)
        self.layout.addWidget(self.log)
        self.layout.addStretch(1)
        self.model.connect("currentIndexChanged(int)", self.updateModelOptions)
        self.seeds.connect("currentNodeChanged(vtkMRMLNode*)", self.updateModelOptions)
        self.backend.connect("editingFinished()", self.refreshGpu)
        self.updateModelOptions()
        self.parameterNode = None
        self.parameterObserver = None
        self.sceneObservers = [slicer.mrmlScene.AddObserver(slicer.vtkMRMLScene.StartCloseEvent, self.onSceneClosing),
                               slicer.mrmlScene.AddObserver(slicer.vtkMRMLScene.EndCloseEvent, self.onSceneClosed),
                               slicer.mrmlScene.AddObserver(slicer.vtkMRMLScene.StartImportEvent, self.onSceneImporting),
                               slicer.mrmlScene.AddObserver(slicer.vtkMRMLScene.EndImportEvent, self.onSceneImported)]
        self.initializeParameters()
        for control, signal in [(self.model, 'currentIndexChanged(int)'), (self.precision, 'currentIndexChanged(int)'),
                                (self.allGpus, 'toggled(bool)'), (self.multiGpu, 'toggled(bool)'),
                                (self.device, 'currentIndexChanged(int)'), (self.gpuList, 'itemChanged(QListWidgetItem*)'),
                                (self.fiberName, 'textChanged(QString)'), (self.seedLabels, 'textChanged(QString)'),
                                (self.preset, 'currentIndexChanged(int)'), (self.backend, 'editingFinished()'),
                                *[(c, 'valueChanged(double)') for c in self.parameters.values()],
                                *[(c, 'toggled(bool)') for c in self.records.values()],
                                *[(c, 'currentNodeChanged(vtkMRMLNode*)') for c in (self.dwi, self.mask, self.seeds, self.result)],
                                (self.showFibers, 'toggled(bool)'), (self.opacity, 'valueChanged(double)'),
                                (self.lineWidth, 'valueChanged(double)'), (self.colorMode, 'currentIndexChanged(int)'),
                                (self.fiberColor, 'colorChanged(QColor)')]:
            control.connect(signal, self.saveParameters)
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

    def initializeParameters(self, *unused):
        if self.cleanedUp:
            return
        self.parameterNode = self.logic.getParameterNode()
        self.parameterObserver = self.parameterNode.AddObserver(vtk.vtkCommand.ModifiedEvent, self.restoreParameters)
        if self.parameterNode.GetParameter('Settings'):
            self.restoreParameters()
        else:
            self.sceneRestoring = False
            self.saveParameters()

    def onSceneClosing(self, *unused):
        self.sceneRestoring = True
        self.logic.cancel()
        if self.parameterNode and self.parameterObserver:
            self.parameterNode.RemoveObserver(self.parameterObserver)
        self.parameterNode = None
        self.parameterObserver = None

    def onSceneClosed(self, *unused):
        qt.QTimer.singleShot(0, self.initializeParameters)

    def onSceneImporting(self, *unused):
        self.sceneRestoring = True

    def onSceneImported(self, *unused):
        qt.QTimer.singleShot(0, self.restoreParameters)

    def saveParameters(self, *unused):
        if self.updatingParameters or self.sceneRestoring or slicer.mrmlScene.IsBatchProcessing() or self.parameterNode is None:
            return
        self.updatingParameters = True
        try:
            checked = self.checkedGpuPci()
            selected = self.gpuDevices.get(self.gpuDevice(), {}).get('pci_bus') if self.gpuDetected else None
            settings = dict(version=1, model=self.model.currentText, precision=self.precision.currentIndex,
                            backend=self.backend.text, fiberName=self.fiberName.text, seedLabels=self.seedLabels.text,
                            preset=self.preset.currentText, allGpus=self.allGpus.checked, multiGpu=self.multiGpu.checked,
                            gpuPci=checked if self.multiGpu.checked else ([selected] if selected else self.pendingGpuPci),
                            parameters={k: c.value for k, c in self.parameters.items()},
                            records={k: c.checked for k, c in self.records.items()},
                            visible=self.showFibers.checked, opacity=self.opacity.value, lineWidth=self.lineWidth.value,
                            colorMode=self.colorMode.currentText,
                            color=[self.fiberColor.color.redF(), self.fiberColor.color.greenF(), self.fiberColor.color.blueF()])
            token = self.parameterNode.StartModify()
            try:
                self.parameterNode.SetParameter('Settings', json.dumps(settings))
                for name in ('dwi', 'mask', 'seeds', 'result'):
                    node = getattr(self, name).currentNode()
                    self.parameterNode.SetNodeReferenceID(name, node.GetID() if node else None)
            finally:
                self.parameterNode.EndModify(token)
        finally:
            self.updatingParameters = False

    def restoreParameters(self, *unused):
        if self.cleanedUp or self.updatingParameters or slicer.mrmlScene.IsBatchProcessing() or self.parameterNode is None or not self.parameterNode.GetParameter('Settings'):
            return
        settings = json.loads(self.parameterNode.GetParameter('Settings'))
        if settings['version'] != 1:
            raise ValueError('Unsupported UKFGPU scene settings format.')
        visibility_changed = self.allGpus.checked != settings['allGpus']
        self.updatingParameters = True
        try:
            for name in ('dwi', 'mask', 'seeds', 'result'):
                getattr(self, name).setCurrentNode(self.parameterNode.GetNodeReference(name))
            self.model.currentIndex = list(MODELS).index(settings['model'])
            self.precision.currentIndex = settings['precision']
            if os.path.isfile(settings['backend']):
                self.backend.text = settings['backend']
            self.fiberName.text = settings['fiberName']
            self.seedLabels.text = settings['seedLabels']
            self.preset.currentIndex = list(PRESETS).index(settings['preset'])
            self.allGpus.checked = settings['allGpus']
            self.multiGpu.checked = settings['multiGpu']
            self.pendingGpuPci = settings['gpuPci']
            for name, value in settings['parameters'].items():
                self.parameters[name].value = value
            for name, value in settings['records'].items():
                self.records[name].checked = value
            self.showFibers.checked = settings['visible']
            self.opacity.value = settings['opacity']
            self.lineWidth.value = settings['lineWidth']
            self.colorMode.currentIndex = self.colorMode.findText(settings['colorMode'])
            self.fiberColor.color = qt.QColor.fromRgbF(*settings['color'])
            self.updateModelOptions()
            self.updateGpuMode()
        finally:
            self.updatingParameters = False
            self.sceneRestoring = False
        self.applyDisplay()
        if self.gpuProbe is None and (visibility_changed or self.pendingGpuPci):
            self.refreshGpu()

    def checkedGpuPci(self):
        return [self.gpuList.item(i).data(qt.Qt.UserRole) for i in range(self.gpuList.count)
                if self.gpuList.item(i).checkState() == qt.Qt.Checked]

    def updateGpuMode(self, *unused):
        self.gpuList.visible = self.multiGpu.checked
        self.device.enabled = self.gpuAvailable and not self.multiGpu.checked and self.gpuProbe is None and self.logic.process is None
        if self.multiGpu.checked and self.gpuDetected and not self.checkedGpuPci():
            selected = self.gpuDevices[self.gpuDevice()]['pci_bus']
            for i in range(self.gpuList.count):
                if self.gpuList.item(i).data(qt.Qt.UserRole) == selected:
                    self.gpuList.item(i).setCheckState(qt.Qt.Checked)

    def recommendGpu(self):
        available = [d for d in self.gpuDevices.values() if d.get('free_memory') is not None]
        if available:
            best = max(available, key=lambda d: d['free_memory'])
            self.device.currentIndex = self.device.findData(best['index'])
            for i in range(self.gpuList.count):
                item = self.gpuList.item(i)
                item.setCheckState(qt.Qt.Checked if item.data(qt.Qt.UserRole) == best['pci_bus'] else qt.Qt.Unchecked)

    def updateElapsed(self):
        if self.startedAt is not None:
            self.elapsed.text = 'Elapsed: %.1f s' % (time.monotonic() - self.startedAt)

    def onProgress(self, progress):
        tracking = progress['stage'] == 'GPU tracking' and progress['total'] > 0
        self.progressBar.setRange(0, progress['total'] if tracking else 0)
        if tracking:
            self.progressBar.value = progress['completed']
            self.progressBar.format = '%v / %m half-tracks completed'
        self.status.text = progress['stage'] + (' (pass %d)' % progress['pass_number'] if progress.get('pass_number', 0) > 1 else '')

    def applyDisplay(self, *unused):
        if self.updatingParameters:
            return
        node = self.result.currentNode()
        if node is not None:
            try:
                color = self.fiberColor.color
                self.logic.updateDisplay(node, self.showFibers.checked, self.opacity.value, self.lineWidth.value,
                                         self.colorMode.currentText, (color.redF(), color.greenF(), color.blueF()))
            except ValueError as exc:
                self.status.text = str(exc)

    def gpuDevice(self):
        if self.gpuDetected:
            return int(self.device.itemData(self.device.currentIndex))
        device = int(self.device.currentText)
        if device < 0:
            raise ValueError("GPU device index must be nonnegative.")
        return device

    def refreshGpu(self, *unused):
        if self.updatingParameters or self.gpuProbe is not None or self.logic.process is not None:
            return
        self.gpuStatus.text = "Detecting CUDA GPUs..."
        self.refreshGpuButton.enabled = False
        self.allGpus.enabled = False
        self.device.enabled = False
        self.multiGpu.enabled = False
        self.gpuList.enabled = False
        self.recommendGpuButton.enabled = False
        self.runButton.enabled = False
        self.gpuProbe = qt.QProcess(self.parent)
        environment = qt.QProcessEnvironment()
        for key, value in process_environment(slicer.util.startupEnvironment(), all_gpus=self.allGpus.checked).items():
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
        if self.gpuDetected:
            selected = self.device.itemData(self.device.currentIndex)
        else:
            try:
                selected = int(self.device.currentText)
            except ValueError:
                selected = 0
        selected_pci = self.gpuDevices.get(selected, {}).get('pci_bus')
        checked = self.pendingGpuPci or self.checkedGpuPci() or ([selected_pci] if selected_pci else [])
        if self.pendingGpuPci:
            selected_pci = self.pendingGpuPci[0]
        self.pendingGpuPci = []
        self.updatingParameters = True
        self.device.clear()
        self.gpuList.clear()
        self.gpuDevices = {device['index']: device for device in devices or []}
        self.gpuDetected = devices is not None and bool(devices)
        self.gpuAvailable = devices is None or bool(devices)
        self.device.setEditable(devices is None)
        if devices is None:
            self.device.addItem("0")
            self.gpuStatus.text = message + " Manual CUDA index is available."
        elif not devices:
            self.gpuStatus.text = "No CUDA GPU is visible. Check the driver and GPU visibility mode, then Refresh."
        else:
            for device in devices:
                free = '%.1f GiB free' % (device['free_memory']/2**30) if device.get('free_memory') is not None else 'free memory unavailable'
                title = "CUDA %d: %s (%s / %.1f GiB) | PCI %s" % (device['index'], device['name'], free, device['memory']/2**30, device['pci_bus'])
                self.device.addItem(title, device['index'])
                item = qt.QListWidgetItem(title)
                item.setData(qt.Qt.UserRole, device['pci_bus'])
                item.setFlags(item.flags() | qt.Qt.ItemIsUserCheckable)
                item.setCheckState(qt.Qt.Checked if device['pci_bus'] in checked else qt.Qt.Unchecked)
                self.gpuList.addItem(item)
            index = next((i for i, device in enumerate(devices) if device['pci_bus'] == selected_pci), self.device.findData(selected))
            self.device.currentIndex = max(0, index)
            scope = "all accessible GPUs" if self.allGpus.checked else "startup CUDA visibility"
            self.gpuStatus.text = "%d CUDA GPU(s) detected (%s). Select a GPU above." % (len(devices), scope)
        self.device.enabled = self.gpuAvailable
        self.refreshGpuButton.enabled = True
        self.allGpus.enabled = True
        self.multiGpu.enabled = self.gpuAvailable
        self.gpuList.enabled = self.gpuAvailable
        self.recommendGpuButton.enabled = any(d.get('free_memory') is not None for d in devices or [])
        self.updateGpuMode()
        self.updatingParameters = False
        self.saveParameters()
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
            selected = [d for d in self.gpuDevices.values() if d['pci_bus'] in self.checkedGpuPci()] if self.multiGpu.checked else []
            if self.multiGpu.checked and not selected:
                raise ValueError('Check at least one GPU for multi-GPU tracking.')
            self.startedAt = time.monotonic()
            self.elapsedTimer.start(1000)
            self.runButton.enabled = False
            self.onProgress(dict(stage='Preparing inputs', completed=0, total=0))
            self.logic.start(self.backend.text, self.dwi.currentNode(), self.mask.currentNode(),
                             self.model.currentText, self.precision.currentIndex,
                             -1 if self.multiGpu.checked else self.gpuDevice(), self.seeds.currentNode(), self.seedLabels.text, options,
                             result_name=self.fiberName.text, all_gpus=self.allGpus.checked,
                             gpu_uuids=[d['uuid'] for d in selected] if selected else None)
            self.appendLog("Selected GPU(s): " + (', '.join(d['pci_bus'] for d in selected) if selected else self.device.currentText) + "\n")
            self.runButton.enabled = False
            self.refreshGpuButton.enabled = False
            self.device.enabled = False
            self.allGpus.enabled = False
            self.multiGpu.enabled = False
            self.gpuList.enabled = False
            self.recommendGpuButton.enabled = False
            self.applyPresetButton.enabled = False
            self.cancelButton.enabled = True
            self.status.text = "Tracking on " + self.device.currentText
        except Exception as exc:
            self.elapsedTimer.stop()
            self.runButton.enabled = self.gpuAvailable and self.gpuProbe is None
            self.progressBar.setRange(0, 100)
            self.progressBar.value = 0
            self.status.text = str(exc)
            slicer.util.errorDisplay(str(exc))

    def onFinished(self, node, error):
        self.runButton.enabled = self.gpuAvailable and self.gpuProbe is None
        self.refreshGpuButton.enabled = True
        self.device.enabled = self.gpuAvailable
        self.allGpus.enabled = True
        self.multiGpu.enabled = self.gpuAvailable
        self.gpuList.enabled = self.gpuAvailable
        self.recommendGpuButton.enabled = any(d.get('free_memory') is not None for d in self.gpuDevices.values())
        self.updateGpuMode()
        self.applyPresetButton.enabled = True
        self.cancelButton.enabled = False
        self.status.text = error or "Added to scene: " + node.GetName() + ". Use Save to keep fibers."
        self.elapsedTimer.stop()
        self.updateElapsed()
        self.progressBar.setRange(0, 100)
        self.progressBar.value = 0 if error else 100
        self.progressBar.format = 'Cancelled' if error == 'Cancelled' else ('Failed' if error else 'Completed')
        if node:
            self.result.setCurrentNode(node)
            self.applyDisplay()
            self.saveParameters()
        if error and error != "Cancelled":
            slicer.util.errorDisplay(error)

    def cleanup(self):
        self.cleanedUp = True
        self.elapsedTimer.stop()
        if self.parameterNode and self.parameterObserver:
            self.parameterNode.RemoveObserver(self.parameterObserver)
        for observer in self.sceneObservers:
            slicer.mrmlScene.RemoveObserver(observer)
        self.gpuTimer.stop()
        if self.gpuProbe is not None:
            self.gpuProbe.kill()
            self.gpuProbe.deleteLater()
            self.gpuProbe = None
        self.logic.onLog = None
        self.logic.onFinished = None
        self.logic.onProgress = None
        self.logic.cancel()


class UKFGPULogic(ScriptedLoadableModuleLogic):
    def __init__(self):
        super().__init__()
        self.process = None
        self.jobDirectory = None
        self.onLog = None
        self.onFinished = None
        self.onProgress = None
        self.lastLog = ""
        self.cancelled = False

    @staticmethod
    def validateInputs(dwi, mask, seeds, labels):
        image = dwi.GetImageData()
        if image is None or image.GetNumberOfPoints() == 0:
            raise ValueError('DWI has no image data.')
        if image.GetNumberOfScalarComponents() != dwi.GetNumberOfGradients():
            raise ValueError('DWI component count does not match gradient metadata.')
        gradients = vtk_to_numpy(dwi.GetDiffusionGradients())
        bvalues = vtk_to_numpy(dwi.GetBValues())
        if not np.isfinite(gradients).all() or not np.isfinite(bvalues).all() or np.any(bvalues < 0) or not np.any(bvalues > 0):
            raise ValueError('DWI gradients/b-values must be finite, with nonnegative b-values and at least one diffusion-weighted measurement.')
        reference = vtk.vtkMatrix4x4()
        dwi.GetIJKToRASMatrix(reference)
        for name, node in (('Brain mask', mask), ('Seed labelmap', seeds)):
            if node is None:
                continue
            data = node.GetImageData()
            if data is None or data.GetDimensions() != image.GetDimensions() or data.GetNumberOfScalarComponents() != 1:
                raise ValueError(name + ' must be a scalar volume with the same dimensions as the DWI.')
            matrix = vtk.vtkMatrix4x4()
            node.GetIJKToRASMatrix(matrix)
            if any(abs(matrix.GetElement(r, c)-reference.GetElement(r, c)) > 1e-6 for r in range(4) for c in range(4)):
                raise ValueError(name + ' spacing, origin or directions differ from the DWI. Resample the labelmap to the DWI geometry first.')
            values = slicer.util.arrayFromVolume(node)
            if not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError(name + ' must contain finite, nonnegative values.')
            if name == 'Seed labelmap' and np.any(values != np.floor(values)):
                raise ValueError('Seed labels must be integers.')
        mask_values = slicer.util.arrayFromVolume(mask)
        if not np.any(mask_values > 0):
            raise ValueError('Brain mask is empty.')
        if seeds is not None:
            selected = [int(value.strip()) for value in labels.split(',')]
            if any(value < 0 for value in selected):
                raise ValueError('Seed labels must be nonnegative integers.')
            if not np.any(np.isin(slicer.util.arrayFromVolume(seeds), selected) & (mask_values > 0)):
                raise ValueError('The selected seed labels contain no voxels inside the brain mask.')

    @staticmethod
    def updateDisplay(node, visible, opacity, width, mode, color):
        poly = node.GetPolyData()
        if poly is None or poly.GetPoints() is None:
            raise ValueError('Selected result has no fiber geometry.')
        array_name = {'Direction': 'UKFGPU.Direction', 'Length': 'UKFGPU.Length', 'FA': 'FA1'}.get(mode)
        if mode == 'FA' and poly.GetPointData().GetArray(array_name) is None:
            raise ValueError('FA coloring requires FA1 data. Enable recordFA when tracking a tensor model.')
        if mode in ('Direction', 'Length') and poly.GetPointData().GetArray(array_name) is None:
            points = vtk_to_numpy(poly.GetPoints().GetData()).astype(float)
            cells = vtk_to_numpy(poly.GetLines().GetData())
            values = np.zeros((len(points), 3), dtype=np.uint8) if mode == 'Direction' else np.zeros(len(points))
            offset = 0
            while offset < len(cells):
                count = int(cells[offset])
                ids = cells[offset+1:offset+count+1]
                if count > 1:
                    differences = np.diff(points[ids], axis=0)
                    if mode == 'Length':
                        values[ids] = np.linalg.norm(differences, axis=1).sum()
                    else:
                        tangents = np.vstack((differences, differences[-1]))
                        lengths = np.linalg.norm(tangents, axis=1, keepdims=True)
                        normalized = np.divide(tangents, lengths, out=np.zeros_like(tangents), where=lengths > 0)
                        values[ids] = np.rint(np.abs(normalized)*255).astype(np.uint8)
                offset += count + 1
            array = numpy_to_vtk(values, deep=True)
            array.SetName(array_name)
            poly.GetPointData().AddArray(array)
        active = [i for i in range(node.GetNumberOfDisplayNodes()) if node.GetNthDisplayNode(i).GetVisibility()]
        if active:
            node.SetAttribute('UKFGPU.VisibleDisplays', json.dumps(active))
        else:
            active = json.loads(node.GetAttribute('UKFGPU.VisibleDisplays') or '[0]')
        for index in range(node.GetNumberOfDisplayNodes()):
            display = node.GetNthDisplayNode(index)
            display.SetVisibility(visible and index in active)
            display.SetOpacity(opacity)
            display.SetLineWidth(width)
            if hasattr(display, 'SetColorModeToSolid'):
                if mode == 'Solid':
                    display.SetColorModeToSolid()
                else:
                    display.SetColorModeToScalarData()
            display.SetScalarVisibility(mode != 'Solid')
            display.SetColor(*color)
            if array_name:
                display.SetActiveScalarName(array_name)
                if mode == 'Direction':
                    display.SetScalarRangeFlag(slicer.vtkMRMLDisplayNode.UseDirectMapping)
                else:
                    display.SetScalarRangeFlag(slicer.vtkMRMLDisplayNode.UseDataScalarRange)
                    display.SetAndObserveColorNodeID('vtkMRMLColorTableNodeRainbow')
        node.SetAttribute('UKFGPU.ColorMode', mode)

    def start(self, executable, dwi, mask, model="Simple2T", precision=0,
              device=0, seeds=None, seed_labels="1", options=None, result_name="", all_gpus=False, gpu_uuids=None):
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
        self.validateInputs(dwi, mask, seeds, seed_labels)
        if gpu_uuids and (len(set(gpu_uuids)) != len(gpu_uuids) or any(not value.startswith('GPU-') for value in gpu_uuids)):
            raise ValueError('Select distinct detected CUDA GPUs.')
        # Build argv before exporting so invalid model/options leave no temporary files.
        args = command(executable, "dwi.nrrd", "mask.nrrd", "fibers.vtk", model,
                       precision, device, "seeds.nrrd" if seeds else None, seed_labels, options)
        self.jobDirectory = tempfile.mkdtemp(prefix="UKFGPU-", dir=slicer.app.temporaryPath)
        self.output = str(Path(self.jobDirectory) / "fibers.vtk")
        self.model = model
        self.resultName = result_name.strip() or "UKF_" + model
        self.precision = precision
        self.device = device
        self.gpuVisibility = "All accessible CUDA GPUs" if all_gpus else "Startup CUDA visibility"
        self.gpuUuids = gpu_uuids or []
        self.progress = TrackingProgress()
        try:
            for node, name in ((dwi, "dwi"), (mask, "mask"), (seeds, "seeds")):
                if node and not slicer.util.exportNode(node, os.path.join(self.jobDirectory, name + ".nrrd")):
                    raise RuntimeError("Could not export " + name)
            self.lastLog = ""
            self.cancelled = False
            self.process = qt.QProcess(slicer.app)
            environment = qt.QProcessEnvironment()
            for key, value in process_environment(slicer.util.startupEnvironment(), executable, all_gpus=all_gpus,
                                                  gpu_uuids=gpu_uuids, progress=True).items():
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
        if self.onProgress:
            self.onProgress(self.progress.feed(text))

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
                node.SetAttribute("UKFGPU.Precision", "FP64" if self.precision == 0 else "Mixed (guarded FP32)")
                node.SetAttribute("UKFGPU.GPUDevice", str(self.device))
                node.SetAttribute("UKFGPU.GPUVisibility", self.gpuVisibility)
                node.SetAttribute("UKFGPU.GPUUUIDs", json.dumps(self.gpuUuids))
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
