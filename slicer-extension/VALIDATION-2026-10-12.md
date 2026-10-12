# Scene-first tracking and hover help (2026-10-12)

The extension no longer asks for an output file before tracking. Each job uses a unique temporary directory; the generated RAS polydata is attached to a scene node before the directory is removed. Result nodes are named by model and remain available to Slicer's Save workflow. Cancellation and failure also remove temporary job files. Logs remain in the module panel; no persistent VTK/log output is created automatically.

Hover help is provided on input, model, precision, device, tracking, and recording controls and their labels. Descriptions cover purpose, units where applicable, UI defaults, model applicability, and relevant compute/memory effects. The native backend and numerical algorithms are unchanged.

## Verification

Tested with `/data/software/Slicer-5.2.2-linux-amd64/Slicer` and the previously validated Linux runtime, using isolated test settings.

- Actual widget construction: no output-path field; input, tracking, and recording controls have tooltips.
- Simple2T mixed precision, Simple1T-FreeWater FP64, and NODDI1F FP64: export inputs, execute GPU tracking, import into the scene, check geometry against existing GPU references, and verify finite recorded arrays.
- Verify that input storage filenames are preserved and all temporary tracking files are removed after import.
- Save scene nodes to VTK after temporary-file removal, then reload with Slicer's reader; verify geometry and recorded-array count.
- Both the generic Model fallback and installed SlicerDMRI FiberBundle paths passed these checks.
- Actual GPU cancellation and invalid-executable failure: callback and temporary-directory cleanup passed.
- Install the new Slicer 5.2 archive using Slicer's Extensions Manager in an isolated directory, then exercise its packaged module and runtime.
- Three existing platform/runtime packaging tests passed; Windows native runtime execution remains unverified.

Earlier scene-first logs: `/data/yijie/ukf-GPU/validation/reports-2026-10-12/slicer-scene-2026-10-12/`.

Updated package: `/data/yijie/ukf-GPU/bin/UKFGPU-Slicer5.2-linux-x86_64-2026-10-12.tar.gz`.

The GPU-reference regression checks above do not establish new CPU equivalence. Existing strict CPU/GPU acceptance failures remain in effect. Detailed historical reports are unchanged.

## GPU selection, model controls, presets, and fiber naming

The current widget asynchronously queries CUDA in a separate Slicer Python process, using the backend's runtime library. Runtime ordinals and PCI bus IDs are used to obtain the corresponding driver names and total memory; CUDA libraries are not loaded into the Slicer GUI process. GPU visibility and ordering follow the same startup environment as tracking. Confirmed empty visibility disables Run; failed discovery exposes a manual CUDA index and an explanation. Refresh preserves a detected selection where possible.

Parameter and recording applicability is defined for all 12 models. Incompatible records are unchecked when switching models, unsupported numeric options are omitted from the command, and branching requires both a compatible multi-tensor model and a seed labelmap. Models without an FP32 signal path force FP64. Presets apply only when requested and preserve the selected model, inputs, GPU, precision, and fiber name. Zero filter-noise values use backend defaults, including the NODDI1F-specific fraction noise.

Fiber name is optional. An empty value uses `UKF_<model>`; names are stripped of surrounding whitespace and made unique in the scene. Unicode names and existing-name collisions are covered by the actual Slicer test.

Additional checks passed:

- All-model applicability and preset contracts (2 unit tests), plus existing platform/runtime tests (3).
- Native CUDA discovery with masks `0,1`, `1,0`, and `-1`: reversed PCI bus order and empty visibility.
- Actual Slicer widget: asynchronous discovery, no-device disabling, manual fallback, model switches clearing incompatible records, branching with/without seeds, explicit preset application, 3T sampling, and preservation of user names.
- Three-model scene/Save/reload and cancellation/failure checks with the updated source and packaged module.

Current reference data: `validation/reference-gpu/`. Current detailed logs: `validation/slicer-settings-2026-10-12/`. Windows GUI and CUDA execution remain unverified; the numerical backend is unchanged.
