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

Logs and artifacts: `/data/yijie/ukf-GPU/validation/slicer-scene-2026-10-12/`.

Updated package: `/data/yijie/ukf-GPU/bin/slicer-extension-scene-5.2-2026-10-12/UKFGPU-Slicer5.2-linux-x86_64-2026-10-12.tar.gz`.

The GPU-reference regression checks above do not establish new CPU equivalence. Existing strict CPU/GPU acceptance failures remain in effect. Detailed historical reports are unchanged.
