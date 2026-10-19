# UKF GPU: 3D Slicer Extension

This extension connects the CUDA UKF backend to Slicer. It provides 12 experimental models, FP64/mixed precision, GPU selection, tracking and recording options, asynchronous execution, cancellation, logs, and fiber display. It also validates input geometry and seed selection, reports actual tracking counts/elapsed time, saves settings with the scene, and provides direction/FA/length/solid coloring.

Linux integration has been tested with Slicer 5.2.2, 5.6.2 and 5.12.4. Windows native compilation, CPU execution, and actual installation/module loading have been checked in CI with Slicer 5.6.2 and 5.12.4. Windows GPU tracking still needs verification on NVIDIA hardware.

## Install

Use the latest **version-adaptive offline installer ZIP** containing a native runtime for your operating system:

- **Windows:** open the [Windows native installer workflow](https://github.com/yishengpoxiao/ukftractography_gpu/actions/workflows/windows-installer-2026-10-14-r6.yml), select the latest successful run, and download its `UKFGPU-Windows-native-*` artifact. Extract the artifact download to find `UKFGPU-Installer-windows-x86_64.zip`.
- **Linux:** use an installer ZIP generated from the current native Linux build; see [Build installers](#build-installers). Linux binaries are not currently distributed through GitHub Actions.

GitHub Releases and the online Slicer Extensions Catalog are not used for distribution. Downloading the repository source ZIP does not provide a compiled backend. A bundle supports only the platforms listed in its `manifest-*.json`.

1. Fully extract the installer ZIP into a writable directory. Locate the extracted `UKFGPU-Installer` directory containing `install.py`. This ZIP is installed through that script inside Slicer.
2. Open Slicer's Python console and run:

   ```python
   import runpy
   runpy.run_path("/full/path/UKFGPU-Installer/install.py", run_name="__main__")
   ```

   Replace the path with the actual extracted location. On Windows, use a raw path such as `r"C:\Users\you\Downloads\UKFGPU-Installer\install.py"`.
3. Restart Slicer and open **Diffusion > Tractography > UKF GPU Tractography**.

The installer verifies checksums, selects the included native runtime, checks native loading and CUDA discovery, and registers the module through Slicer's Extensions Manager. It generates an archive matching the running Slicer version and revision. Checksums detect corruption; obtain the bundle from a trusted source.

The installer supports Slicer 5.2 or newer and the platforms included in its dated manifest. Native CUDA and C++ runtime libraries are installed privately; no separate CUDA Toolkit, Visual Studio, Conda, or pip environment is needed by end users. Slicer supplies Python, Qt, NumPy, and UI VTK; SlicerDMRI is optional. The NVIDIA driver, Linux glibc, and dynamic loader remain system prerequisites. Linux binaries require x86_64 with AVX.

GPU targets are selected from the build compiler's supported architectures, with baseline PTX included. A compatible NVIDIA driver is required; PTX JIT additionally requires support for the Toolkit's PTX version. Older drivers may need a build with an older compatible Toolkit. RTX 4090 execution is validated; other GPUs and Windows GPU execution remain unverified. macOS and ARM CUDA backends are unavailable.

### Upgrade an existing installation

Run the new bundle's `install.py` as above. If UKFGPU is already installed, the installer preserves it and reports the path of a newly generated `.tar.gz` archive in the bundle's `packages` directory. Remove UKFGPU in **Extensions Manager**, restart Slicer, select **Install from file...** and that generated archive, then restart again. Use the archive generated for the Slicer installation you are upgrading.

## Use

The [parameter reference and examples](https://github.com/yishengpoxiao/ukftractography_gpu/blob/main/USAGE.md) explain CLI flags, model-dependent defaults, and the HCP benchmark settings.

1. Load a DWI NRRD with DWMRI gradients and b-value metadata.
2. Select the brain mask and optional seed labelmap. Export segmentation nodes to labelmaps first; enter labels separated by commas.
3. Select the model and precision. The GPU list automatically shows all accessible NVIDIA CUDA GPUs, with index, model, free/total memory and PCI bus address. Choose a single GPU, or enable **Use multiple selected GPUs** and check the cards to use together. **Recommend GPU** selects the detected card with the most free memory; refresh to update this snapshot. **Show all accessible CUDA GPUs** is enabled by default: it clears `CUDA_VISIBLE_DEVICES` only in discovery and tracking child processes. Uncheck it to inherit Slicer's startup visibility and ordering. Container and driver access restrictions remain in effect. Click **Refresh** after changing the backend; the selected physical GPU is preserved by PCI address when available, even if its CUDA index changes. GPU controls are locked during discovery/tracking. Detection runs in a separate process; if it fails, enter a CUDA index manually. Confirmed empty discovery disables tracking.
4. Set **Fiber name** if desired; leave it empty to use `UKF_<model>`. Duplicate names are automatically distinguished by Slicer. Choose a **Tracking preset** and click **Apply**, or adjust **Tracking and recording**, then click **Run GPU tracking**. No output path is required.
5. Fibers are added directly to the current scene and displayed on completion. Use Slicer's **Save** dialog to save the fibers or scene when needed. Temporary tracking inputs and output are removed after import; unsaved fibers are lost when the scene is discarded. Cancellation or failure also removes the job's temporary files. Logs remain in the module's log panel for inspection and copying.

Hover over parameter labels or controls to see their purpose, units, defaults, model applicability, and relevant runtime/memory implications. Recording checkboxes also explain the arrays they add.

Available presets are **Standard**, **Quick preview**, and **Fine sampling**. Standard uses model-specific filter noise defaults and standard sampling (0.3/0.9 mm step/record spacing for 1T/2T, 0.15/0.45 mm for 3T). Quick preview limits each half-fiber to 50 mm and uses 1.8 mm recording spacing. Fine sampling halves the standard step and record spacing; it costs more computation and does not guarantee better anatomical accuracy. Apply replaces tracking parameters and clears all recording options, while preserving model, precision, device, inputs and fiber name. Selecting a preset alone does not overwrite manual settings.

Controls are enabled by model: tensor FA/eigenvalue parameters and tensor recording, FreeWater noise/recording, and NODDI noise/recording. Branching additionally requires a multi-tensor model and a seed labelmap. Switching models clears incompatible record flags and disables unsupported precision selection; Simple3T, Full Tensor, FreeWater and NODDI use FP64. Disabled numeric options are omitted from the backend command.

Models include Simple1T/2T/3T, Full1T/2T/3T, Simple1T/2T-FreeWater, Full1T/2T-FreeWater, and NODDI1F/2F. Mixed precision lowers Simple1T/2T signal calculations to FP32; sensitive calculations and other models retain FP64. Zero Qm/Ql/Qw/Rs values use backend model defaults. Branching is disabled by default.

FP64 is the default precision. Mixed precision can exceed the 0.001 mm position tolerance on real data and can change recorded states; select it explicitly for exploratory workloads and compare against FP64 before using its results quantitatively. Scene settings preserve previously selected precision values.

With SlicerDMRI's FiberBundle node class available, fibers are imported as a FiberBundle; otherwise, a Model node is used. Automated integration covers both Model and SlicerDMRI FiberBundle display, Save and reload in Slicer 5.2.2. UKF VTK coordinates are RAS and are read directly to avoid a generic reader's LPS conversion. Input export preserves original node storage paths. Inputs with parent transforms are unsupported; use aligned volumes without parent transforms.

Unavailable GPUs or a backend CPU fallback are reported as failures. Inspect the retained log for driver, runtime, parameter, or algorithm errors.

For precision acceptance and upstream numerical limitations, see the [repository status](https://github.com/yishengpoxiao/ukftractography_gpu#features-and-current-status).

## Build installers

These steps are for building packages from source. For installation, follow [Install](#install).

**Windows:** the [native build script](https://github.com/yishengpoxiao/ukftractography_gpu#build) builds dependencies, stages the runtime, and produces `UKFGPU-Installer-windows-x86_64.zip`. The Windows workflow runs this same build and installation checks.

**Linux:** build the native backend using the [repository build instructions](https://github.com/yishengpoxiao/ukftractography_gpu#build), then stage its runtime. From the repository root, use a new output directory:

```bash
python scripts/package_slicer_extension.py \
  --binary /path/to/UKFTractography_gpu \
  --gpu-library /path/to/libukf_gpu.so \
  --cuda-runtime /path/to/libcudart.so.12 \
  --slicer-version 5.12 \
  --output /new/package-directory
```

The Linux packager copies and strips binaries, removes development RPATHs, and creates an isolated launcher. If the CUDA runtime directory contains a LICENSE file, it is included. The staged runtime is `/new/package-directory/UKFGPU/UKFGPU/Runtime`.

Create the version-adaptive installer from that runtime:

```bash
python scripts/package_portable_installer.py \
  --runtime linux-x86_64=/new/package-directory/UKFGPU/UKFGPU/Runtime \
  --output /new/UKFGPU-Installer-linux-x86_64.zip
```

The portable packager collects Linux shared-library dependencies and records the required glibc version. To assemble a combined bundle on Linux, add `--runtime windows-x86_64=/path/to/native/windows/Runtime` from a matching native Windows build. Include only platforms you have built. Output paths must be new; existing packages and manifests are not overwritten.

## Development checks

```bash
python -m unittest discover -s slicer-extension/UKFGPU/Testing -p 'test_*.py'
```

These checks cover model settings, tracking controls, platform runtime selection, and installer/packager contracts without requiring Slicer or a GPU. To install and reload a compiled bundle in a real Slicer application:

```bash
python scripts/verify_slicer_installation.py \
  --slicer /path/to/Slicer \
  --bundle /path/to/UKFGPU-Installer.zip \
  --output /new/installation-check
```

On Windows, `scripts/test_windows_slicer.ps1 -Bundle C:\UKF\UKFGPU-Installer.zip -WorkDirectory C:\UKF\slicer-check -Versions 5.6.2,latest` downloads Slicer and performs installation/loading checks. These checks require a new work directory and do not establish GPU tracking accuracy. Numerical acceptance scripts and reports are maintained separately from the source repository.

This is a modified UKFTractography integration. Original software attribution and redistribution terms are in [LICENSE.txt](https://github.com/yishengpoxiao/ukftractography_gpu/blob/main/LICENSE.txt).
