# Offline installer for 3D Slicer

The portable installer detects the running Slicer version and the operating system. It generates the corresponding `lib/Slicer-<major.minor>/qt-scripted-modules` layout and installs through Slicer's Extensions Manager. It requires Slicer 5.2 or newer and imposes no major-version upper bound. Automatic layout selection does not guarantee compatibility with every future Slicer API.

## Install

1. Extract the entire `UKFGPU-Installer-*.zip` to a writable directory. This ZIP adapts at installation time; do not select it directly in Extensions Manager.
2. Start your Slicer, open its Python console, and run:

   ```python
   import runpy
   runpy.run_path("/full/path/UKFGPU-Installer/install.py", run_name="__main__")
   ```

   On Windows, use a raw path such as `r"C:\Users\you\Downloads\UKFGPU-Installer\install.py"`.

   Alternatively, launch your Slicer executable with `--python-script /full/path/UKFGPU-Installer/install.py`.
3. Restart Slicer and open **Diffusion > Tractography > UKF GPU Tractography**. The GPU dropdown lists accessible NVIDIA CUDA cards with model, memory and PCI address; select the card to use. **Show all accessible CUDA GPUs** is enabled by default. Uncheck it to inherit Slicer's startup `CUDA_VISIBLE_DEVICES` mask.

If UKFGPU is already installed, the installer creates a new matching `.tar.gz` in its `packages` directory and leaves the existing installation intact. Remove the old UKFGPU in Extensions Manager, restart Slicer, then use **Install from file** to install that generated archive. Do not overwrite DLLs belonging to a running Windows process.

The installer verifies file checksums, checks the platform and Linux glibc requirements, tests native library loading, and queries CUDA in a child process. Checksums detect incomplete extraction or corruption; they are not a digital signature. Use packages from a trusted source. No package manager or administrator privileges are needed for the bundled runtime. Slicer manages extension registration; the installer does not directly rewrite Slicer configuration files.

## Dependencies and limits

| Component | Installation behavior |
| --- | --- |
| Python, Qt, NumPy and VTK used by the UI | Provided by Slicer; no pip or Conda environment is installed |
| ITK, VTK and Teem used by the backend | Included in the compiled backend or packaged native dependency closure |
| CUDA runtime | Installed privately with the extension; no CUDA Toolkit installation needed |
| Linux C++ / GCC runtime | Collected automatically when building the installer and installed privately |
| Windows MSVC runtime | Required at packaging time and installed privately beside the executable |
| SlicerDMRI | Optional; fibers display as Model nodes if FiberBundle support is absent |
| NVIDIA graphics driver | System prerequisite; cannot be installed privately as an extension dependency |
| Linux glibc and dynamic loader | System prerequisites; checked against the actual packaged binaries |

The installer supports native **Linux x86_64** and **Windows x86_64** runtime selection. A bundle contains only the platforms listed in its dated manifest. If a native Windows runtime is absent, Windows installation stops with an explicit message. A source ZIP does not supply a native Windows executable. See the latest release and validation report for available native runtimes. Windows GPU execution requires a separate hardware acceptance run. macOS and ARM CUDA backends are unavailable.

Build scripts select the GPU architectures supported by the chosen CUDA compiler and include baseline PTX. There is no fixed Toolkit-version check. A distributed runtime still needs a compatible NVIDIA driver; its bundled CUDA runtime must match the native backend. Older drivers may require a build with an older compatible Toolkit. Only RTX 4090 execution is currently validated. PTX JIT requires a driver that understands the toolkit's PTX version; a CUDA-family minimum driver alone does not guarantee JIT support. CPU hardware must support the AVX instructions used by the current Linux backend. Driver or hardware problems are reported and do not trigger system driver changes.

See [NVIDIA CUDA compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/) and [Slicer's installation API](https://slicer.readthedocs.io/en/latest/developer_guide/script_repository/gui.html).

## Build a portable installer

First stage a native `Runtime` directory using `scripts/package_slicer_extension.py`. Then:

```bash
python scripts/package_portable_installer.py \
  --runtime linux-x86_64=/path/to/linux/Runtime \
  --output /new/UKFGPU-Installer-linux-x86_64.zip
```

To include both operating systems, add `--runtime windows-x86_64=/path/to/windows/Runtime` with real Windows x64 binaries and DLLs. See [the Windows build guide](WINDOWS-2026-10-11.md). Packaging Linux runtimes requires `ldd` and `readelf`; Windows runtimes are checked for PE architecture and required CUDA/MSVC DLLs. Assemble combined bundles on Linux.

Output paths must be new; existing packages and manifests are not overwritten. The numerical GPU accuracy limitations remain unchanged.

See the [installer validation report](INSTALLER-VALIDATION-2026-10-12.md) for executed tests and unverified platforms.
