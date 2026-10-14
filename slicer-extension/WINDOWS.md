# Windows build and packaging

Native Windows x86_64 installers are available in the [preview release](https://github.com/yishengpoxiao/ukftractography_gpu/releases/tag/v0.2.0-preview.20261014). End users need a compatible NVIDIA driver; the installer bundles native dependencies privately, without requiring Visual Studio or a separate CUDA Toolkit. Follow the [installer guide](INSTALLER.md) to install the version-adaptive ZIP in Slicer 5.2 or newer.

Windows native compilation, CPU execution, and extension installation/module loading have been checked in CI with Slicer 5.6.2 and 5.12.4. GPU tracking accuracy and performance still require validation on Windows NVIDIA hardware.

## Build prerequisites

- Visual Studio 2022 or Build Tools with Desktop development with C++, MSVC x64 tools, and the Windows SDK.
- A CUDA Toolkit supporting C++17, the selected host compiler, and the target GPU architectures. No fixed Toolkit version is required.
- CMake 3.22+, Ninja, Git, and Python 3.9+ on PATH.

Use a short checkout/build path, such as `C:\UKF`, and open **x64 Developer PowerShell for VS 2022**. The driver's CUDA/UMD version does not identify an installed Toolkit; use `nvcc --version` to inspect the compiler.

## Build installers

From the repository root:

```powershell
.\scripts\build_windows_gpu.ps1 `
  -BuildDirectory C:\UKF\build `
  -PackageDirectory C:\UKF\package-5.6 `
  -SlicerVersion 5.6 `
  -CudaPath 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8'
```

Replace `CudaPath` with the installed Toolkit directory. `PackageDirectory` must be new. The script builds ITK, VTK, Teem, Eigen, SlicerExecutionModel, and zlib using `platforms/superbuild-2026-10-14-r4`, then packages the standalone backend. The first build downloads and compiles dependencies; no Slicer SDK or Slicer source build is needed.

GPU targets are selected from `nvcc --list-gpu-code`, including supported architectures from compute capability 6.0 onward and baseline PTX. Override them with `-CudaArchitectures '89-real;89-virtual'`, for example, when building specifically for an RTX 4090. Supported compilation targets do not replace hardware validation.

The MSVC runtime defaults to `$env:VCToolsRedistDir\x64\Microsoft.VC143.CRT`. Use `-MsvcRuntimeDirectory` to select another x64 CRT directory. Packages contain `UKFTractography_gpu.exe`, `ukf_gpu.dll`, the matching `cudart64_*.dll`, and CRT DLLs. The script verifies GPU DLL exports and runs the packaged executable with a restricted PATH to detect missing runtime libraries.

Outputs include a version-specific `UKFGPU-Slicer5.6-windows-x86_64-<date>.tar.gz` and a version-adaptive `UKFGPU-Installer-windows-x86_64.zip`. The ZIP adapts its installation layout to the running Slicer version. For a version-specific archive, choose `SlicerVersion` matching the target application's major.minor version.

To package an existing native build independently:

```powershell
python .\scripts\package_slicer_extension.py `
  --platform windows `
  --binary C:\UKF\build\backend\bin\UKFTractography_gpu.exe `
  --gpu-library C:\UKF\build\backend\bin\ukf_gpu.dll `
  --cuda-runtime 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\cudart64_12.dll' `
  --runtime-dir 'C:\path\to\x64\Microsoft.VC143.CRT' `
  --slicer-version 5.6 `
  --output C:\UKF\new-package
```

Use the executable and GPU DLL from the same build. The packager requires native Windows x64 PE files and matching runtime libraries.

## Automated checks

The Windows installer workflow builds the native backend, checks packaging contracts, executes a synthetic CPU case, and installs/reloads the extension in real Slicer applications. To run installation checks for a local bundle:

```powershell
.\scripts\test_windows_slicer.ps1 `
  -Bundle C:\UKF\package-5.6\UKFGPU-Installer-windows-x86_64.zip `
  -WorkDirectory C:\UKF\slicer-check `
  -Versions 5.6.2,latest
```

These checks download Slicer and require a new work directory. They establish installation and loading behavior; GPU tracking must also satisfy the accuracy requirements in the [main README](../README.md#features-and-current-status) on the target hardware. Numerical acceptance tools and results are maintained outside this repository.
