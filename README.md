# UKFTractography GPU

CUDA acceleration for unscented Kalman filter diffusion MRI tractography, with a command-line backend and a 3D Slicer extension.

This is a modified version of [pnlbwh/ukftractography](https://github.com/pnlbwh/ukftractography), based on upstream commit `2d2b661`. The original authors' contributions, source notices, and [license](LICENSE.txt) are retained.

## Features and current status

- GPU filtering, constraints, tracking, and recording for 12 experimental models: Simple1T/2T/3T, Full1T/2T/3T, Simple1T/2T-FreeWater, Full1T/2T-FreeWater, and NODDI1F/2F.
- FP64 by default. `--gpuPrecision 1` uses guarded FP32 range selection inside compatible signal exponentials, with FP64 fallback near rounding boundaries. Interpolation, exponential polynomials, state/covariance, and filter updates remain FP64. This is a limited mixed path; it does not run the whole observation or filter in FP32. Unsupported host math policies retain FP64. DWI input storage is FP32 in both modes.
- GPU tracking kernels and compact record transfers; see [how acceleration is implemented](ACCELERATION.md).
- A Slicer interface with input validation, automatic GPU discovery/free-memory recommendation, single or explicitly selected multi-GPU tracking, real completion counts and elapsed time, scene-persistent parameters, custom fiber names, and direction/FA/length/solid coloring.
- Linux integration tested with Slicer 5.2.2, 5.6.2 and 5.12.4. Tracking results appear directly in the scene; parameter controls provide hover help. Windows native compilation, CPU execution, and extension installation have been checked in CI with Slicer 5.6.2 and 5.12.4. Windows GPU tracking still requires verification on NVIDIA hardware.

Accuracy acceptance requires the same fiber order and per-fiber point counts, positional Euclidean error at most **0.001 mm**, and recorded-array checks. Results remain workload-specific; long whole-brain, branching, and other platform/device combinations require separate validation. Full2T FreeWater can also fail in the upstream CPU constraint solver for some inputs; the original CPU baseline is retained, and these inputs remain a known limitation. The upstream zero-tensor FreeWater case can produce NaN FA; its CPU behavior is intentionally preserved, and that case does not pass finite-record validation.

## 3D Slicer

The recommended route is to **build and install on your own computer** using the [local build commands](slicer-extension/README.md#install). The scripts locate Slicer and the CUDA Toolkit, download/build the library dependencies, and launch Slicer for installation. Precompiled installer ZIPs remain an optional alternative. Slicer 5.2 or newer, Linux/Windows x86_64, and a compatible NVIDIA driver are required.

The same guide covers prerequisites, upgrades, [Slicer usage](slicer-extension/README.md#use), and [optional installer packaging](slicer-extension/README.md#build-installers). All installation steps are maintained there.

## Build

For the recommended local Slicer build/install scripts, use the [Slicer installation guide](slicer-extension/README.md#install). The commands below are the lower-level backend build route for developers.

Clone this repository:

```bash
git clone https://github.com/yishengpoxiao/ukftractography_gpu.git
cd ukftractography_gpu
```

The standalone backend requires a C++17 compiler, CMake 3.22+, CUDA Toolkit compatible with your compiler and GPU, ITK 5, VTK 9, Eigen 3.4, Teem, SlicerExecutionModel, and zlib. Use a CUDA Toolkit that supports your compiler and GPU architecture.

With dependencies already available, configure `platforms/standalone` and provide their CMake package paths. Alternatively, the dependency superbuild provides pinned dependency sources. Use a new build directory when changing build configurations.

On **Linux**, find the CUDA compiler with:

```bash
command -v nvcc
```

Replace `/path/to/nvcc` below with the returned executable path, for example `/usr/local/cuda/bin/nvcc`. If no path is returned, install the CUDA Toolkit or add its `bin` directory to `PATH`.

```bash
cmake -S platforms/superbuild -B ../build-ukfgpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/path/to/nvcc
cmake --build ../build-ukfgpu --parallel 8
```

GPU targets are detected from `nvcc --list-gpu-code`; no fixed CUDA Toolkit version is required. Override with `-DUKF_CUDA_ARCHITECTURES="75-real;89-real;75-virtual"` when needed. The Toolkit must support C++17 and the selected host compiler. The backend is produced in `../build-ukfgpu/backend/bin/`. The standalone backend has been compiled and checked on Linux; a complete Linux dependency superbuild from scratch remains to be verified.

On **Windows**, open **x64 Developer PowerShell for VS 2022** and locate the compiler and Toolkit directory:

```powershell
where.exe nvcc
$env:CUDA_PATH
```

`where.exe nvcc` prints the executable path, for example `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nvcc.exe`. `CudaPath` takes the Toolkit root directory, without `\bin\nvcc.exe`; `$env:CUDA_PATH` usually contains that directory. Build with:

```powershell
.\scripts\build_windows_gpu.ps1 `
  -BuildDirectory C:\UKF\build `
  -PackageDirectory C:\UKF\package-5.6 `
  -SlicerVersion 5.6 `
  -CudaPath $env:CUDA_PATH
```

If `CUDA_PATH` is empty, supply the actual root explicitly, for example `-CudaPath 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8'`. The version in this path is only an example; use your installed Toolkit version. If `where.exe nvcc` finds nothing, check that the Toolkit is installed and its `bin` directory is on `PATH`.

Use a short checkout/build path, and install the Visual Studio C++ workload and Windows SDK, CMake, Ninja, Git, and Python 3.9+ on PATH. The driver's CUDA/UMD version does not identify an installed Toolkit; check `nvcc --version`.

The script builds the dependencies and backend without a Slicer SDK, packages native dependencies privately, verifies GPU DLL exports, and checks executable loading with a restricted PATH. `PackageDirectory` must be new. Select the target Slicer's major.minor version with `SlicerVersion`. Override GPU targets with `-CudaArchitectures '89-real;89-virtual'` when needed. The MSVC runtime defaults to `$env:VCToolsRedistDir\x64\Microsoft.VC143.CRT`; use `-MsvcRuntimeDirectory` for another x64 CRT directory.

The Windows installer workflow produces optional build artifacts and checks backend changes. The local-build workflow additionally checks tool discovery without compiling a distribution package. Windows GPU execution still requires hardware validation.

## Command-line usage

See [parameters and worked examples](USAGE.md) for model flags, defaults, CPU/GPU runs, ROI seeding, FreeWater/NODDI, and Slicer settings.

```bash
/path/to/UKFTractography_gpu \
  --dwiFile UKFTractography/Data/Input/two_tensor.nhdr \
  --maskFile UKFTractography/Data/Input/mask.nhdr \
  --seedsFile UKFTractography/Data/Input/seed.nhdr \
  --tracts /new/output/fibers.vtk \
  --numTensor 2 --seedsPerVoxel 1 --useGPU \
  --gpuDevice 0 --gpuPrecision 0
```

Run the executable with `--help` for the complete option list. The output directory must exist. Runtime library search paths must include the GPU library and CUDA runtime; packaged Slicer runtimes handle this through their launcher.

## HCP YA speed comparison

One HCP Young Adult (HCP YA) b≈3000 s/mm² dataset: **1.25 mm isotropic**, **145 × 174 × 145 voxels**, and **108 volumes** (90 diffusion-weighted + 18 b0). The DWI and brain mask were copied to local disk before timing; network-copy time is excluded. Output was approximately **0.58 million streamlines**.

Simple2T, FP64, **3 seeds per voxel**; CPU: **16 physical cores / 16 threads** on an Intel Xeon Gold 6330; GPU: **one RTX 4090**, selected when idle. Both runs used `--numThreads 16`, a 0.3 mm tracking step, 0.9 mm recording spacing, 250 mm maximum half-fiber length, seed FA 0.18, stopping FA 0.10, stopping mean signal 0.05, `Qm=0.001`, `Ql=10`, `Rs=0.015`, and no branching or optional recording arrays.

| Backend | Tracking | End-to-end |
| --- | ---: | ---: |
| CPU, 16 cores | 2047.5 s | 2182.1 s |
| GPU, one RTX 4090 | 392.7 s | 501.0 s |
| Speedup (CPU / GPU) | **5.21×** | **4.36×** |

End-to-end includes local input loading, normalization, seeding, tracking, postprocessing, and VTK writing. These are single sequential measurements on a shared server; speedup depends on the workload. CPU/GPU fiber order and per-fiber point counts matched, with maximum stored position error 0 mm. See [acceleration details](ACCELERATION.md).

## Development and data

Small synthetic inputs and CPU baselines are included. The approximately 100 MB upstream whole-brain DWI payload is excluded from the source tree. Restore it when needed:

```bash
python scripts/fetch_benchmark_data.py
```

This downloads the data from a pinned upstream commit and checks its SHA-256 hash. The matching header and masks remain in `UKFTractography/Data/Input/`.

The current standalone and dependency-superbuild entries are `platforms/standalone` and `platforms/superbuild`. The current CLI description is `UKFTractography/UKFTractography.xml`; the standalone directory contains the executable entry point and Windows UTF-8 manifest. Root CMake/SuperBuild files provide the original upstream build route.

The local-build workflow checks Linux Python contracts and Windows tool/Slicer discovery from ordinary PowerShell. The Windows installer workflow additionally checks native compilation, CPU execution, packaging contracts, and actual extension installation/module loading. Python unit and installation checks are in `slicer-extension/UKFGPU/Testing/`.

Numerical acceptance scripts, detailed benchmark logs, generated results, build outputs, and packaged runtimes are maintained outside this repository. Original upstream CPU regression tests and their input/baseline data are retained for the upstream build route.

## Attribution and license

Original UKFTractography authors: Yogesh Rathi, Stefan Lienhard, Yinpeng Li, Martin Styner, Ipek Oguz, Yundi Shi, Christian Baumgartner, Ryan Eckbo, Tashrif Billah, and Dheshan Mohandass.

All or portions of this licensed product (such portions are the "Software") have been obtained under license from The Brigham and Women's Hospital, Inc. and are subject to the terms and conditions in [LICENSE.txt](LICENSE.txt). This repository contains modifications to the original software. Preserve the license and applicable source attributions when redistributing it.
