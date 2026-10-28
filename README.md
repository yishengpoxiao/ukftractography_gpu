# UKFTractography GPU

CUDA-accelerated unscented Kalman filter tractography for diffusion MRI, with a command-line backend and a 3D Slicer extension.

This project modifies [pnlbwh/ukftractography](https://github.com/pnlbwh/ukftractography), based on upstream commit `2d2b661`. Original source notices are retained; redistribution terms are in [LICENSE.txt](LICENSE.txt).

## Models and status

The GPU backend supports 12 experimental models: Simple1T/2T/3T, Full1T/2T/3T, Simple1T/2T-FreeWater, Full1T/2T-FreeWater, and NODDI1F/2F.

FP64 is the default. Optional mixed precision (`--gpuPrecision 1`) uses guarded FP32 exponential range selection; sensitive filtering and tracking calculations remain FP64. See [acceleration details](ACCELERATION.md).

The Slicer extension provides GPU selection, tracking controls, asynchronous execution, logs, and fiber display. Linux integration has been tested with Slicer 5.2.2, 5.6.2, and 5.12.4. Windows build, CPU execution, and extension installation are checked in CI; Windows GPU tracking still needs NVIDIA hardware validation.

Known limitations include upstream CPU constraint-solver failures for some Full2T FreeWater inputs and NaN FA in the upstream zero-tensor FreeWater case. See the [parameter reference and examples](USAGE.md) for model behavior and options.

## Install and build

The recommended route is to build and install locally with the [Linux or Windows build scripts](slicer-extension/README.md#install). They build pinned dependencies, compile the backend, package it, and launch Slicer for installation. You need Slicer 5.2 or newer, an NVIDIA driver, and a compatible CUDA Toolkit; platform-specific compiler and build-tool requirements are listed in the [installation guide](slicer-extension/README.md#install).

Clone the repository:

```bash
git clone https://github.com/yishengpoxiao/ukftractography_gpu.git
cd ukftractography_gpu
```

Then run the command for your operating system:

```bash
python3 scripts/build_local_linux.py --slicer /path/to/Slicer
```

```powershell
.\scripts\build_local_windows.ps1 -SlicerPath 'C:\path\to\Slicer.exe'
```

Use `--check-only` / `-CheckOnly` to inspect prerequisites or `--no-install` / `-NoInstall` to build without launching Slicer. Prebuilt installer ZIPs are an optional alternative described in the [installation guide](slicer-extension/README.md#optional-offline-installer).

See [USAGE.md](USAGE.md) for command-line parameters and examples, and [ACCELERATION.md](ACCELERATION.md) for the GPU implementation.

## HCP YA speed comparison

HCP Young Adult b≈3000 s/mm² data: **1.25 mm isotropic**, **145 × 174 × 145 voxels**, **108 volumes**, copied to local disk. Simple2T, **3 seeds/voxel**, about **0.58 million streamlines**. CPU: **16 physical cores** on an Intel Xeon Gold 6330; GPU: **one RTX 4090 selected while idle**. All runs used the same Release/AVX build and [tracking parameters](USAGE.md#reproduce-the-hcp-cpugpu-parameter-set).

| Backend | Tracking | End-to-end | Tracking speedup | Total speedup |
| --- | ---: | ---: | ---: | ---: |
| CPU, 16 cores | 1881.7 s | 2015.6 s | 1.00× | 1.00× |
| GPU, FP64 | 289.5 s | 410.6 s | 6.50× | 4.91× |
| GPU, guarded mixed | 286.1 s | 395.8 s | 6.58× | 5.09× |

Single sequential measurements on a shared server; end-to-end includes input loading and VTK writing. Input loading varied between runs, so the small mixed/FP64 difference does not establish a consistent mixed-precision advantage. Fiber order and per-fiber point counts matched across all three outputs, with **0 mm maximum stored position error** (limit: 0.001 mm); recorded arrays also matched. Detailed logs and outputs remain outside this repository.

## Development data

Small synthetic inputs and CPU baselines are included. The approximately 100 MB upstream whole-brain DWI payload can be restored with `python scripts/fetch_benchmark_data.py`; the script checks its pinned SHA-256 hash.
