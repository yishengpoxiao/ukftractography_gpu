# Parameters and usage examples

This guide covers the current standalone CLI and the parameters exposed by the Slicer module. Use the [current Slicer installation guide](slicer-extension/README.md#install) for installer downloads and setup, or the [source build instructions](README.md#install-and-build) to compile the backend. The [acceleration document](ACCELERATION.md) explains the GPU implementation.

## Inputs and output

| Parameter | Meaning |
| --- | --- |
| `--dwiFile` | DWI NRRD containing DWMRI gradients and b-value metadata. For a detached `.nhdr`, keep its referenced payload alongside it or use a valid payload path. |
| `--maskFile` | Brain mask defining the tracking region. It must share the DWI's voxel grid and spatial geometry. |
| `--tracts` | Output `.vtk` or `.vtp` file. Create its parent directory before running. |
| `--seedsFile` | Optional integer labelmap defining seed regions. Without it or a WM probability map, seeds are drawn from eligible voxels in the brain mask. |
| `--seedLabels` | Comma-separated seed labels, such as `1,2`; default `1`. Used with `--seedsFile`. |

The CLI consumes the diffusion metadata from the NRRD header. A detached header and its data file form one input. Slicer inputs should be aligned volumes without parent transforms; export segmentation seed regions to a labelmap before selecting them.

Use brain masks with **0 outside and 1 inside**. To preserve the upstream CPU reference, byte masks are interpreted through signed `char` on the supported Linux/Windows builds: unsigned-byte values 128–255 are treated as outside the tracking mask. The GPU follows the same interpretation; convert a 0/255 mask to 0/1 before tracking.

## Model selection

Use exactly one row's model flags. `--fullTensorModel` permits three distinct eigenvalues per tensor; Simple models constrain the two smaller eigenvalues to be equal. FreeWater adds an isotropic compartment. NODDI uses its own compartment model and orientation count.

| Slicer model | CLI model flags |
| --- | --- |
| Simple1T | `--numTensor 1` |
| Simple2T | `--numTensor 2` |
| Simple3T | `--numTensor 3` |
| Full1T | `--numTensor 1 --fullTensorModel` |
| Full2T | `--numTensor 2 --fullTensorModel` |
| Full3T | `--numTensor 3 --fullTensorModel` |
| Simple1T-FW | `--numTensor 1 --freeWater` |
| Simple2T-FW | `--numTensor 2 --freeWater` |
| Full1T-FW | `--numTensor 1 --fullTensorModel --freeWater` |
| Full2T-FW | `--numTensor 2 --fullTensorModel --freeWater` |
| NODDI1F | `--numTensor 1 --noddi` |
| NODDI2F | `--numTensor 2 --noddi` |

FreeWater tensor models support 1T/2T. NODDI supports one or two orientations; use `--recordViso` for its isotropic fraction. Tensor branching requires a multi-tensor model and an explicit seed labelmap. Some inherited `--help` model descriptions predate the current GPU implementation; this table reflects the current model selection.

## Execution and precision

| Parameter | Default | Meaning |
| --- | --- | --- |
| `--useGPU` | Off | Enable CUDA tracking. Omit this switch for the CPU reference path. |
| `--gpuDevice` | `0` | CUDA ordinal within the visible device list. `-1` distributes seeds over visible GPUs. |
| `--gpuPrecision` | `0` | `0`: FP64. `1`: guarded FP32 range selection for compatible signal exponentials, with FP64 fallback near rounding boundaries. Interpolation, exponential polynomials, state/covariance, and filtering remain FP64. DWI input storage is FP32 in both modes. |
| `--numThreads` | Detected CPU count | CPU worker count; set `16` for the benchmark configuration. This does not set CUDA thread or GPU counts. |

FP64 is the default. The mixed path preserves sensitive trajectory arithmetic in FP64 and uses FP32 only for selected operations; any speed benefit depends on the workload and host math policy. See the [accuracy requirements and known limitations](README.md#models-and-status).

`CUDA_VISIBLE_DEVICES` restricts and reorders the visible CUDA devices. With `CUDA_VISIBLE_DEVICES=1`, `--gpuDevice 0` selects the first device in that restricted list. GPU UUIDs can also identify a physical card. Check device availability before starting work on a shared machine. A successful GPU run logs `tractography done on device`; a `falling back` message means CPU fallback and must be excluded from GPU speed measurements.

## Seeding, sampling, and stopping

These are the CLI defaults; the Slicer presets are described below.

| Parameter | Default | Effect |
| --- | --- | --- |
| `--seedsPerVoxel` | `1` | Initial seed positions per eligible voxel. More seeds increase work, output density, and memory. Each accepted seed tracks in both directions; filtering and stopping affect the final streamline count. |
| `--seedingThreshold` | `0.18` | Initial FA cutoff. Whole-brain mask seeding also applies this value to normalized mean signal. The initial FA filter remains active with a seed labelmap or WM probability map. |
| `--stepLength` | `0.3` mm | Tracking step in physical space. Smaller steps increase computation. |
| `--recordLength` | `0.9` mm | Nominal spacing between saved fiber points. Smaller spacing increases output size and memory. Must be at least `stepLength`; choose integer multiples because recording uses an integer number of steps. |
| `--maxHalfFiberLength` | `250` mm | Tracking length limit in each direction from a seed. The combined fiber can span approximately twice this length. |
| `--stoppingFA` | `0.15` | Tensor FA stopping cutoff. Lower values allow tracking through lower-FA regions; unused by NODDI. |
| `--stoppingThreshold` | `0.1` | Normalized model mean-signal stopping cutoff. Lower values allow continuation at weaker signal. |
| `--maxBranchingAngle` | `0` degrees | `0` disables branching. A positive value enables eligible tensor branches up to this angle. |
| `--minBranchingAngle` | `0` degrees | Lower branching angle. Enabled branching requires `0 <= min < max <= 90`. |

The CLI's omitted step/record lengths remain 0.3/0.9 mm for 3T. Use explicit 0.15/0.45 mm values to match the Slicer Standard 3T preset. The stopping descriptions above concern the usual mask-based workflow; additional probability-map and stopping-labelmap options are listed in `--help`.

## Filter noise

| Parameter | Meaning | Applicable models |
| --- | --- | --- |
| `--Qm` | Orientation process noise; larger values allow more direction change between filter updates. | All |
| `--Ql` | Tensor eigenvalue process noise. | Tensor models |
| `--Qw` | FreeWater mixture-weight process noise. | FreeWater tensor models |
| `--Rs` | Expected signal measurement noise; larger values give measurements less weight. | All |
| `--Qkappa` | NODDI orientation-concentration process noise. | NODDI |
| `--Qvic` | NODDI intracellular/isotropic compartment-fraction process noise. | NODDI |

For these parameters, an explicit `0` requests the backend's model-dependent value. The table below lists those **automatic values**, including FreeWater variants where applicable.

| Model family | Qm | Ql | Rs | Additional automatic values |
| --- | ---: | ---: | ---: | --- |
| Simple1T / Full1T | 0.005 | 300 | 0.01 | FreeWater: Qw = 0.0025 |
| Simple2T | 0.001 | 50 | 0.02 | FreeWater: Qw = 0.0015 |
| Full2T | 0.002 | 50 | 0.01 | FreeWater: Qw = 0.0015 |
| Simple3T | 0.001 | 100 | 0.02 | — |
| Full3T | 0.002 | 100 | 0.01 | — |
| NODDI1F | 0.0025 | — | 0.01 | Qkappa = 0.01; Qvic = 0.0005 |
| NODDI2F | 0.001 | — | 0.02 | Qkappa = 0.01; Qvic = 0.004 |

The CLI initializes omitted `Qvic` to 0.004. For NODDI1F's automatic value, explicitly pass `--Qvic 0`; the Slicer presets already request automatic noise values. The HCP benchmark explicitly used `Ql=10` and `Rs=0.015`, which differ from the automatic Simple2T values. Parameter choices change the fitted trajectories and should be checked for the intended dataset.

## Recording

Recording options are switches: write `--recordFA`, without a following `true` or `false`. Optional recording is off by default in the current standalone CLI and Slicer presets.

| Switch | Recorded quantity |
| --- | --- |
| `--recordFA` | Tensor fractional anisotropy. |
| `--recordTrace` | Tensor eigenvalue sum. |
| `--recordTensors` | Fitted tensors, useful for tensor-based coloring and analysis. |
| `--recordFreeWater` | FreeWater fraction; requires a FreeWater tensor model. |
| `--recordVic` | NODDI intracellular fraction. |
| `--recordKappa` | NODDI orientation concentration. |
| `--recordViso` | NODDI isotropic fraction. |
| `--recordNMSE` | Normalized mean squared signal-fitting error. |
| `--recordState` | Complete UKF state for analysis/debugging. |
| `--recordCovariance` | UKF covariance for analysis/debugging; storage grows quadratically with state dimension. |

Extra arrays increase memory and output size. Enable the quantities you need. The HCP timing comparison used no optional recording switches.

## Bash examples

Run from the repository root. For a packaged Linux backend, use its launcher so that private runtime libraries are found:

```bash
UKF=/absolute/path/to/Runtime/bin/ukf-gpu
UKF_OUT=../ukf-output
mkdir -p "$UKF_OUT"
```

A small example using the included synthetic input:

```bash
"$UKF" \
  --dwiFile UKFTractography/Data/Input/two_tensor.nhdr \
  --maskFile UKFTractography/Data/Input/mask.nhdr \
  --seedsFile UKFTractography/Data/Input/seed.nhdr --seedLabels 1 \
  --tracts "$UKF_OUT/synthetic_2t.vtk" \
  --numTensor 2 --numThreads 16 --seedsPerVoxel 1 \
  --useGPU --gpuDevice 0 --gpuPrecision 0 \
  --recordFA --recordTensors
```

### Reproduce the HCP CPU/GPU parameter set

Copy the DWI header, its referenced payload, the brain-mask header, and its payload to a local directory first. Replace `UKF_DATA` with that directory. These Bash arrays keep the CPU/GPU inputs and tracking parameters identical:

```bash
UKF_DATA=/local/HCP/103818
UKF_DATA_ARGS=(
  --dwiFile "$UKF_DATA/103818_b3000_dwi.nhdr"
  --maskFile "$UKF_DATA/103818_brain_mask.nhdr"
  --numThreads 16 --seedsPerVoxel 3
)
UKF_TENSOR_ARGS=(
  --numTensor 2 --seedingThreshold 0.18
  --stoppingFA 0.10 --stoppingThreshold 0.05
  --Qm 0.001 --Ql 10 --Rs 0.015
  --stepLength 0.3 --recordLength 0.9 --maxHalfFiberLength 250
  --minBranchingAngle 0 --maxBranchingAngle 0
)

# CPU, 16 worker threads
"$UKF" "${UKF_DATA_ARGS[@]}" "${UKF_TENSOR_ARGS[@]}" \
  --tracts "$UKF_OUT/hcp_cpu.vtk"

# One GPU: visible CUDA ordinal 0
CUDA_VISIBLE_DEVICES=0 "$UKF" "${UKF_DATA_ARGS[@]}" "${UKF_TENSOR_ARGS[@]}" \
  --useGPU --gpuDevice 0 --gpuPrecision 0 \
  --tracts "$UKF_OUT/hcp_gpu.vtk"
```

The measured benchmark additionally pinned both processes to CPU cores 0–15 with Linux `taskset -c 0-15`. `--numThreads 16` alone selects worker count without CPU affinity. Run the commands sequentially when comparing times. For the mixed run, repeat the GPU command with `--gpuPrecision 1` and a new output filename. See the [measured HCP results](README.md#hcp-ya-speed-comparison).

### ROI seeding and model variants

Reuse `UKF_DATA_ARGS` from above. Select labels from an aligned seed map:

```bash
"$UKF" "${UKF_DATA_ARGS[@]}" "${UKF_TENSOR_ARGS[@]}" \
  --seedsFile /local/seed_labels.nhdr --seedLabels 1,2 \
  --useGPU --gpuDevice 0 --gpuPrecision 0 \
  --recordFA --recordTensors --tracts "$UKF_OUT/roi_2t.vtk"
```

The following model examples use automatic filter noise and the normal stopping defaults, rather than the HCP-specific overrides:

```bash
# Full3T, with sampling matching Slicer's Standard 3T preset
"$UKF" "${UKF_DATA_ARGS[@]}" \
  --numTensor 3 --fullTensorModel --stepLength 0.15 --recordLength 0.45 \
  --useGPU --gpuDevice 0 --gpuPrecision 0 \
  --recordFA --recordTensors --tracts "$UKF_OUT/full3t.vtk"

# Simple2T FreeWater
"$UKF" "${UKF_DATA_ARGS[@]}" \
  --numTensor 2 --freeWater --Qm 0 --Ql 0 --Qw 0 --Rs 0 \
  --useGPU --gpuDevice 0 --gpuPrecision 0 \
  --recordFreeWater --tracts "$UKF_OUT/simple2t_fw.vtk"

# NODDI2F; for NODDI1F, change numTensor to 1
"$UKF" "${UKF_DATA_ARGS[@]}" \
  --numTensor 2 --noddi --Qm 0 --Rs 0 --Qkappa 0 --Qvic 0 \
  --useGPU --gpuDevice 0 --gpuPrecision 0 \
  --recordVic --recordKappa --recordViso --tracts "$UKF_OUT/noddi2f.vtk"
```

To branch the multi-tensor ROI example, change the two angle values in `UKF_TENSOR_ARGS` to `--minBranchingAngle 10 --maxBranchingAngle 60`, then run the ROI command. Set these arguments once; repeating a value argument is rejected by the CLI parser. Branching increases work and output count. For multiple GPUs, select the visible devices and use `--gpuDevice -1`:

```bash
env -u UKF_GPU_DEVICES CUDA_VISIBLE_DEVICES=0,1 \
  "$UKF" "${UKF_DATA_ARGS[@]}" "${UKF_TENSOR_ARGS[@]}" \
  --useGPU --gpuDevice -1 --gpuPrecision 0 \
  --tracts "$UKF_OUT/hcp_multi_gpu.vtk"
```

Each GPU keeps its own input copy; GPU memory is not pooled. `--gpuPrecision 1` enables guarded mixed precision for all models: compatible signal exponentials select their range in FP32 and fall back to FP64 near rounding boundaries. Their residuals and polynomials, interpolation, state/covariance, and UKF updates remain FP64. Other host math policies and the Eigen packet exponential retain their FP64 paths. This limits FP32 work to operations that preserve the reference signal values; speed depends on the model and hardware.

## Windows PowerShell example

Use the packaged native executable and create the output directory. This example uses the HCP parameter set:

```powershell
$UkfExe = 'C:\UKF\Runtime\bin\UKFTractography_gpu.exe'
$UkfData = 'C:\HCP\103818'
$UkfOutput = 'C:\UKF\output'
New-Item -ItemType Directory -Force $UkfOutput | Out-Null

& $UkfExe `
  --dwiFile "$UkfData\103818_b3000_dwi.nhdr" `
  --maskFile "$UkfData\103818_brain_mask.nhdr" `
  --tracts "$UkfOutput\hcp_gpu.vtk" `
  --numTensor 2 --numThreads 16 --seedsPerVoxel 3 `
  --seedingThreshold 0.18 --stoppingFA 0.10 --stoppingThreshold 0.05 `
  --Qm 0.001 --Ql 10 --Rs 0.015 `
  --stepLength 0.3 --recordLength 0.9 --maxHalfFiberLength 250 `
  --minBranchingAngle 0 --maxBranchingAngle 0 `
  --useGPU --gpuDevice 0 --gpuPrecision 0
```

For CPU tracking, omit the final GPU switches and choose a different output filename. Windows native build/installation checks have passed; Windows GPU hardware validation remains pending.

## Slicer parameter workflow

Open **Diffusion > Tractography > UKF GPU Tractography**, select the DWI, mask, and optional seed labelmap, then choose a model and an idle GPU. **Fiber name** sets the output scene-node name; results appear directly in the scene. Use Slicer's Save dialog to persist them.

Choose a preset and click **Apply** before making manual changes. Standard uses automatic model noise and 0.3/0.9 mm step/record spacing, or 0.15/0.45 mm for 3T. Quick preview uses 50 mm per direction and 1.8 mm recording spacing. Fine sampling halves Standard sampling. Applying a preset replaces tracking values and clears recording options; model, precision, GPU selection, inputs, and fiber name are preserved.

For the HCP benchmark settings, select Simple2T and FP64, apply Standard, then set seeds per voxel to 3, CPU threads to 16, stopping FA to 0.10, stopping mean signal to 0.05, Qm to 0.001, Ql to 10, and Rs to 0.015. Keep branching and optional recording off. Hover over labels or controls for help; the module disables parameters that do not apply to the selected model. The [Slicer guide](slicer-extension/README.md#use) covers installation, GPU selection, visualization, and saving in more detail.
