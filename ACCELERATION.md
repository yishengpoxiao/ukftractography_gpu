# GPU acceleration

The CUDA backend keeps the repeated UKF tracking steps on the GPU. Input loading, normalization, seed initialization, fiber assembly, and VTK writing remain on the CPU. The original CPU model equations provide the numerical reference.

## Main changes

- **Device tracking:** signal interpolation, sigma-point prediction, model observations, filter updates, applicable constraints, stopping decisions, and recording run in CUDA kernels across independent fibers. Tensor branching and all 12 tensor, FreeWater, and NODDI models use this path. See [gpu_kernels.cu](ukf/gpu/gpu_kernels.cu) and [gpu_general.cuh](ukf/gpu/gpu_general.cuh).
- **Less synchronization:** ordinary tracking compacts the active-fiber list every 16 steps. Kernels skip inactive fibers between compactions; branching and profiling samples synchronize when their outputs are needed. This reduces host/device round trips.
- **Parallel filter calculations:** sigma observations use multiple warps, and filter updates use 128-thread blocks. Small matrix operations and independent products run cooperatively, while sensitive reductions retain the CPU calculation order. See [gpu_eigen_math.cuh](ukf/gpu/gpu_eigen_math.cuh).
- **Reuse model calculations:** FreeWater caches the state-independent isotropic signal. NODDI shares each Legendre recurrence prefix and evaluates independent sigma observations in parallel, reducing repeated polynomial and signal calculations. See [gpu_general.cuh](ukf/gpu/gpu_general.cuh) and [gpu_noddi.cuh](ukf/gpu/gpu_noddi.cuh).
- **Smaller transfers:** tracking uses device-side work buffers and memory-sized fiber chunks. When recording is sparse, only populated records are packed and downloaded, avoiding unused fixed-stride capacity. See [gpu_record_transfer.cuh](ukf/gpu/gpu_record_transfer.cuh) and [gpu_tractography.cc](ukf/gpu/gpu_tractography.cc).
- **Optional recording work:** NMSE is computed only when requested. GPU scalar record buffers for NMSE, trace, and FA are allocated and written only when enabled; the batch memory estimate follows those choices. FA calculations needed for stopping and tensor selection are retained.
- **Reuse interpolation weights:** the up-to-27 valid Gaussian spatial weights are computed once per fiber per tracking step and reused across all gradient channels. The FP64 weight calculations and ascending neighbor accumulation order are preserved.
- **GPU selection:** automatic multi-GPU selection accounts for the DWI, masks, probability maps, and model tables before applying its working-memory threshold. This avoids selecting a card solely because it has 2 GiB free when the input itself needs much of that memory. Explicit device selection remains available.

## Precision and timing

FP64 trajectory arithmetic is used in both precision modes. `--gpuPrecision 1` retains the mixed-precision request but currently auto-promotes interpolation, signal prediction, and filtering to FP64; it follows the same trajectory calculation path as `--gpuPrecision 0` and does not add FP32 compute speed. The DWI input is stored as FP32 in both modes. See the [accuracy requirements and known limitations](README.md#features-and-current-status).

Tracking time measures the tracking phase, including its host/device work. End-to-end time also includes local input loading, normalization, seeding, postprocessing, and file writing. These CPU stages limit overall speedup even when tracking accelerates substantially. The [HCP YA comparison](README.md#hcp-ya-speed-comparison) uses identical input and tracking parameters for CPU and GPU.
