# GPU acceleration

The CUDA backend keeps the repeated UKF tracking steps on the GPU. Input loading, normalization, seed initialization, fiber assembly, and VTK writing remain on the CPU. The original CPU model equations provide the numerical reference.

## Main changes

- **Device tracking:** signal interpolation, sigma-point prediction, model observations, filter updates, applicable constraints, stopping decisions, and recording run in CUDA kernels across independent fibers. Tensor branching and all 12 tensor, FreeWater, and NODDI models use this path. See [gpu_kernels.cu](ukf/gpu/gpu_kernels.cu) and [gpu_general.cuh](ukf/gpu/gpu_general.cuh).
- **Less synchronization:** ordinary tracking compacts the active-fiber list every 16 steps. Kernels skip inactive fibers between compactions; branching and profiling samples synchronize when their outputs are needed. This reduces host/device round trips.
- **Parallel filter calculations:** sigma observations use multiple warps, and filter updates use 128-thread blocks. Small matrix operations and independent products run cooperatively, while sensitive reductions retain the CPU calculation order. See [gpu_eigen_math.cuh](ukf/gpu/gpu_eigen_math.cuh).
- **Parallel Simple tensor construction:** Simple1T/2T sigma observations build the two tensor frames concurrently within a warp and distribute their matrix entries across lanes, preserving the original FP64 product and addition order.
- **Reuse model calculations:** FreeWater caches the state-independent isotropic signal. NODDI shares each Legendre recurrence prefix and evaluates independent sigma observations in parallel, reducing repeated polynomial and signal calculations. See [gpu_general.cuh](ukf/gpu/gpu_general.cuh) and [gpu_noddi.cuh](ukf/gpu/gpu_noddi.cuh).
- **Smaller transfers:** tracking uses device-side work buffers and memory-sized fiber chunks. When recording is sparse, only populated records are packed and downloaded, avoiding unused fixed-stride capacity. See [gpu_record_transfer.cuh](ukf/gpu/gpu_record_transfer.cuh) and [gpu_tractography.cc](ukf/gpu/gpu_tractography.cc).
- **Optional recording work:** NMSE is computed only when requested. GPU scalar record buffers for NMSE, trace, and FA are allocated and written only when enabled; the batch memory estimate follows those choices. FA calculations needed for stopping and tensor selection are retained.
- **Reuse interpolation weights:** the up-to-27 valid Gaussian spatial weights are computed once per fiber per tracking step and reused across all gradient channels. The FP64 weight calculations and ascending neighbor accumulation order are preserved.
- **GPU selection:** automatic multi-GPU selection accounts for the DWI, masks, probability maps, and model tables before applying its working-memory threshold. This avoids selecting a card solely because it has 2 GiB free when the input itself needs much of that memory. Explicit device selection remains available.

## Precision and timing

FP64 remains the default. With `--gpuPrecision 1`, compatible signal exponentials use FP32 multiplication and rounding to select a table interval. A conservative error margin detects ambiguous half-integer boundaries and recomputes the interval with the original FP64 expression. The selected interval then uses the original FP64 residual, polynomial, and scaling operations. Other host math policies and Eigen's packet exponential retain FP64. See [gpu_mixed_math.cuh](ukf/gpu/gpu_mixed_math.cuh).

Interpolation, state/covariance, constraints, and filter updates retain FP64, and both modes store DWI input as FP32. Even rounding changes in small exponential terms can cause trajectory divergence in sensitive models, so the whole observation/filter is not converted to FP32. The guarded range selection is a limited optimization, not a promise of a large mixed-precision speedup. See the [accuracy requirements and known limitations](README.md#models-and-status).

Tracking time measures the tracking phase, including its host/device work. End-to-end time also includes local input loading, normalization, seeding, postprocessing, and file writing. These CPU stages limit overall speedup even when tracking accelerates substantially. The [HCP YA comparison](README.md#hcp-ya-speed-comparison) uses identical input and tracking parameters for CPU and GPU.
