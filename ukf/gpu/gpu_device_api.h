/**
 * \file gpu_device_api.h
 * \brief C++ ABI surface between the host orchestration (gpu_tractography.cc,
 *        compiled without nvcc) and the CUDA implementation
 *        (gpu_kernels.cu).
 */
#ifndef UKF_GPU_DEVICE_API_H_
#define UKF_GPU_DEVICE_API_H_

#include <string>
#include <vector>
#include <cstddef>

#include "gpu_types.h"

#if defined(_WIN32) && defined(UKF_GPU_SHARED)
# if defined(UKF_GPU_BUILD)
#  define UKF_GPU_API __declspec(dllexport)
# else
#  define UKF_GPU_API __declspec(dllimport)
# endif
#else
# define UKF_GPU_API
#endif

/* Returns true if at least one CUDA device is usable. */
UKF_GPU_API bool gpu_available(std::string& info);

/* Number of visible CUDA devices (0 on error). */
UKF_GPU_API void cuda_count_devices(int& ndev);

/* Free/total device memory in bytes. */
UKF_GPU_API void cuda_device_mem_info(int device, size_t& free_b, size_t& total_b);

/* Runs the wavefront tractography for S seeds.
 * As in NrrdData, gradients contain G directions followed by their negatives,
 * with the same b-values. Signal values for the two halves are identical.
 * Output arrays are sized S * max_rec (per record scalar) and written in
 * seed order. Optional scalar outputs are only written when their recording
 * flag is enabled and may otherwise be null. State records may also be null
 * when record_state is disabled. out_reccount holds the number of recorded points per fiber. */
UKF_GPU_API bool gpu_compute(const GPUDarParams& params_in,
                 const float* dwi, const void* mask, const void* stop,
                 const float* gm, const float* csf,
                 const double* gradients, const double* bvalues,
                 std::size_t dwi_elems,
                 const double* seed_pos, const double* seed_state,
                 const double* seed_cov, const double* seed_fa,
                 const double* seed_fa2, const double* seed_trace,
                 const double* seed_trace2, const double* seed_dir,
                 int S,
                 int* out_reccount,
                 double* out_rec_pos, double* out_rec_norm, double* out_rec_state,
                 double* out_rec_cov, double* out_rec_nmse,
                 double* out_rec_trace, double* out_rec_trace2,
                 double* out_rec_fa, double* out_rec_fa2,
                 int device, int chunk_size, std::string& err,
                 const double* noddi_table = NULL,
                 std::vector<GPUBranchSeed>* branches = NULL);

#endif  /* UKF_GPU_DEVICE_API_H_ */
