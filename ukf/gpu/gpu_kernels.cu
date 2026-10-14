/**
 * \file gpu_kernels.cu
 * \brief CUDA implementation of the UKF tractography hot loop for the
 *        unconstrained Simple 1-tensor / 2-tensor models.
 *
 * Precision: --gpuPrecision 0 uses FP64 throughout. --gpuPrecision 1
 * uses FP32 for interpolation and signal prediction, while retaining FP64
 * state, covariance, sigma points, reductions, and information-form updates.
 * This avoids loss of precision in the ill-conditioned small matrices.
 * -fmad=false keeps the FP64 operation order close to the CPU reference.
 *
 * Numerics: every reduction accumulates in ascending order (matching the
 * sequential k-loops of the dense Eigen products on CPU), the matrix inverse
 * mirrors Eigen's PartialPivLU semantics, and the Cholesky mirrors Eigen's
 * LLT right-looking column algorithm.
 */
#include <cuda_runtime.h>
#include <math.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <chrono>
#include <numeric>
#include <string>
#include <vector>

#include "gpu_types.h"
#include "gpu_device_api.h"
#include "gpu_qp.cuh"
#include "gpu_noddi.cuh"

#define CHK(expr) do { cudaError_t e_ = (expr); if (e_ != cudaSuccess) { err = cudaGetErrorString(e_); return false; } } while (0)

#include "gpu_record_transfer.cuh"

extern __shared__ __align__(16) unsigned char ukf_smem_raw[];

/* ------------------------------------------------------------------ */
/* typed view of the device buffers                                    */
/* ------------------------------------------------------------------ */
struct DevBufs
{
  double *pos, *state, *cov, *fa, *fa2, *trace, *trace2, *nmse, *m1;
  double *z, *X, *Z, *scratch;
  double *rec_pos, *rec_norm, *rec_state, *rec_cov, *rec_nmse;
  double *rec_trace, *rec_trace2, *rec_fa, *rec_fa2;
  double *gradients, *bvalues, *noddi_table;
  float *dwi;
  void *mask; int mask_type;
  void *stop; int stop_type;
  float *gm; int gm_type;
  float *csf; int csf_type;
  int *active, *reccount, *act_idx, *act_idx_tmp, *n_active_tmp, *invalid;
  GPUDevPointers raw;

  static DevBufs from(const GPUDevPointers& r)
  {
    DevBufs d;
    d.raw = r;
    d.pos = (double*)r.pos; d.state = (double*)r.state; d.cov = (double*)r.cov;
    d.fa = (double*)r.fa; d.fa2 = (double*)r.fa2; d.trace = (double*)r.trace; d.trace2 = (double*)r.trace2;
    d.nmse = (double*)r.nmse; d.m1 = (double*)r.m1;
    d.z = (double*)r.z; d.X = (double*)r.X; d.Z = (double*)r.Z; d.scratch = (double*)r.scratch;
    d.rec_pos = (double*)r.rec_pos; d.rec_norm = (double*)r.rec_norm; d.rec_state = (double*)r.rec_state;
    d.rec_cov = (double*)r.rec_cov; d.rec_nmse = (double*)r.rec_nmse;
    d.rec_trace = (double*)r.rec_trace; d.rec_trace2 = (double*)r.rec_trace2;
    d.rec_fa = (double*)r.rec_fa; d.rec_fa2 = (double*)r.rec_fa2;
    d.gradients = (double*)r.gradients; d.bvalues = (double*)r.bvalues;
    d.noddi_table = (double*)r.noddi_table;
    d.dwi = r.dwi; d.mask = r.mask; d.mask_type = r.mask_type;
    d.stop = r.stop; d.stop_type = r.stop_type;
    d.gm = r.gm; d.gm_type = r.gm_type;
    d.csf = r.csf; d.csf_type = r.csf_type;
    d.active = r.active; d.reccount = r.reccount; d.invalid = r.invalid;
    d.act_idx = r.act_idx; d.act_idx_tmp = r.act_idx_tmp; d.n_active_tmp = r.n_active_tmp;
    return d;
  }
};

/* ------------------------------------------------------------------ */
/* small device helpers — order of operations mirrors the CPU code     */
/* ------------------------------------------------------------------ */

template<typename T>
__device__ __forceinline__ void d_initNormalized(T& x, T& y, T& z)
{
  // linalg.h: initNormalized -> m << a,b,c; m.normalize();
  const T n = sqrt(x * x + y * y + z * z);
  x /= n; y /= n; z /= n;
}

template<typename T>
__device__ __forceinline__ T d_l2fa(T l1, T l2, T l3)
{
  // utilities.cc l2fa
  if (l2 == l3)
    {
    return fabs(l1 - l2) / sqrt(l1 * l1 + T(2.0) * l2 * l2);
    }
  else
    {
    return sqrt(T(0.5) * ((l1 - l2) * (l1 - l2)
                        + (l2 - l3) * (l2 - l3)
                        + (l3 - l1) * (l3 - l1))
                / (l1 * l1 + l2 * l2 + l3 * l3));
    }
}

/* Frobenius norm, accumulated in Eigen column-major storage order. */
template<typename T>
__device__ __forceinline__ T d_frobnorm_colmajor(const T* A, int n)
{
  T s = 0;
  for (int c = 0; c < n; ++c)
    for (int r = 0; r < n; ++r)
      {
      const T v = A[r * n + c];
      s += v * v;
      }
  return sqrt(s);
}

/* In-place lower Cholesky. Each entry receives the same ascending-k
 * updates as the serial right-looking LLT; independent entries run in parallel. */
template<int N, typename T>
__device__ __forceinline__ void d_cholesky_lower(const T* A, T* L, int dimension = N)
{
  const int n = N ? N : dimension;
  for (int e = threadIdx.x; e < n*n; e += blockDim.x)
    {
    L[e] = e/n >= e%n ? A[e] : T(0);
    }
  __syncthreads();
  for (int k = 0; k < n; ++k)
    {
    if (threadIdx.x == 0) { L[k*n+k] = sqrt(L[k*n+k]); }
    __syncthreads();
    for (int i = k+1+threadIdx.x; i < n; i += blockDim.x)
      {
      L[i*n+k] /= L[k*n+k];
      }
    __syncthreads();
    for (int e = threadIdx.x; e < n*n; e += blockDim.x)
      {
      const int i = e/n, j = e%n;
      if (i > k && j > k && i >= j) { L[e] -= L[i*n+k] * L[j*n+k]; }
      }
    __syncthreads();
    }
}

/* Block-parallel LU inverse: identical per-element operation order to a
 * serial LU-with-partial-pivoting inverse (rows-parallel elimination,
 * columns-parallel solves). All threads of the block must call it. */
template<int N, typename T>
__device__ __forceinline__ void d_block_inverse( const T* Ain, T* Ainv,
                                T* lu, int* perm, int* pivslot, int dimension = N)
{
  const int n = N ? N : dimension;
  for (int i = threadIdx.x; i < n*n; i += blockDim.x) { lu[i] = Ain[i]; }
  for (int i = threadIdx.x; i < n; i += blockDim.x) { perm[i] = i; }
  __syncthreads();
  for (int k = 0; k < n; ++k)
    {
    if (threadIdx.x == 0)
      {
      int piv = k;
      T mx = fabs(lu[k * n + k]);
      for (int i = k + 1; i < n; ++i)
        {
        const T v = fabs(lu[i * n + k]);
        if (v > mx)
          {
          mx = v;
          piv = i;
          }
        }
      *pivslot = piv;
      }
    __syncthreads();
    const int piv = *pivslot;
    if (piv != k)
      {
      for (int j = threadIdx.x; j < n; j += blockDim.x)
        {
        const T t = lu[k * n + j];
        lu[k * n + j] = lu[piv * n + j];
        lu[piv * n + j] = t;
        }
      if (threadIdx.x == 0)
        {
        const int t = perm[k];
        perm[k] = perm[piv];
        perm[piv] = t;
        }
      }
    __syncthreads();
    const T d = lu[k * n + k];
    for (int i = k + 1 + threadIdx.x; i < n; i += blockDim.x)
      {
      lu[i*n+k] /= d;
      }
    __syncthreads();
    // Independent trailing entries retain the same ascending-k updates.
    for (int e = threadIdx.x; e < n*n; e += blockDim.x)
      {
      const int i = e/n, j = e%n;
      if (i > k && j > k) { lu[e] -= lu[i*n+k] * lu[k*n+j]; }
      }
    __syncthreads();
    }
  for (int j = threadIdx.x; j < n; j += blockDim.x)
    {
    T yy[N ? N : UKF_GPU_MAX_STATE];
    T xc[N ? N : UKF_GPU_MAX_STATE];
    for (int i = 0; i < n; ++i)
      {
      yy[i] = (perm[i] == j) ? T(1) : T(0);
      }
    for (int i = 1; i < n; ++i)
      {
      T s2 = yy[i];
      for (int k2 = 0; k2 < i; ++k2)
        {
        s2 -= lu[i * n + k2] * yy[k2];
        }
      yy[i] = s2;
      }
    for (int i = n - 1; i >= 0; --i)
      {
      T s2 = yy[i];
      for (int k2 = i + 1; k2 < n; ++k2)
        {
        s2 -= lu[i * n + k2] * xc[k2];
        }
      xc[i] = s2 / lu[i * n + i];
      }
    for (int i = 0; i < n; ++i)
      {
      Ainv[i * n + j] = xc[i];
      }
    }
  __syncthreads();
}

/* diffusion(m, lambdas) — linalg.h; l3 == l2 handled by caller.
 * Replicates Eigen fixed-size evaluation order:
 * T1 = R*Lambda, T2 = T1*R^T (sequential accumulation), scaled by 1e-6. */
template<typename T>
__device__ void d_diffusion(const T m0, const T m1, const T m2,
                            T l0, T l1, T l2, T* D)
{
  const T R[3][3] =
  {
    { m0, m1, m2 },
    { m1, m1 * m1 / (1 + m0) - 1, m1 * m2 / (1 + m0) },
    { m2, m1 * m2 / (1 + m0), m2 * m2 / (1 + m0) - 1 }
  };
  T T1[3][3];
  for (int a = 0; a < 3; ++a)
    {
    T1[a][0] = R[a][0] * l0;
    T1[a][1] = R[a][1] * l1;
    T1[a][2] = R[a][2] * l2;
    }
  for (int a = 0; a < 3; ++a)
    {
    for (int b = 0; b < 3; ++b)
      {
      // Eigen's scaled 3x3 product evaluates each coefficient right-associated.
      const T s = T1[a][0]*R[b][0] + (T1[a][1]*R[b][1] + T1[a][2]*R[b][2]);
      D[a * 3 + b] = s * T(1e-6);   /* GLOBAL_TENSOR_UNPACK_VALUE */
      }
    }
}

/* Per-sigma precompute: normalize dirs, clamp lambdas, build D1 (and D2 for
 * 2T). Same expressions in the same order as the CPU H (which computes the
 * diffusion matrix once per sigma, outside the gradient loop). */
template<typename T>
__device__ void d_H_sigma_prep(int model, const T* st,
                               T* D1, T* D2, T& w1, T& w2)
{
  T m0 = st[0], m1 = st[1], m2 = st[2];
  d_initNormalized(m0, m1, m2);
  const T l1v = st[3] > T(100) ? st[3] : T(100);
  const T l2v = st[4] > T(100) ? st[4] : T(100);
  if (m0 < 0) { m0 = -m0; m1 = -m1; m2 = -m2; }
  d_diffusion(m0, m1, m2, l1v, l2v, l2v, D1);
  w1 = (model == UKF_GPU_MODEL_SIMPLE2T) ? T(0.5) : T(1.0);
  w2 = T(0.5);
  if (model == UKF_GPU_MODEL_SIMPLE2T)
    {
    T n0 = st[5], n1 = st[6], n2 = st[7];
    d_initNormalized(n0, n1, n2);
    const T p1v = st[8] > T(100) ? st[8] : T(100);
    const T p2v = st[9] > T(100) ? st[9] : T(100);
    if (n0 < 0) { n0 = -n0; n1 = -n1; n2 = -n2; }
    d_diffusion(n0, n1, n2, p1v, p2v, p2v, D2);
    }
}

/* signal value for one gradient from a precomputed sigma */
template<typename T>
__device__ T d_H_grad(int model, const T* D1, const T* D2,
                      T w1, T w2, const T* grads, const T* b, int j)
{
  const T u0 = grads[j * 3 + 0];
  const T u1 = grads[j * 3 + 1];
  const T u2 = grads[j * 3 + 2];
  const T bj = b[j];
  const T du0 = D1[0] * u0 + D1[1] * u1 + D1[2] * u2;
  const T du1 = D1[3] * u0 + D1[4] * u1 + D1[5] * u2;
  const T du2 = D1[6] * u0 + (D1[7] * u1 + D1[8] * u2);
  T val = exp(-bj * (u0 * du0 + u1 * du1 + u2 * du2)) * w1;
  if (model == UKF_GPU_MODEL_SIMPLE2T)
    {
    const T ev0 = D2[0] * u0 + D2[1] * u1 + D2[2] * u2;
    const T ev1 = D2[3] * u0 + D2[4] * u1 + D2[5] * u2;
    const T ev2 = D2[6] * u0 + (D2[7] * u1 + D2[8] * u2);
    val += exp(-bj * (u0 * ev0 + u1 * ev1 + u2 * ev2)) * w2;
    }
  return val;
}

/* mask/stop lookups — exact ports of NrrdData::ScalarMaskValue / ScalarStopValue */
__device__ __forceinline__ double d_mask_value(const DevBufs& D, const GPUDarParams& P,
                                               int x, int y, int z)
{
  if ((x < 0 || P.nx <= x) || (y < 0 || P.ny <= y) || (z < 0 || P.nz <= z))
    {
    return 0.0;
    }
  const std::int64_t index = (std::int64_t)P.nz * P.ny * x + (std::int64_t)P.nz * y + z;
  double value;
  switch (D.mask_type)
    {
    case 1:  value = (double)((signed char*)D.mask)[index]; break;
    case 2:  value = (double)((unsigned char*)D.mask)[index]; break;
    case 3:  /* fall through: CPU casts both 3/4 to signed short */
    case 4:  value = (double)((short*)D.mask)[index]; break;
    default: value = 0.0; break;
    }
  return value;
}

__device__ double d_stop_value(const DevBufs& D, const GPUDarParams& P,
                               int x, int y, int z)
{
  double stop = 0.0;
  if ((x < 0 || P.nx <= x) || (y < 0 || P.ny <= y) || (z < 0 || P.nz <= z))
    {
    return 0.0;
    }
  const std::int64_t index = (std::int64_t)P.nz * P.ny * x + (std::int64_t)P.nz * y + z;
  if (D.stop != NULL)
    {
    int stop_value = 0;
    switch (D.stop_type)
      {
      case 2: stop_value = (int)((unsigned char*)D.stop)[index]; break;
      case 3: stop_value = (int)((short*)D.stop)[index]; break;
      case 5: stop_value = (int)((int*)D.stop)[index]; break;
      default: stop_value = 0; break;
      }
    for (int i = 0; i < P.num_stop_labels; ++i)
      {
      if (P.stop_labels[i] == stop_value)
        {
        stop = 1.0;
        }
      }
    }
  else
    {
    if (D.gm != NULL)
      {
      float gm_prob = 0.0f;
      if (D.gm_type == 9 || D.gm_type == 10)
        {
        gm_prob = D.gm[index];
        }
      if ((double)gm_prob > P.gm_thr)
        {
        stop = 1.0;
        }
      }
    if (D.csf != NULL)
      {
      float csf_prob = 0.0f;
      if (D.csf_type == 9 || D.csf_type == 10)
        {
        csf_prob = D.csf[index];
        }
      if ((double)csf_prob > P.csf_thr)
        {
        stop = 1.0;
        }
      }
    }
  return stop;
}

/* curve_radius over the recorded positions (utilities.cc) */
__device__ double d_curve_radius(const DevBufs& D, const GPUDarParams& P, int f)
{
  const int len = D.reccount[f];
  if (len < 3)
    {
    return 1.0;
    }
  const double* p1 = D.rec_pos + ((size_t)f * P.max_rec + (len - 3)) * 3;
  const double* p2 = D.rec_pos + ((size_t)f * P.max_rec + (len - 2)) * 3;
  const double* p3 = D.rec_pos + ((size_t)f * P.max_rec + (len - 1)) * 3;
  double v10 = p2[0] - p1[0], v11 = p2[1] - p1[1], v12 = p2[2] - p1[2];
  double v20 = p3[0] - p2[0], v21 = p3[1] - p2[1], v22 = p3[2] - p2[2];
  d_initNormalized(v10, v11, v12);
  d_initNormalized(v20, v21, v22);
  const double n1 = sqrt(v10 * v10 + v11 * v11 + v12 * v12);
  const double n2 = sqrt(v20 * v20 + v21 * v21 + v22 * v22);
  const double a = (v20 - v10) / (n2 + n1);
  const double b = (v21 - v11) / (n2 + n1);
  const double c = (v22 - v12) / (n2 + n1);
  const double curv = sqrt(a * a + b * b + c * c);
  if (isnan(curv))
    {
    return 0.0;
    }
  return 1.0 / curv;
}

/* ------------------------------------------------------------------ */
/* kernels                                                             */
/* ------------------------------------------------------------------ */

#include "gpu_general.cuh"

/* Record the seed as first fiber point (CPU: Record() before the loop). */
__global__ void k_record_start(GPUDarParams P, DevBufs D, int F)
{
  const int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= F)
    {
    return;
    }
  const int n = P.n;
  const double* st = D.state + (size_t)f * n;
  double* rs = D.rec_state + ((size_t)f * P.max_rec + 0) * n;
  for (int d = 0; d < n; ++d)
    {
    rs[d] = st[d];
    }
  if (n == 5 || n == 10 || n == 15)
    {
    d_initNormalized(rs[0], rs[1], rs[2]);
    if (n == 10) { d_initNormalized(rs[5], rs[6], rs[7]); }
    if (n == 15) { d_initNormalized(rs[10], rs[11], rs[12]); }
    }
  double* rp = D.rec_pos + ((size_t)f * P.max_rec + 0) * 3;
  rp[0] = D.pos[f * 3 + 0]; rp[1] = D.pos[f * 3 + 1]; rp[2] = D.pos[f * 3 + 2];
  D.rec_norm[(size_t)f * P.max_rec + 0] = d_frobnorm_colmajor(D.cov + (size_t)f * n * n, n);
  D.rec_nmse[(size_t)f * P.max_rec + 0] = double(0);
  D.rec_trace[(size_t)f * P.max_rec + 0] = D.trace[f];
  D.rec_trace2[(size_t)f * P.max_rec + 0] = D.trace2[f];
  D.rec_fa[(size_t)f * P.max_rec + 0] = D.fa[f];
  D.rec_fa2[(size_t)f * P.max_rec + 0] = D.fa2[f];
  if (P.record_cov)
    {
    double* rc = D.rec_cov + ((size_t)f * P.max_rec + 0) * n * n;
    const double* cov = D.cov + (size_t)f * n * n;
    for (int i = 0; i < n * n; ++i)
      {
      rc[i] = cov[i];
      }
    }
  D.reccount[f] = 1;
}

/* Interp3Signal: gaussian 3x3x3 gather, per-channel accumulation in voxel
 * order (matches NrrdData.cc). One block per fiber. */
template<typename T>
__global__ void k_interp(GPUDarParams P, DevBufs D, int F)
{
  const int f = D.act_idx[blockIdx.x];
  if (f >= F || D.active[f] == 0)
    {
    return;
    }
  const int G = P.G;
  const T px = D.pos[f * 3 + 0];
  const T py = D.pos[f * 3 + 1];
  const T pz = D.pos[f * 3 + 2];
  T sigma = (T)P.sigma_signal;
  if (sigma == T(0))
    {
    sigma = fmin(fmin((T)P.vx, (T)P.vy), (T)P.vz);
    }
  const int rx = (int)round((T)px);
  const int ry = (int)round((T)py);
  const int rz = (int)round((T)pz);
  double* sig = D.z + (size_t)f * 2 * G;

  for (int q = threadIdx.x; q < G; q += blockDim.x)
    {
    T acc = T(0);
    T w_sum = T(1e-16);
    for (int xx = -1; xx <= 1; ++xx)
      {
      const int x = rx + xx;
      if (x < 0 || P.nx <= x) { continue; }
      const T dx = (x - px) * (T)P.vx;
      const T dxx = dx * dx;
      for (int yy = -1; yy <= 1; ++yy)
        {
        const int y = ry + yy;
        if (y < 0 || P.ny <= y) { continue; }
        const T dy = (y - py) * (T)P.vy;
        const T dyy = dy * dy;
        for (int zz = -1; zz <= 1; ++zz)
          {
          const int z = rz + zz;
          if (z < 0 || P.nz <= z) { continue; }
          const T dz = (z - pz) * (T)P.vz;
          const T dzz = dz * dz;
          const T w = exp(-(dxx + dyy + dzz) / sigma);
          acc += w * (T)D.dwi[P.step1 * x + P.step2 * y + z * G + q];
          w_sum += w;
          }
        }
      }
    sig[q] = acc / w_sum;
    sig[q + G] = sig[q];
    }
}

/* Sigma points + process model F. One block per fiber.
 * Thread 0 computes the Cholesky, all threads spread and transform. */
template<int N>
__global__ void k_sigmas(GPUDarParams P, DevBufs D, int F)
{
  const int f = D.act_idx[blockIdx.x];
  if (f >= F || D.active[f] == 0)
    {
    return;
    }
  const int n = N ? N : P.n;
  const int S = 2 * n + 1;
  double* smem = (double*)ukf_smem_raw;                 /* L[n*n] then x[n] */
  double* L = smem;
  double* xv = smem + n * n;

  d_cholesky_lower<N>(D.cov + (size_t)f * n * n, L, n);
  if (threadIdx.x == 0)
    {
    for (int d = 0; d < n; ++d)
      {
      if (!isfinite(L[d*n+d]) || L[d*n+d] <= 0)
        {
        D.active[f] = 0;
        atomicAdd(D.invalid, 1);
        break;
        }
      }
    const double* st = D.state + (size_t)f * n;
    for (int d = 0; d < n; ++d)
      {
      xv[d] = st[d];
      }
    }
  __syncthreads();

  if (D.active[f] == 0) { return; }

  const double scale = sqrt((double)(n) + double(0.01));     /* sqrt(dim + m_SigmaPointSpread) */
  double* X = D.X + (size_t)f * S * n;
  /* spread: col 0 = x, cols 1..n = x + scale*L, cols n+1..2n = x - scale*L */
  for (int i = threadIdx.x; i < S; i += blockDim.x)
    {
    double* col = X + (size_t)i * n;
    if (i == 0)
      {
      for (int d = 0; d < n; ++d) { col[d] = xv[d]; }
      }
    else
      {
      const double sgn = (i <= n) ? double(1) : double(-1);
      const int c = (i <= n) ? i - 1 : i - n - 1;
      for (int d = 0; d < n; ++d)
        {
        col[d] = xv[d] + sgn * scale * L[d * n + c];
        }
      }
    }
  __syncthreads();

  for (int i = threadIdx.x; i < S; i += blockDim.x)
    {
    double* col = X + (size_t)i * n;
    if (P.model == UKF_GPU_MODEL_GENERAL)
      {
      if (P.num_constraints) { continue; }
      const int width = P.full ? 6 : 5;
      for (int t = 0; t < P.num_tensors; ++t)
        {
        double* tensor = col + t*width;
        if (!P.full)
          {
          double inv = 1/sqrt(tensor[0]*tensor[0]+tensor[1]*tensor[1]+tensor[2]*tensor[2]);
          for (int c = 0; c < 3; ++c) { tensor[c] *= inv; }
          }
        if (!P.noddi)
          {
          for (int c = 3; c < width; ++c)
            {
            if (P.free_water && tensor[c] < -1e-4) { D.active[f] = 0; atomicAdd(D.invalid,1); }
            tensor[c] = fmax(tensor[c], P.free_water ? 0.0 : P.lambda_min);
            }
          }
        }
      if (P.free_water) { col[n-1] = fmax(col[n-1], 0.0); }
      continue;
      }
    double ninv = 1/sqrt(col[0]*col[0]+col[1]*col[1]+col[2]*col[2]);
    col[0] *= ninv; col[1] *= ninv; col[2] *= ninv;
    if (P.model == UKF_GPU_MODEL_SIMPLE2T)
      {
      ninv = 1/sqrt(col[5]*col[5]+col[6]*col[6]+col[7]*col[7]);
      col[5] *= ninv; col[6] *= ninv; col[7] *= ninv;
      col[8] = fmax(col[8], P.lambda_min); col[9] = fmax(col[9], P.lambda_min);
      }
    col[3] = fmax(col[3], P.lambda_min); col[4] = fmax(col[4], P.lambda_min);
    }

}

template<int N>
__global__ void k_constrain_sigmas(GPUDarParams P, DevBufs D, int F, int active)
{
  const int index=blockIdx.x*blockDim.x+threadIdx.x, S=2*N+1;
  if (index>=active*S) { return; }
  const int f=D.act_idx[index/S];
  if (f>=F || !D.active[f]) { return; }
  double* col=D.X+((size_t)f*S+index%S)*N;
  if (!d_constrain<N>(P,col,D.cov+(size_t)f*N*N))
    { atomicExch(D.active+f,0); atomicAdd(D.invalid,1); return; }
  const int width=P.full?6:5;
  for (int t=0;t<P.num_tensors;++t)
    {
    double* tensor=col+t*width;
    if (!P.full)
      { const double inv=1/sqrt(tensor[0]*tensor[0]+tensor[1]*tensor[1]+tensor[2]*tensor[2]); for(int c=0;c<3;++c) { tensor[c]*=inv; } }
    if (!P.noddi)
      {
      for(int c=3;c<width;++c)
        { if (tensor[c]<-1e-4) { atomicExch(D.active+f,0); atomicAdd(D.invalid,1); } tensor[c]=fmax(tensor[c],0.0); }
      }
    }
  if (P.free_water) { col[N-1]=fmax(col[N-1],0.0); }
}

template<int N>
__global__ void k_constrain_state(GPUDarParams P, DevBufs D, int F, int active)
{
  const int index=blockIdx.x*blockDim.x+threadIdx.x;
  if (index>=active) { return; }
  const int f=D.act_idx[index];
  if (f>=F || !D.active[f]) { return; }
  if (!d_constrain<N>(P,D.state+(size_t)f*N,D.scratch+(size_t)f*max(2*N*(2*P.G),N*N)))
    { D.active[f]=0; atomicAdd(D.invalid,1); }
}

/* Signal prediction for all sigma points. One block per fiber;
 * warp-per-sigma: each sigma's diffusion matrix is reconstructed once. */
template<typename T>
__global__ void k_H(GPUDarParams P, DevBufs D, int F)
{
  const int f = D.act_idx[blockIdx.x];
  if (f >= F || D.active[f] == 0)
    {
    return;
    }
  const int n = P.n;
  const int S = 2 * n + 1;
  const int Q2 = 2 * P.G;
  const double* X = D.X + (size_t)f * S * n;
  double* Z = D.Z + (size_t)f * S * Q2;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int nwarps = blockDim.x >> 5;
  T* smemH = (T*)ukf_smem_raw;                /* nwarps * (9 + 9 + 2) */
  T* D1s = smemH + warp * 20;
  T* D2s = D1s + 9;
  T* ws = D2s + 9;
  for (int i = warp; i < S; i += nwarps)
    {
    if (lane == 0)
      {
      T state[UKF_GPU_MAX_STATE];
      for (int d = 0; d < n; ++d) { state[d] = (T)X[(size_t)i * n + d]; }
      d_H_sigma_prep(P.model, state, D1s, D2s, *ws, ws[1]);
      }
    __syncwarp();
    for (int j = lane; j < P.G; j += 32)
      {
      const T grad[3] = {(T)D.gradients[j*3], (T)D.gradients[j*3+1], (T)D.gradients[j*3+2]};
      const T bval[1] = {(T)D.bvalues[j]};
      const double value = d_H_grad(P.model, D1s, D2s, ws[0], ws[1], grad, bval, 0);
      Z[(size_t)i * Q2 + j] = value;
      Z[(size_t)i * Q2 + j + P.G] = value;
      }
    __syncwarp();
    }
}

// The second gradient half repeats the first. Reuse rounded products,
// while retaining every addition in the original order.
template<int G>
__device__ __forceinline__ double d_mirrored_dot(const double* a, const double* b)
{
  double products[G ? G : 1];
  double acc = 0;
#pragma unroll
  for (int q = 0; q < G; ++q) { products[q] = a[q]*b[q]; acc += products[q]; }
#pragma unroll
  for (int q = 0; q < G; ++q) { acc += products[q]; }
  return acc;
}

/* The UKF information-form update. One block per fiber. */
template<int N, int MirrorG = 0>
__global__ void k_update(GPUDarParams P, DevBufs D, int F)
{
  const int f = D.act_idx[blockIdx.x];
  if (f >= F || D.active[f] == 0)
    {
    return;
    }
  const int n = N ? N : P.n;
  const int S = 2 * n + 1;
  const int Q2 = MirrorG ? 2*MirrorG : 2*P.G;
  const int G = MirrorG ? MirrorG : (P.noddi ? Q2 : P.G);
  const int nn = n * n;
  double* smem = (double*)ukf_smem_raw;
  /* shared layout: Xhat[n] | pnew[nn] | Yk[nn] | yhat[n] | Zhat[Q2] | I[nn] |
   *                ivec[n] | v2[Q2] | xnew[n] | Wsh[S] | lu[nn] | weighted_X[S*n] | perm[n+1 ints] */
  double* Xhat = smem;
  double* pnew = Xhat + n;
  double* Yk = pnew + nn;
  double* yhat = Yk + nn;
  double* Zhat = yhat + n;
  double* Imat = Zhat + Q2;
  double* ivec = Imat + nn;
  double* v2 = ivec + n;
  double* xnew = v2 + Q2;
  double* Wsh = xnew + n;
  double* lu = Wsh + S;
  double* weighted_X = lu + nn;
  int* perm = (int*)(weighted_X + S*n);
  double* X = D.X + (size_t)f * S * n;
  double* Z = D.Z + (size_t)f * S * Q2;
  double* Pxz = D.scratch + (size_t)f * max(2*n*Q2,n*n);
  double* Ht = Pxz + n * Q2;

  /* weights */
  if (threadIdx.x == 0)
    {
    Wsh[0] = double(0.01) / (double(n) + double(0.01));
    for (int i = 1; i < S; ++i)
      {
      Wsh[i] = double(0.5) / (double(n) + double(0.01));
      }
    }
  __syncthreads();

  /* X_hat = X * weights */
  for (int d = threadIdx.x; d < n; d += blockDim.x)
    {
    double acc = double(0);
    for (int i = 0; i < S; ++i)
      {
      acc += X[(size_t)i * n + d] * Wsh[i];
      }
    Xhat[d] = acc;
    }
  __syncthreads();

  // These residuals are reused for every covariance/signal column.
  // Store the same rounded intermediates instead of recomputing them.
  for (int e = threadIdx.x; e < S*n; e += blockDim.x)
    {
    X[e] -= Xhat[e % n];
    weighted_X[e] = X[e] * Wsh[e / n];
    }
  __syncthreads();

  /* p_new = (X - Xhat) Wrep (X - Xhat)' + Q ; only diagonal Q terms */
  for (int e = threadIdx.x; e < nn; e += blockDim.x)
    {
    const int d = e / n;
    const int c = e - d * n;
    double acc = double(0);
    for (int i = 0; i < S; ++i)
      {
      acc += weighted_X[(size_t)i * n + d] * X[(size_t)i * n + c];
      }
    double q = double(0);
    if (d == c)
      {
      if (P.model == UKF_GPU_MODEL_SIMPLE1T)
        {
        if (d <= 2) { q = (double)P.Qm; } else { q = (double)P.Ql; }
        }
      else
        {
        if (d <= 2 || (d >= 5 && d <= 7)) { q = (double)P.Qm; } else { q = (double)P.Ql; }
        }
      }
    if (P.has_process_noise) { q = d == c ? P.process_noise[d] : 0; }
    pnew[e] = acc + q;
    }
  __syncthreads();

  /* Yk = p_new^-1 (block-parallel) ; yhat = Yk * X_hat */
  d_block_inverse<N>( pnew, Yk, lu, perm, perm + n, n);
  for (int d = threadIdx.x; d < n; d += blockDim.x)
    {
    double acc = double(0);
    for (int k = 0; k < n; ++k)
      {
      acc += Yk[d * n + k] * Xhat[k];
      }
    yhat[d] = acc;
    }
  __syncthreads();

  /* Z_hat = Z * weights */
  for (int q = threadIdx.x; q < G; q += blockDim.x)
    {
    double acc = double(0);
    for (int i = 0; i < S; ++i)
      {
      acc += Z[(size_t)i * Q2 + q] * Wsh[i];
      }
    Zhat[q] = acc; if (G < Q2) { Zhat[q+G] = acc; }
    }
  __syncthreads();

  for (int e = threadIdx.x; e < S*G; e += blockDim.x)
    {
    const int q = e % G, sigma = e / G;
    const size_t index = (size_t)sigma * Q2 + q;
    const double value = Z[index] - Zhat[q];
    Z[index] = value; if (G < Q2) { Z[index+G] = value; }
    }
  __syncthreads();

  /* Pxz = (X - Xhat) Wrep (Z - Zhat)' */
  {
    const int total = n * G;
    for (int e = threadIdx.x; e < total; e += blockDim.x)
      {
      const int d = e / G;
      const int q = e - d * G;
      double acc = double(0);
      for (int i = 0; i < S; ++i)
        {
        acc += weighted_X[(size_t)i * n + d] * Z[(size_t)i * Q2 + q];
        }
      Pxz[d*Q2+q] = acc; if (G < Q2) { Pxz[d*Q2+q+G] = acc; }
      }
  }
  __syncthreads();

  /* Ht = Yk * Pxz */
  {
    const int total = n * G;
    for (int e = threadIdx.x; e < total; e += blockDim.x)
      {
      const int d = e / G;
      const int q = e - d * G;
      double acc = double(0);
      for (int k = 0; k < n; ++k)
        {
        acc += Yk[d * n + k] * Pxz[k * Q2 + q];
        }
      Ht[d*Q2+q] = acc; if (G < Q2) { Ht[d*Q2+q+G] = acc; }
      }
  }
  __syncthreads();

  /* dNormMSE */
  if (threadIdx.x == 0)
    {
    const double* z = D.z + (size_t)f * Q2;
    double num = double(0), den = double(0);
    for (int q = 0; q < Q2; ++q)
      {
      const double dd = z[q] - Zhat[q];
      num += dd * dd;
      den += z[q] * z[q];
      }
    D.nmse[f] = num / den;
    }

  /* Form the residual before reusing Pxz for the noise-scaled Ht. */
  {
    const double* z = D.z + (size_t)f * Q2;
    for (int q = threadIdx.x; q < G; q += blockDim.x)
      {
      double acc = double(0);
      for (int d = 0; d < n; ++d)
        {
        acc += Pxz[d * Q2 + q] * yhat[d];
        }
      v2[q] = (z[q] - Zhat[q]) + acc; if (G < Q2) { v2[q+G] = v2[q]; }
      }
  }
  __syncthreads();

  // Each rounded Ht*R value is reused by n information-matrix entries
  // and the information vector. Pxz is no longer needed after v2.
  const double R = double(1) / (double)P.Rs;
  for (int e = threadIdx.x; e < n*G; e += blockDim.x)
    {
    const int d = e / G, q = e % G;
    const double value = Ht[d*Q2+q] * R;
    Pxz[d*Q2+q] = value; if (G < Q2) { Pxz[d*Q2+q+G] = value; }
    }
  __syncthreads();

  /* I = (Ht*R) * Ht'; preserve the full Q2 accumulation order. */
  for (int e = threadIdx.x; e < nn; e += blockDim.x)
    {
    const int d = e / n, c = e % n;
    double acc = double(0);
    if (MirrorG) { acc = d_mirrored_dot<MirrorG>(Pxz + d*Q2, Ht + c*Q2); }
    else
      {
      for (int q = 0; q < Q2; ++q) { acc += Pxz[d * Q2 + q] * Ht[c * Q2 + q]; }
      }
    Imat[e] = acc;
    }

  /* i = (Ht*R) * v2 */
  for (int d = threadIdx.x; d < n; d += blockDim.x)
    {
    double acc = double(0);
    if (MirrorG) { acc = d_mirrored_dot<MirrorG>(Pxz + d*Q2, v2); }
    else
      {
      for (int q = 0; q < Q2; ++q) { acc += Pxz[d * Q2 + q] * v2[q]; }
      }
    ivec[d] = acc;
    }
  __syncthreads();

  /* p_new = (Yk + I)^-1 ; x_new = p_new * (i + yhat) */
  for (int e = threadIdx.x; e < nn; e += blockDim.x)
    {
    pnew[e] = Yk[e] + Imat[e];
    }
  __syncthreads();
  d_block_inverse<N>( pnew, Yk, lu, perm, perm + n, n);   /* reuse Yk as p_new */
  for (int d = threadIdx.x; d < n; d += blockDim.x)
    {
    double acc = double(0);
    for (int k = 0; k < n; ++k)
      {
      acc += Yk[d * n + k] * (ivec[k] + yhat[k]);
      }
    xnew[d] = acc;
    }
  __syncthreads();
  if (threadIdx.x == 0)
    {
    if (P.num_constraints)
      { for (int e = 0; e < nn; ++e) { D.scratch[(size_t)f*max(2*n*Q2,n*n)+e] = pnew[e]; } }
    double* st = D.state + (size_t)f * n;
    double* cov = D.cov + (size_t)f * n * n;
    for (int d = 0; d < n; ++d)
      {
      st[d] = xnew[d];
      }
    for (int e = 0; e < nn; ++e)
      {
      cov[e] = Yk[e];
      }
    bool valid = true;
    for (int d = 0; d < n; ++d) { valid = valid && isfinite(st[d]) && cov[d*n+d] > 0; }
    for (int e = 0; e < nn; ++e) { valid = valid && isfinite(cov[e]); }
    if (!valid) { D.active[f] = 0; atomicAdd(D.invalid, 1); }
    }
}

/* mask/stop lookups and the stopping test + record step. One block per fiber. */
__global__ void k_step_stop(GPUDarParams P, DevBufs D, int F, int stepnr)
{
  const int f = D.act_idx[blockIdx.x];
  if (D.active[f] == 0) { return; }
  double* smem = (double*)ukf_smem_raw;   /* single-col signal [2G] | D1[9] | D2[9] | ws[2] */
  double* ws_single = smem + 2 * P.G + 18;
  const int n = P.n;
  const int Q2 = 2 * P.G;
  double* st = D.state + (size_t)f * n;
  double* x = D.pos + f * 3;

  /* ---- State2Tensor + fa/trace + position step (Step1T/Step2T tails) ---- */
  double fa = D.fa[f];
  double fa2 = D.fa2[f];
  if (threadIdx.x == 0)
    {
    if (P.model == UKF_GPU_MODEL_SIMPLE1T)
      {
      double m0 = st[0], m1 = st[1], m2 = st[2];
      d_initNormalized(m0, m1, m2);
      const double l0 = st[3] > double(100) ? st[3] : double(100);
      const double l1 = st[4] > double(100) ? st[4] : double(100);
      const double l2 = l1;
      D.trace[f] = l0 + l1 + l2;
      D.m1[f * 3 + 0] = m0; D.m1[f * 3 + 1] = m1; D.m1[f * 3 + 2] = m2;
      fa = (l0 < l1 || l0 < l2) ? double(0) : d_l2fa(l0, l1, l2);
      }
    else
      {
      /* Step2T: State2Tensor2T + swap heuristics */
      const double od0 = D.m1[f * 3 + 0];
      const double od1 = D.m1[f * 3 + 1];
      const double od2 = D.m1[f * 3 + 2];
      double m10 = st[0], m11 = st[1], m12 = st[2];
      d_initNormalized(m10, m11, m12);
      double m20 = st[5], m21 = st[6], m22 = st[7];
      d_initNormalized(m20, m21, m22);
      double l1_0 = st[3] > double(100) ? st[3] : double(100);
      double l1_1 = st[4] > double(100) ? st[4] : double(100);
      double l1_2 = l1_1;
      double l2_0 = st[8] > double(100) ? st[8] : double(100);
      double l2_1 = st[9] > double(100) ? st[9] : double(100);
      double l2_2 = l2_1;
      if (m10 * od0 + m11 * od1 + m12 * od2 < 0) { m10 = -m10; m11 = -m11; m12 = -m12; }
      if (m20 * od0 + m21 * od1 + m22 * od2 < 0) { m20 = -m20; m21 = -m21; m22 = -m22; }
      const double trace1 = l1_0 + l1_1 + l1_2;
      const double trace2 = l2_0 + l2_1 + l2_2;
      double fa_t1 = d_l2fa(l1_0, l1_1, l1_2);
      double fa_t2 = d_l2fa(l2_0, l2_1, l2_2);
      const double dot12 = m10 * m20 + m11 * m21 + m12 * m22;
      const double dot_clamped = dot12 < double(-1) ? double(-1) : (dot12 > double(1) ? double(1) : dot12);
      const double tensor_angle = acos(dot_clamped) * double(180.0 / 3.141592653589793238462643383279502884);
      double* cov = D.cov + (size_t)f * n * n;
      auto swap_state2t = [&](void)
        {
        const int h = n / 2;
        for (int a = 0; a < h; ++a)
          {
          double ts2 = st[a]; st[a] = st[h + a]; st[h + a] = ts2;
          for (int b = 0; b < h; ++b)
            {
            const double tc = cov[a * n + b];
            cov[a * n + b] = cov[(h + a) * n + (h + b)];
            cov[(h + a) * n + (h + b)] = tc;
            const double td = cov[a * n + h + b];
            cov[a * n + h + b] = cov[(h + a) * n + b];
            cov[(h + a) * n + b] = td;
            }
          }
        };
      if (m10 * od0 + m11 * od1 + m12 * od2 < m20 * od0 + m21 * od1 + m22 * od2)
        {
        double tm = m10; m10 = m20; m20 = tm;
        tm = m11; m11 = m21; m21 = tm;
        tm = m12; m12 = m22; m22 = tm;
        double tl0 = l1_0; l1_0 = l2_0; l2_0 = tl0;
        double tl1 = l1_1; l1_1 = l2_1; l2_1 = tl1;
        double tl2 = l1_2; l1_2 = l2_2; l2_2 = tl2;
        double ts = fa_t1; fa_t1 = fa_t2; fa_t2 = ts;
        D.trace[f] = trace2; D.trace2[f] = trace1;
        swap_state2t();
        }
      else
        {
        D.trace[f] = trace1; D.trace2[f] = trace2;
        }
      if (tensor_angle <= double(20) && fmin(fa_t1, fa_t2) <= double(0.2) && fa_t1 <= double(0.2))
        {
        double tm = m10; m10 = m20; m20 = tm;
        tm = m11; m11 = m21; m21 = tm;
        tm = m12; m12 = m22; m22 = tm;
        double tl0 = l1_0; l1_0 = l2_0; l2_0 = tl0;
        double tl1 = l1_1; l1_1 = l2_1; l2_1 = tl1;
        double tl2 = l1_2; l1_2 = l2_2; l2_2 = tl2;
        swap_state2t();
        }
      D.m1[f * 3 + 0] = m10; D.m1[f * 3 + 1] = m11; D.m1[f * 3 + 2] = m12;
      if (l1_0 < l1_1 || l1_0 < l1_2)
        {
        fa = double(0); fa2 = double(0);
        }
      else
        {
        fa = d_l2fa(l1_0, l1_1, l1_2);
        fa2 = d_l2fa(l2_0, l2_1, l2_2);
        }
      }
    /* position advance: dx << dir2/v0, dir1/v1, dir0/v2 ; x += dx*stepLength */
    const double d0 = D.m1[f * 3 + 2] / (double)P.vx;
    const double d1 = D.m1[f * 3 + 1] / (double)P.vy;
    const double d2 = D.m1[f * 3 + 0] / (double)P.vz;
    x[0] = x[0] + d0 * (double)P.stepLength;
    x[1] = x[1] + d1 * (double)P.stepLength;
    x[2] = x[2] + d2 * (double)P.stepLength;
    D.fa[f] = fa;
    D.fa2[f] = fa2;
    }
  __syncthreads();

  /* ---- stopping checks (Follow*double) ---- */
  const int rx = (int)round((double)x[0]);
  const int ry = (int)round((double)x[1]);
  const int rz = (int)round((double)x[2]);
  const bool is_brain = d_mask_value(D, P, rx, ry, rz) > 0;
  const bool is_stop = d_stop_value(D, P, rx, ry, rz) > 0;

  /* single-column H for the mean signal (diffusion precomputed once) */
  if (threadIdx.x == 0)
    {
    d_H_sigma_prep(P.model, st, smem + Q2, smem + Q2 + 9, ws_single[0], ws_single[1]);
    }
  __syncthreads();
  for (int q = threadIdx.x; q < P.G; q += blockDim.x)
    {
    smem[q] = smem[q + P.G] = d_H_grad(P.model, smem + Q2, smem + Q2 + 9, ws_single[0], ws_single[1],
                       D.gradients, D.bvalues, q);
    }
  __syncthreads();
  double mean_signal = 0;
  if (threadIdx.x == 0)
    {
    for (int q = 0; q < Q2; ++q)
      {
      mean_signal += smem[q];
      }
    mean_signal /= Q2;
    smem[Q2 + 18] = mean_signal;
    }
  __syncthreads();
  mean_signal = smem[Q2 + 18];

  bool in_csf;
  if (P.use_gmcsf)
    {
    in_csf = false;
    }
  else
    {
    in_csf = (mean_signal < (double)P.mean_signal_min) || (fa < (double)P.fa_min);
    }
  const bool is_curving = d_curve_radius(D, P, f) < (double)P.min_radius;

  if (threadIdx.x == 0)
    {
    if (!is_brain || in_csf || is_stop || is_curving)
      {
      D.active[f] = 0;
      return;
      }
    if ((stepnr + 1) % P.steps_per_record == 0)
      {
      const int r = D.reccount[f];
      double* rp = D.rec_pos + ((size_t)f * P.max_rec + r) * 3;
      rp[0] = x[0]; rp[1] = x[1]; rp[2] = x[2];
      D.rec_norm[(size_t)f * P.max_rec + r] = d_frobnorm_colmajor(D.cov + (size_t)f * n * n, n);
      D.rec_nmse[(size_t)f * P.max_rec + r] = D.nmse[f];
      D.rec_trace[(size_t)f * P.max_rec + r] = D.trace[f];
      D.rec_trace2[(size_t)f * P.max_rec + r] = D.trace2[f];
      D.rec_fa[(size_t)f * P.max_rec + r] = D.fa[f];
      D.rec_fa2[(size_t)f * P.max_rec + r] = D.fa2[f];
      double* rs = D.rec_state + ((size_t)f * P.max_rec + r) * n;
      for (int d = 0; d < n; ++d)
        {
        rs[d] = st[d];
        }
      if (P.model == UKF_GPU_MODEL_SIMPLE1T)
        {
        d_initNormalized(rs[0], rs[1], rs[2]);
        }
      else
        {
        d_initNormalized(rs[0], rs[1], rs[2]);
        d_initNormalized(rs[5], rs[6], rs[7]);
        }
      if (P.record_cov)
        {
        double* rc = D.rec_cov + ((size_t)f * P.max_rec + r) * n * n;
        const double* cov = D.cov + (size_t)f * n * n;
        for (int i = 0; i < n * n; ++i)
          {
          rc[i] = cov[i];
          }
        }
      D.reccount[f] = r + 1;
      }
    }
}

/* Compact the active list: keep fibers still marked active. */
__global__ void k_compact(GPUDevPointers D, int n_scan)
{
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n_scan)
    {
    return;
    }
  const int f = D.act_idx[t];
  if (D.active[f] != 0)
    {
    const int pos = atomicAdd(D.n_active_tmp, 1);
    D.act_idx_tmp[pos] = f;
    }
}


/* Own allocations for the whole call, including all error exits. */
struct DeviceBuffers
{
  std::vector<void*> allocations;
  bool allocate(void** dst, size_t bytes, std::string& err)
  {
    const cudaError_t status = cudaMalloc(dst, bytes);
    if (status != cudaSuccess) { err = cudaGetErrorString(status); return false; }
    allocations.push_back(*dst);
    return true;
  }
  ~DeviceBuffers()
  {
    for (auto it = allocations.rbegin(); it != allocations.rend(); ++it) { cudaFree(*it); }
  }
};

bool gpu_available(std::string& info)
{
  int count = 0;
  const cudaError_t status = cudaGetDeviceCount(&count);
  info = status == cudaSuccess ? std::to_string(count) + " CUDA devices" : cudaGetErrorString(status);
  return status == cudaSuccess && count > 0;
}

void cuda_count_devices(int& ndev)
{
  ndev = 0;
  cudaGetDeviceCount(&ndev);
}

void cuda_device_mem_info(int device, size_t& free_b, size_t& total_b)
{
  free_b = total_b = 0;
  if (cudaSetDevice(device) == cudaSuccess) { cudaMemGetInfo(&free_b, &total_b); }
}

bool upload_values(void* dst, const double* src, size_t count, std::string& err)
{
  CHK(cudaMemcpy(dst, src, count * sizeof(double), cudaMemcpyHostToDevice));
  return true;
}

bool download_values(double* dst, const void* src, size_t count, std::string& err)
{
  const size_t piece = 1 << 20;
  for (size_t offset = 0; offset < count; offset += piece)
    {
    const size_t length = std::min(piece, count - offset);
    CHK(cudaMemcpy(dst + offset, static_cast<const double*>(src) + offset,
                   length * sizeof(double), cudaMemcpyDeviceToHost));
    }
  return true;
}

struct WaveProfile
{
  bool enabled = getenv("UKF_GPU_PROFILE") != NULL;
  cudaEvent_t events[6] = {};
  float total[5] = {};
  int samples = 0;
  WaveProfile()
  {
    if (enabled) { for (auto& event : events) { cudaEventCreate(&event); } }
  }
  ~WaveProfile()
  {
    if (enabled)
      {
      fprintf(stderr, "[GPU profile] samples=%d interp=%.1f sigma=%.1f H=%.1f update=%.1f step=%.1f ms\n",
              samples, total[0], total[1], total[2], total[3], total[4]);
      for (auto event : events) { cudaEventDestroy(event); }
      }
  }
};

bool run_waves(const GPUDarParams& P, GPUDevPointers& raw, int F, std::string& err, std::vector<GPUBranchSeed>* branches, int device, int base, int total)
{
  DevBufs D = DevBufs::from(raw);
  const int block = 32;
  const int block_h = P.noddi ? 32 : 256;
  const int n = P.n, nn = n * n, Q2 = 2 * P.G, S2 = 2 * n + 1;
  const size_t smem_sigmas = sizeof(double) * (nn + n);
  const size_t smem_update = sizeof(double) * (4 * nn + 4 * n + 2 * Q2 + S2 + S2*n) + sizeof(int) * (n + 1);
  const size_t smem_step = sizeof(double) * (P.noddi ? 2*Q2+16 : Q2+32);
  const size_t smem_h = P.model == UKF_GPU_MODEL_GENERAL ? sizeof(double)*(block_h/32)*(P.noddi ? Q2+16 : 32) :
    (P.use_float ? sizeof(float) : sizeof(double))*(block_h/32)*20;
  k_record_start<<<(F + block - 1) / block, block>>>(P, D, F);
  CHK(cudaGetLastError());
  WaveProfile profile;
  int active = F;
  const bool report_progress = getenv("UKF_GPU_PROGRESS") != NULL;
  auto last_progress = std::chrono::steady_clock::now();
  for (int step = 1; step <= P.max_length && active > 0; ++step)
    {
    if (P.branching) { CHK(cudaMemset(raw.branch_count,0,sizeof(int))); }
    const bool sample = profile.enabled && (step % 64 == 1);
    if (sample) { cudaEventRecord(profile.events[0]); }
    if (P.use_float) { k_interp<float><<<active, block>>>(P, D, F); }
    else { k_interp<double><<<active, block>>>(P, D, F); }
    CHK(cudaGetLastError());
    if (sample) { cudaEventRecord(profile.events[1]); }
    if (P.model == UKF_GPU_MODEL_GENERAL)
      {
#define SIGMA_CASE(N) case N: k_sigmas<N><<<active,block,smem_sigmas>>>(P,D,F); break
      switch(n) { SIGMA_CASE(5); SIGMA_CASE(6); SIGMA_CASE(7); SIGMA_CASE(10); SIGMA_CASE(11); SIGMA_CASE(12); SIGMA_CASE(13); SIGMA_CASE(15); SIGMA_CASE(18); }
#undef SIGMA_CASE
      }
    else if (n == 5) { k_sigmas<5><<<active, block, smem_sigmas>>>(P, D, F); }
    else { k_sigmas<10><<<active, block, smem_sigmas>>>(P, D, F); }
    CHK(cudaGetLastError());
    if (P.num_constraints)
      {
#define CONSTRAIN_SIGMA_CASE(N) case N: k_constrain_sigmas<N><<<(active*(2*N+1)+31)/32,32>>>(P,D,F,active); break
      switch(n) { CONSTRAIN_SIGMA_CASE(6); CONSTRAIN_SIGMA_CASE(7); CONSTRAIN_SIGMA_CASE(11); CONSTRAIN_SIGMA_CASE(13); }
#undef CONSTRAIN_SIGMA_CASE
      CHK(cudaGetLastError());
      }
    if (sample) { cudaEventRecord(profile.events[2]); }
    if (P.model == UKF_GPU_MODEL_GENERAL)
      {
      if (P.use_float && !P.noddi) { k_H_general<float><<<active, block_h, smem_h>>>(P,D,F); }
      else { k_H_general<double><<<active, block_h, smem_h>>>(P,D,F); }
      }
    else if (P.use_float) { k_H<float><<<active, block_h, smem_h>>>(P, D, F); }
    else { k_H<double><<<active, block_h, smem_h>>>(P, D, F); }
    CHK(cudaGetLastError());
    if (sample) { cudaEventRecord(profile.events[3]); }
    if (P.model == UKF_GPU_MODEL_GENERAL)
      {
#define UPDATE_CASE(N) case N: k_update<N><<<active,block,smem_update>>>(P,D,F); break
      switch(n) { UPDATE_CASE(5); UPDATE_CASE(6); UPDATE_CASE(7); UPDATE_CASE(10); UPDATE_CASE(11); UPDATE_CASE(12); UPDATE_CASE(13); UPDATE_CASE(15); UPDATE_CASE(18); }
#undef UPDATE_CASE
      }
    else if (n == 5) { k_update<5><<<active, block, smem_update>>>(P, D, F); }
    else if (P.G == 51) { k_update<10,51><<<active, block, smem_update>>>(P,D,F); }
    else { k_update<10><<<active, block, smem_update>>>(P, D, F); }
    CHK(cudaGetLastError());
    if (P.num_constraints)
      {
#define CONSTRAIN_STATE_CASE(N) case N: k_constrain_state<N><<<(active+31)/32,32>>>(P,D,F,active); break
      switch(n) { CONSTRAIN_STATE_CASE(6); CONSTRAIN_STATE_CASE(7); CONSTRAIN_STATE_CASE(11); CONSTRAIN_STATE_CASE(13); }
#undef CONSTRAIN_STATE_CASE
      CHK(cudaGetLastError());
      }
    if (sample) { cudaEventRecord(profile.events[4]); }
    if (P.model == UKF_GPU_MODEL_GENERAL || P.branching) { k_step_general<<<active, block, smem_step>>>(P,D,F,step); }
    else { k_step_stop<<<active, block, smem_step>>>(P, D, F, step); }
    CHK(cudaGetLastError());
    if (sample) { cudaEventRecord(profile.events[5]); }
    CHK(cudaMemset(D.n_active_tmp, 0, sizeof(int)));
    k_compact<<<(active + block - 1) / block, block>>>(raw, active);
    CHK(cudaGetLastError());
    CHK(cudaMemcpy(&active, D.n_active_tmp, sizeof(int), cudaMemcpyDeviceToHost));
    if (report_progress)
      {
      const auto now = std::chrono::steady_clock::now();
      if (step == 0 || active == 0 || std::chrono::duration<double>(now-last_progress).count() >= 0.25)
        {
        fprintf(stderr, "[UKF_PROGRESS] {\"device\":%d,\"completed\":%d,\"total\":%d}\n",
                device, base+F-active, total);
        last_progress = now;
        }
      }
    if (P.branching)
      {
      int count = 0;
      CHK(cudaMemcpy(&count,raw.branch_count,sizeof(int),cudaMemcpyDeviceToHost));
      const size_t offset = branches->size(); branches->resize(offset+count);
      if (count) { CHK(cudaMemcpy(branches->data()+offset,raw.branches,count*sizeof(GPUBranchSeed),cudaMemcpyDeviceToHost)); }
      }
    if (sample)
      {
      for (int i = 0; i < 5; ++i)
        {
        float ms = 0;
        CHK(cudaEventElapsedTime(&ms, profile.events[i], profile.events[i+1]));
        profile.total[i] += ms;
        }
      ++profile.samples;
      }
    std::swap(raw.act_idx, raw.act_idx_tmp);
    D.act_idx = raw.act_idx;
    D.act_idx_tmp = raw.act_idx_tmp;
    }
  return true;
}

bool gpu_compute(const GPUDarParams& P,
                 const float* dwi, const void* mask, const void* stop,
                 const float* gm, const float* csf,
                 const double* gradients, const double* bvalues, std::size_t dwi_elems,
                 const double* seed_pos, const double* seed_state,
                 const double* seed_cov, const double* seed_fa,
                 const double* seed_fa2, const double* seed_trace,
                 const double* seed_trace2, const double* seed_dir, int S,
                 int* out_reccount,
                 double* out_rec_pos, double* out_rec_norm, double* out_rec_state,
                 double* out_rec_cov, double* out_rec_nmse,
                 double* out_rec_trace, double* out_rec_trace2,
                 double* out_rec_fa, double* out_rec_fa2,
                 int device, int chunk_size, std::string& err, const double* noddi_table, std::vector<GPUBranchSeed>* branches)
{
  CHK(cudaSetDevice(device));
  if (S == 0) { return true; }
  DeviceBuffers storage;
  GPUDevPointers D = {};
  const int n = P.n, Q2 = 2 * P.G, S2 = 2 * n + 1;
  const size_t voxels = (size_t)P.nx * P.ny * P.nz;
#define ALLOC(field, bytes) if (!storage.allocate(reinterpret_cast<void**>(&D.field), (bytes), err)) { return false; }
#define UPLOAD(field, src, count) if (!upload_values(D.field, (src), (count), err)) { return false; }
  if (P.noddi)
    {
    ALLOC(noddi_table, sizeof(double)*P.G*10);
    UPLOAD(noddi_table, noddi_table, P.G*10);
    }
  ALLOC(gradients, sizeof(double) * Q2 * 3);
  UPLOAD(gradients, gradients, Q2 * 3);
  ALLOC(bvalues, sizeof(double) * Q2);
  UPLOAD(bvalues, bvalues, Q2);
  ALLOC(dwi, sizeof(float) * dwi_elems);
  CHK(cudaMemcpy(D.dwi, dwi, sizeof(float) * dwi_elems, cudaMemcpyHostToDevice));
  D.mask_type = P.mask_type; D.stop_type = P.stop_type;
  D.gm_type = P.gm_type; D.csf_type = P.csf_type;
  if (mask)
    {
    const size_t bytes = voxels * ((P.mask_type == 3 || P.mask_type == 4) ? 2 : 1);
    ALLOC(mask, bytes);
    CHK(cudaMemcpy(D.mask, mask, bytes, cudaMemcpyHostToDevice));
    }
  if (stop)
    {
    const size_t bytes = voxels * (P.stop_type == 5 ? 4 : (P.stop_type == 3 ? 2 : 1));
    ALLOC(stop, bytes);
    CHK(cudaMemcpy(D.stop, stop, bytes, cudaMemcpyHostToDevice));
    }
  if (gm)
    {
    ALLOC(gm, sizeof(float) * voxels);
    CHK(cudaMemcpy(D.gm, gm, sizeof(float) * voxels, cudaMemcpyHostToDevice));
    }
  if (csf)
    {
    ALLOC(csf, sizeof(float) * voxels);
    CHK(cudaMemcpy(D.csf, csf, sizeof(float) * voxels, cudaMemcpyHostToDevice));
    }

  size_t free_b = 0, total_b = 0;
  CHK(cudaMemGetInfo(&free_b, &total_b));
  const size_t transfer_values = 1 << 20; // At most 8 MiB of temporary device/host storage.
  const size_t transfer_bytes = transfer_values*sizeof(double);
  const size_t per_fiber = sizeof(double) * (size_t)(3 + n + n*n + 6 + 3 + Q2 + S2*n + S2*Q2 + std::max(2*n*Q2,n*n)
    + P.max_rec * (3 + 1 + n + 5 + (P.record_cov ? n*n : 0))) + 4 * sizeof(int) + sizeof(size_t) + (P.branching ? 2*sizeof(GPUBranchSeed) : 0);
  const size_t capacity = (size_t)((free_b > transfer_bytes ? free_b-transfer_bytes : 0) * 0.85) / per_fiber;
  if (capacity == 0) { err = "insufficient device memory for one fiber"; return false; }
  size_t want = chunk_size > 0 ? (size_t)chunk_size : (size_t)S;
  const char* env_chunk = getenv("UKF_GPU_CHUNK");
  if (env_chunk)
    {
    const long value = strtol(env_chunk, NULL, 10);
    if (value <= 0) { err = "UKF_GPU_CHUNK must be positive"; return false; }
    want = (size_t)value;
    }
  const int CF = (int)std::min(std::min(want, capacity), std::min((size_t)S, (size_t)131072));
  fprintf(stderr, "[GPU worker] dev=%d precision=%s slice_F=%d CF=%d per_fiber=%.0fKB\n",
          device, P.use_float ? "mixed-fp32" : "fp64", S, CF, per_fiber / 1024.0);
  if (P.branching) { ALLOC(branch_count,sizeof(int)); ALLOC(branches,2*CF*sizeof(GPUBranchSeed)); }
  ALLOC(pos, sizeof(double)*3*CF);
  ALLOC(state, sizeof(double)*n*CF);
  ALLOC(cov, sizeof(double)*n*n*CF);
  ALLOC(fa, sizeof(double)*CF); ALLOC(fa2, sizeof(double)*CF);
  ALLOC(trace, sizeof(double)*CF); ALLOC(trace2, sizeof(double)*CF);
  ALLOC(nmse, sizeof(double)*CF); ALLOC(m1, sizeof(double)*3*CF);
  ALLOC(active, sizeof(int)*CF); ALLOC(reccount, sizeof(int)*CF);
  ALLOC(act_idx, sizeof(int)*CF); ALLOC(act_idx_tmp, sizeof(int)*CF);
  ALLOC(n_active_tmp, sizeof(int)); ALLOC(invalid, sizeof(int));
  ALLOC(z, sizeof(double)*Q2*CF); ALLOC(X, sizeof(double)*S2*n*CF);
  ALLOC(Z, sizeof(double)*S2*Q2*CF); ALLOC(scratch, sizeof(double)*std::max(2*n*Q2,n*n)*CF);
  const size_t records = (size_t)CF * P.max_rec;
  ALLOC(rec_pos, sizeof(double)*records*3); ALLOC(rec_norm, sizeof(double)*records);
  ALLOC(rec_state, sizeof(double)*records*n); ALLOC(rec_nmse, sizeof(double)*records);
  ALLOC(rec_trace, sizeof(double)*records); ALLOC(rec_trace2, sizeof(double)*records);
  ALLOC(rec_fa, sizeof(double)*records); ALLOC(rec_fa2, sizeof(double)*records);
  if (P.record_cov) { ALLOC(rec_cov, sizeof(double)*records*n*n); }
  size_t* device_offsets = NULL;
  double* transfer_buffer = NULL;
  const size_t transfer_capacity = std::min(transfer_values, records*(P.record_cov ? n*n : n));
  if (!storage.allocate(reinterpret_cast<void**>(&device_offsets),sizeof(size_t)*(CF+1),err) ||
      !storage.allocate(reinterpret_cast<void**>(&transfer_buffer),sizeof(double)*transfer_capacity,err)) { return false; }
  std::vector<double> transfer_host(transfer_capacity);
  std::vector<int> indices(CF), ones(CF, 1);
  std::iota(indices.begin(), indices.end(), 0);
  for (int base = 0; base < S; base += CF)
    {
    const int F = std::min(S - base, CF);
    UPLOAD(pos, seed_pos + (size_t)base*3, (size_t)F*3);
    UPLOAD(state, seed_state + (size_t)base*n, (size_t)F*n);
    UPLOAD(cov, seed_cov + (size_t)base*n*n, (size_t)F*n*n);
    UPLOAD(fa, seed_fa + base, F); UPLOAD(fa2, seed_fa2 + base, F);
    UPLOAD(trace, seed_trace + base, F); UPLOAD(trace2, seed_trace2 + base, F);
    UPLOAD(m1, seed_dir + (size_t)base*3, (size_t)F*3);
    CHK(cudaMemcpy(D.active, ones.data(), sizeof(int)*F, cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(D.act_idx, indices.data(), sizeof(int)*F, cudaMemcpyHostToDevice));
    const auto wave_start = std::chrono::steady_clock::now();
    CHK(cudaMemset(D.invalid, 0, sizeof(int)));
    const size_t first_branch = branches ? branches->size() : 0;
    if (!run_waves(P, D, F, err, branches, device, base, S)) { return false; }
    if (branches) { for (size_t i = first_branch; i < branches->size(); ++i) { (*branches)[i].fiber += base; } }
    int invalid = 0;
    CHK(cudaMemcpy(&invalid, D.invalid, sizeof(int), cudaMemcpyDeviceToHost));
    if (invalid) { err = "GPU UKF encountered invalid state/covariance in " + std::to_string(invalid) + " fibers"; return false; }
    CHK(cudaMemcpy(out_reccount + base, D.reccount, sizeof(int)*F, cudaMemcpyDeviceToHost));
    const auto download_start = std::chrono::steady_clock::now();
    // Only F rows exist in the host tail, even when the device capacity is CF.
    const size_t count = (size_t)F * P.max_rec, offset = (size_t)base * P.max_rec;
    std::vector<size_t> offsets(F+1,0);
    for (int f = 0; f < F; ++f) { offsets[f+1] = offsets[f]+out_reccount[base+f]; }
    const bool compact = offsets.back()*2 < count;
    if (compact) { CHK(cudaMemcpy(device_offsets,offsets.data(),offsets.size()*sizeof(size_t),cudaMemcpyHostToDevice)); }
    size_t transferred_values = 0;
    auto download = [&](double* output,const void* source,int width)
      {
      transferred_values += (compact ? offsets.back() : count)*width;
      if (compact) { return download_compact_records(output+offset*width,source,P.max_rec,width,
        offsets,device_offsets,transfer_buffer,transfer_host,err); }
      return download_values(output+offset*width,source,count*width,err);
      };
#define DOWNLOAD(field, output, width) if (!download((output),D.field,(width))) { return false; }
    DOWNLOAD(rec_pos, out_rec_pos, 3); DOWNLOAD(rec_norm, out_rec_norm, 1);
    DOWNLOAD(rec_state, out_rec_state, n);
    if (P.record_nmse) { DOWNLOAD(rec_nmse, out_rec_nmse, 1); }
    if (P.record_trace) { DOWNLOAD(rec_trace, out_rec_trace, 1); DOWNLOAD(rec_trace2, out_rec_trace2, 1); }
    if (P.record_fa) { DOWNLOAD(rec_fa, out_rec_fa, 1); DOWNLOAD(rec_fa2, out_rec_fa2, 1); }
    if (P.record_cov) { DOWNLOAD(rec_cov, out_rec_cov, n*n); }
#undef DOWNLOAD
    if (getenv("UKF_GPU_PROGRESS"))
      { fprintf(stderr, "[UKF_PROGRESS] {\"device\":%d,\"completed\":%d,\"total\":%d}\n", device, base+F, S); }
    if (getenv("UKF_GPU_PROFILE"))
      {
      const auto finish = std::chrono::steady_clock::now();
      fprintf(stderr, "[GPU transfer] mode=%s records=%zu/%zu downloaded=%.1f MiB\n",
        compact ? "compact" : "dense",offsets.back(),count,transferred_values*sizeof(double)/(1024.0*1024));
      fprintf(stderr, "[GPU phase] waves=%.3f download=%.3f s\n",
        std::chrono::duration<double>(download_start-wave_start).count(),
        std::chrono::duration<double>(finish-download_start).count());
      }
    }
#undef UPLOAD
#undef ALLOC
  return true;
}

