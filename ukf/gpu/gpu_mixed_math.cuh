// Guarded FP32 range selection for the GNU libc exp evaluation.
// The FP64 reduction, polynomial, and scaling follow gpu_libm_math.cuh.
// Copyright (C) 2001-2020 Free Software Foundation, Inc.
// LGPL-2.1-or-later. See third-party/glibc-LGPL2.1.txt.
#ifndef UKF_GPU_MIXED_MATH_CUH
#define UKF_GPU_MIXED_MATH_CUH
#include "gpu_eigen_math.cuh"

__device__ __forceinline__ double d_mixed_exp(double x, int fused)
{
  if (!(fabs(x) < 512) || fabs(x) < 0x1p-54) { return d_libm_exp(x, fused); }

  // Cast/product error is < 3.01 FP32 unit roundoffs. Use four, and
  // recompute with the original FP64 expression near a half-integer.
  // This selects the same integer; no approximate signal enters the UKF.
  const float q = __fmul_rn(__double2float_rn(x), float(0x1.71547652b82fep7));
  const int nearest = __float2int_rn(q);
  double kd = double(nearest);
  unsigned long long ki = 0x4338000000000000ULL + (long long)nearest;
  if (fabsf(q - float(nearest)) >= 0.5f - fabsf(q) * 0x1p-22f)
    {
    kd = d_libm_madd(0x1.71547652b82fep7, x, 0x1.8p52, fused);
    ki = d_libm_bits(kd);
    kd -= 0x1.8p52;
    }
  double r = d_libm_madd(kd, -0x1.62e42fefa0000p-8, x, fused);
  r = d_libm_madd(kd, -0x1.cf79abc9e3b3ap-47, r, fused);
  const int idx = 2 * (ki % 128);
  const double tail = d_libm_value(d_libm_exp_table[idx]);
  const double scale = d_libm_value(d_libm_exp_table[idx + 1] + (ki << 45));
  const double r2 = r * r;
  double t = d_libm_madd(r, 0x1.555555555543cp-3, 0x1.ffffffffffdbdp-2, fused);
  t = d_libm_madd(r2, t, tail + r, fused);
  t = d_libm_madd(r2 * r2,
      d_libm_madd(r, 0x1.1111167a4d017p-7, 0x1.55555cf172b91p-5, fused), t, fused);
  return d_libm_madd(scale, t, scale, fused);
}

__device__ __forceinline__ double d_signal_exp(int policy, bool mixed, double x)
{
  return mixed && policy ? d_mixed_exp(x, policy - 1) : d_host_exp(policy, x);
}
#endif
