// CUDA implementations retaining Eigen 3.4's floating-point evaluation order.
// Copyright (C) 2007 Julien Pommier
// Copyright (C) 2014 Pedro Gonnet
// Copyright (C) 2009-2019 Gael Guennebaud
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. See third-party/Eigen-MPL2.txt or https://mozilla.org/MPL/2.0/.
#ifndef UKF_GPU_EIGEN_MATH_CUH
#define UKF_GPU_EIGEN_MATH_CUH
#include "gpu_libm_math.cuh"

__device__ inline double d_eigen_madd(double a, double b, double c, int fused)
{
  return fused ? fma(a,b,c) : a*b+c;
}

__device__ inline double d_eigen_exp(double input, int fused)
{
  if (isnan(input)) { return input; }
  double x = fmax(fmin(input,709.784),-709.784);
  const double exponent = floor(d_eigen_madd(1.4426950408889634073599,x,0.5,fused));
  x -= exponent*0.693145751953125;
  x -= exponent*1.42860682030941723212e-6;
  const double x2 = x*x;
  double numerator = d_eigen_madd(1.26177193074810590878e-4,x2,3.02994407707441961300e-2,fused);
  numerator = d_eigen_madd(numerator,x2,9.99999999999999999910e-1,fused)*x;
  double denominator = d_eigen_madd(3.00198505138664455042e-6,x2,2.52448340349684104192e-3,fused);
  denominator = d_eigen_madd(denominator,x2,2.27265548208155028766e-1,fused);
  denominator = d_eigen_madd(denominator,x2,2.00000000000000000009e0,fused);
  x = numerator/(denominator-numerator);
  x = d_eigen_madd(2.0,x,1.0,fused);
  return fmax(ldexp(x,(int)exponent),input);
}

template<typename T>
__device__ inline T d_host_exp(int policy, T value)
{
  return sizeof(T) == sizeof(double) && policy ? (T)d_libm_exp(value,policy-1) : exp(value);
}

__device__ inline double d_full2fw_exp(const GPUDarParams& P, double value, int column, int row)
{
  const int size = 6*P.G;
  return P.eigen_exp_packet > 1 && column*2*P.G+row < size/P.eigen_exp_packet*P.eigen_exp_packet
    ? d_eigen_exp(value,P.eigen_fma) : (P.libm_exp ? d_libm_exp(value,P.libm_exp-1) : exp(value));
}

// Eigen's full row tiles accumulate in order. Single packet row tiles split
// eight inner products into even/odd accumulators; scalar AVX tails use four.
__device__ inline int d_eigen_gemm_order(const GPUDarParams& P, int row, int rows,
                                        int column, int columns, int depth)
{
  const int packet = P.eigen_packet;
  if (packet <= 1 || rows+columns+depth < 20 || column >= columns/4*4) { return 0; }
  const int three = P.eigen_gemm_mr >= 3*packet ? rows/(3*packet)*(3*packet) : 0;
  const int two = P.eigen_gemm_mr >= 2*packet ? three+(rows-three)/(2*packet)*(2*packet) : three;
  if (row < two) { return 0; }
  const int full = two+(rows-two)/packet*packet;
  const int half = packet >= 4 ? full+(rows-full)/(packet/2)*(packet/2) : full;
  if (row < half) { return 1; }
  return packet%4 == 0 ? 2 : 0;
}

__device__ inline double d_eigen_dot(const GPUDarParams& P, const double* a, int a_stride,
                                    const double* b, int b_stride, int count, int order)
{
  double sum = 0;
  if (order == 1)
    {
    double even = 0, odd = 0;
    const int end = count/8*8;
    for (int k = 0; k < end; k += 8)
      {
      for (int j = 0; j < 8; ++j)
        {
        double& value = j%2 ? odd : even;
        value = d_eigen_madd(a[(k+j)*a_stride],b[(k+j)*b_stride],value,P.eigen_fma);
        }
      }
    sum = even+odd;
    for (int k = end; k < count; ++k) { sum = d_eigen_madd(a[k*a_stride],b[k*b_stride],sum,P.eigen_fma); }
    return sum;
    }
  if (order == 2)
    {
    double partial[4][2] = {};
    const int width = P.eigen_packet/4;
    const int end = count/(4*width)*(4*width);
    for (int k = 0; k < end; k += 4*width)
      {
      for (int j = 0; j < 4; ++j)
        for (int lane = 0; lane < width; ++lane)
          {
          const int index = k+j*width+lane;
          partial[j][lane] = d_eigen_madd(a[index*a_stride],b[index*b_stride],partial[j][lane],P.eigen_fma);
          }
      }
    double lanes[2] = {};
    for (int lane = 0; lane < width; ++lane)
      { lanes[lane] = (partial[0][lane]+partial[1][lane])+(partial[2][lane]+partial[3][lane]); }
    int k = end;
    for (; k+width <= count; k += width)
      for (int lane = 0; lane < width; ++lane)
        { lanes[lane] = d_eigen_madd(a[(k+lane)*a_stride],b[(k+lane)*b_stride],lanes[lane],P.eigen_fma); }
    sum = width == 2 ? lanes[0]+lanes[1] : lanes[0];
    for (; k < count; ++k) { sum += a[k*a_stride]*b[k*b_stride]; }
    return sum;
    }
  for (int k = 0; k < count; ++k) { sum = d_eigen_madd(a[k*a_stride],b[k*b_stride],sum,P.eigen_fma); }
  return sum;
}

__device__ inline double d_eigen_vector_dot(const GPUDarParams& P, const double* a, int a_stride,
                                           const double* b, int b_stride, int count, bool fractional_tail = false)
{
  if (P.eigen_packet <= 1) { return d_eigen_dot(P,a,a_stride,b,b_stride,count,0); }
  double lanes[8] = {};
  const int packet = P.eigen_packet, end = count/packet*packet;
  for (int k = 0; k < end; k += packet)
    for (int lane = 0; lane < packet; ++lane)
      { lanes[lane] = d_eigen_madd(a[(k+lane)*a_stride],b[(k+lane)*b_stride],lanes[lane],P.eigen_fma); }
  for (int width = packet/2; width; width /= 2)
    for (int lane = 0; lane < width; ++lane) { lanes[lane] += lanes[lane+width]; }
  double sum = lanes[0];
  int k = end;
  if (fractional_tail && packet >= 4 && k+packet/2 <= count)
    {
    const int half = packet/2;
    for (int lane = 0; lane < half; ++lane) { lanes[lane] = a[(k+lane)*a_stride]*b[(k+lane)*b_stride]; }
    for (int width = half/2; width; width /= 2)
      for (int lane = 0; lane < width; ++lane) { lanes[lane] += lanes[lane+width]; }
    sum += lanes[0];
    k += half;
    }
  for (; k < count; ++k) { sum += a[k*a_stride]*b[k*b_stride]; }
  return sum;
}

__device__ inline void d_eigen_solve_lower(const GPUDarParams& P, const double* matrix, int stride,
                                          int offset, int n, double* x, int column, int columns)
{
  const int panel = P.eigen_solve_panel > 0 ? P.eigen_solve_panel : n;
  for (int start = 0; start < n; start += panel)
    {
    const int end = min(n,start+panel);
    for (int i = start; i < end; ++i)
      for (int r = i+1; r < end; ++r) { x[r] -= x[i]*matrix[(offset+r)*stride+offset+i]; }
    for (int r = end; r < n; ++r)
      {
      const int order = d_eigen_gemm_order(P,r-end,n-end,column,columns,end-start);
      x[r] -= d_eigen_dot(P,matrix+(offset+r)*stride+offset+start,1,x+start,1,end-start,order);
      }
    }
}

__device__ inline void d_eigen_solve_upper(const GPUDarParams& P, const double* matrix,
                                          int n, double* x, int column, int columns)
{
  const int panel = P.eigen_solve_panel > 0 ? P.eigen_solve_panel : n;
  for (int end = n; end > 0; end -= panel)
    {
    const int start = max(0,end-panel);
    for (int i = end-1; i >= start; --i)
      {
      x[i] *= 1.0/matrix[i*n+i];
      for (int r = start; r < i; ++r) { x[r] -= x[i]*matrix[r*n+i]; }
      }
    for (int r = 0; r < start; ++r)
      {
      const int order = d_eigen_gemm_order(P,r,start,column,columns,end-start);
      x[r] -= d_eigen_dot(P,matrix+r*n+start,1,x+start,1,end-start,order);
      }
    }
}

// PartialPivLU uses 8-column panels above dimension 16. Independent inverse
// columns run in parallel, retaining Eigen's triangular solve/update ordering.
template<int N>
__device__ inline void d_block_inverse(const double* input, double* inverse, double* lu,
                                       int* permutation, int* pivot, const GPUDarParams& P, int dimension = N)
{
  const int n = N ? N : dimension, panel = n <= 16 ? n : 8;
  for (int e = threadIdx.x; e < n*n; e += blockDim.x) { lu[e] = input[e]; }
  for (int i = threadIdx.x; i < n; i += blockDim.x) { permutation[i] = i; }
  __syncthreads();
  for (int base = 0; base < n; base += panel)
    {
    const int end = min(n,base+panel);
    for (int k = base; k < end; ++k)
      {
      if (threadIdx.x == 0)
        {
        *pivot = k;
        double best = fabs(lu[k*n+k]);
        for (int r = k+1; r < n; ++r)
          { const double value = fabs(lu[r*n+k]); if (value > best) { best = value; *pivot = r; } }
        }
      __syncthreads();
      const int row = *pivot;
      if (row != k)
        {
        for (int c = threadIdx.x; c < n; c += blockDim.x)
          { const double value = lu[k*n+c]; lu[k*n+c] = lu[row*n+c]; lu[row*n+c] = value; }
        if (threadIdx.x == 0)
          { const int value = permutation[k]; permutation[k] = permutation[row]; permutation[row] = value; }
        }
      __syncthreads();
      const double diagonal = lu[k*n+k];
      if (diagonal != 0)
        for (int r = k+1+threadIdx.x; r < n; r += blockDim.x) { lu[r*n+k] /= diagonal; }
      __syncthreads();
      for (int e = threadIdx.x; e < n*n; e += blockDim.x)
        {
        const int r = e/n, c = e%n;
        if (r > k && c > k && c < end) { lu[e] -= lu[r*n+k]*lu[k*n+c]; }
        }
      __syncthreads();
      }
    if (end == n) { continue; }
    for (int c = end+threadIdx.x; c < n; c += blockDim.x)
      {
      double x[N ? N : UKF_GPU_MAX_STATE];
      for (int r = base; r < end; ++r) { x[r-base] = lu[r*n+c]; }
      d_eigen_solve_lower(P,lu,n,base,end-base,x,c-end,n-end);
      for (int r = base; r < end; ++r) { lu[r*n+c] = x[r-base]; }
      }
    __syncthreads();
    for (int e = threadIdx.x; e < n*n; e += blockDim.x)
      {
      const int r = e/n, c = e%n;
      if (r >= end && c >= end)
        {
        const int order = d_eigen_gemm_order(P,r-end,n-end,c-end,n-end,end-base);
        lu[e] -= d_eigen_dot(P,lu+r*n+base,1,lu+base*n+c,n,end-base,order);
        }
      }
    __syncthreads();
    }
  for (int c = threadIdx.x; c < n; c += blockDim.x)
    {
    double x[N ? N : UKF_GPU_MAX_STATE];
    for (int r = 0; r < n; ++r) { x[r] = permutation[r] == c ? 1.0 : 0.0; }
    d_eigen_solve_lower(P,lu,n,0,n,x,c,n);
    d_eigen_solve_upper(P,lu,n,x,c,n);
    for (int r = 0; r < n; ++r) { inverse[r*n+c] = x[r]; }
    }
  __syncthreads();
}
#endif
