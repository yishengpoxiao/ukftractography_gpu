// Compare the CUDA Goldfarb-Idnani projection with CPU and check box KKT conditions.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
#include "gpu_types.h"
#include "gpu_qp.cuh"
#include "QuadProg++_Eigen.h"

__global__ void project(GPUDarParams p, double* x, const double* w, int* ok, int count)
{
  const int f = blockIdx.x*blockDim.x+threadIdx.x;
  if (f < count)
    {
    switch(p.n)
      {
#define PROJECT_CASE(N) case N: ok[f]=d_constrain<N>(p,x+(size_t)f*N,w+(size_t)f*N*N); break
      PROJECT_CASE(6); PROJECT_CASE(7); PROJECT_CASE(11); PROJECT_CASE(13);
#undef PROJECT_CASE
      }
    }
}

int main()
{
  std::mt19937 random(718);
  std::normal_distribution<double> normal(0,1);
  const int count = 128;
  for (int n : {6,7,11,13})
    {
    GPUDarParams p = {};
    p.n = n; p.num_constraints = 4;
    p.constraint_index[0] = p.constraint_index[1] = n-1;
    p.constraint_sign[0] = -1; p.constraint_offset[0] = 1;
    p.constraint_sign[1] = 1;
    p.constraint_index[2] = 3; p.constraint_sign[2] = 1;
    p.constraint_index[3] = 4; p.constraint_sign[3] = 1;
    std::vector<double> x(count*n), original(count*n), w(count*n*n), expected(count*n);
    for (int f = 0; f < count; ++f)
      {
      ukfMatrixType A(n,n);
      for (int i = 0; i < n; ++i) { for (int j = 0; j < n; ++j) { A(i,j) = normal(random); } }
      ukfMatrixType W = A.transpose()*A+0.2*ukfMatrixType::Identity(n,n);
      ukfVectorType v(n);
      for (int i = 0; i < n; ++i)
        {
        v[i] = normal(random);
        original[f*n+i] = x[f*n+i] = v[i];
        for (int j = 0; j < n; ++j) { w[(f*n+i)*n+j] = W(i,j); }
        }
      ukfMatrixType CI = ukfMatrixType::Zero(n,p.num_constraints), CE = ukfMatrixType::Zero(n,1);
      ukfVectorType ci(p.num_constraints), ce = ukfVectorType::Zero(1);
      for (int c = 0; c < p.num_constraints; ++c) { CI(p.constraint_index[c],c) = p.constraint_sign[c]; ci[c] = p.constraint_offset[c]; }
      ukfVectorType g0 = -W.transpose()*v;
      if ((-CI.transpose()*v-ci).maxCoeff()>0) { QuadProgPP::solve_quadprog(W,g0,CE,ce,CI,ci,v); }
      for (int i = 0; i < n; ++i) { expected[f*n+i] = v[i]; }
      }
    double *dx, *dw; int* dok;
    cudaMalloc(&dx,x.size()*sizeof(double)); cudaMalloc(&dw,w.size()*sizeof(double)); cudaMalloc(&dok,count*sizeof(int));
    cudaMemcpy(dx,x.data(),x.size()*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(dw,w.data(),w.size()*sizeof(double),cudaMemcpyHostToDevice);
    project<<<(count+31)/32,32>>>(p,dx,dw,dok,count);
    if (cudaDeviceSynchronize() != cudaSuccess) { return 2; }
    std::vector<int> ok(count);
    cudaMemcpy(x.data(),dx,x.size()*sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(ok.data(),dok,count*sizeof(int),cudaMemcpyDeviceToHost);
    cudaFree(dx); cudaFree(dw); cudaFree(dok);
    double max_error = 0;
    for (int f = 0; f < count; ++f)
      {
      if (!ok[f]) { fprintf(stderr,"QP failed n=%d case=%d\n",n,f); return 1; }
      for (int i = 0; i < n; ++i)
        {
        max_error = fmax(max_error,fabs(x[f*n+i]-expected[f*n+i]));
        double grad = 0;
        for (int j = 0; j < n; ++j) { grad += w[(f*n+i)*n+j]*(x[f*n+j]-original[f*n+j]); }
        bool active = false;
        for (int c = 0; c < p.num_constraints; ++c)
          {
          if (i != p.constraint_index[c]) { continue; }
          const double residual = p.constraint_sign[c]*x[f*n+i]+p.constraint_offset[c];
          if (residual < -1e-7) { fprintf(stderr,"Infeasible n=%d case=%d residual=%g\n",n,f,residual); return 1; }
          if (fabs(residual)<1e-7)
            { active = true; if (grad*p.constraint_sign[c]<-1e-7) { fprintf(stderr,"Dual infeasible\n"); return 1; } }
          }
        if (!active && fabs(grad)>1e-7) { fprintf(stderr,"Stationarity failed n=%d case=%d grad=%g\n",n,f,grad); return 1; }
        }
      }
    if (max_error>1e-7) { fprintf(stderr,"CPU/GPU QP mismatch %g\n",max_error); return 1; }
    printf("PASS QP n=%d cases=%d max_cpu_gpu_error=%.3g; feasibility and KKT verified\n",n,count,max_error);
    }
}
