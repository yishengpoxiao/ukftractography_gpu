// Model-dependent signal and tracking operations. Filter matrices remain FP64.
#ifndef UKF_GPU_GENERAL_CUH
#define UKF_GPU_GENERAL_CUH
#include "gpu_eigen_math.cuh"

template<typename T>
__device__ void d_rotation(const GPUDarParams& P, const T* state, T* R)
{
  T ct,st,cp,sp,cs,ss;
  if (P.libm_rotation)
    {
    double sine,cosine;
    d_libm_sincos(state[0],sine,cosine,P.libm_rotation-1); st=sine; ct=cosine;
    d_libm_sincos(state[1],sine,cosine,P.libm_rotation-1); sp=sine; cp=cosine;
    d_libm_sincos(state[2],sine,cosine,P.libm_rotation-1); ss=sine; cs=cosine;
    }
  else
    { ct=cos(state[0]); st=sin(state[0]); cp=cos(state[1]); sp=sin(state[1]); cs=cos(state[2]); ss=sin(state[2]); }
  R[0] = ct*cp*cs-sp*ss; R[3] = ct*cs*sp+cp*ss; R[6] = -cs*st;
  R[1] = -cs*sp-ct*cp*ss; R[4] = cp*cs-ct*sp*ss; R[7] = st*ss;
  R[2] = cp*st; R[5] = st*sp; R[8] = ct;
}

template<typename T>
__device__ void d_tensor_matrix(const GPUDarParams& P, const double* state, T* matrix)
{
  T s[6];
  for (int c = 0; c < (P.full ? 6 : 5); ++c) { s[c] = (T)state[c]; }
  const T lower = P.free_water ? T(0) : T(P.lambda_min);
  T l[3] = {fmax(s[3], lower), fmax(s[4], lower), P.full ? fmax(s[5],lower) : fmax(s[4],lower)};
  if (!P.full)
    {
    d_initNormalized(s[0],s[1],s[2]);
    if (s[0] < 0) { for (int c = 0; c < 3; ++c) { s[c] = -s[c]; } }
    d_diffusion(s[0],s[1],s[2],l[0],l[1],l[2],matrix);
    return;
    }
  T R[9], scaled[9];
  d_rotation(P,s,R);
  for (int a = 0; a < 3; ++a)
    { for (int c = 0; c < 3; ++c) { scaled[a*3+c] = R[a*3+c]*l[c]; } }
  for (int a = 0; a < 3; ++a)
    {
    for (int b = 0; b < 3; ++b)
      {
      const T v = scaled[a*3]*R[b*3] + (scaled[a*3+1]*R[b*3+1] + scaled[a*3+2]*R[b*3+2]);
      matrix[a*3+b] = v*T(1e-6);
      }
    }
}

template<typename T>
__device__ T d_tensor_signal(const GPUDarParams& P, const T* matrices, const double* state,
                            const double* gradients, double b, int q)
{
  const T u[3] = {(T)gradients[q*3],(T)gradients[q*3+1],(T)gradients[q*3+2]};
  T signal = 0;
  for (int t = 0; t < P.num_tensors; ++t)
    {
    const T* D = matrices + t*9;
    const T du0 = D[0]*u[0]+D[1]*u[1]+D[2]*u[2];
    const T du1 = D[3]*u[0]+D[4]*u[1]+D[5]*u[2];
    const T du2 = D[6]*u[0]+(D[7]*u[1]+D[8]*u[2]);
    const T exponent = -(T)b*(u[0]*du0+u[1]*du1+u[2]*du2);
    const T value = P.eigen_exp_packet > 1 ? (T)d_full2fw_exp(P,(double)exponent,t,q) : d_host_exp(P.libm_exp,exponent);
    // The original 1T free-water model does not apply a tensor mixture weight.
    signal += value * (P.num_tensors == 1 && P.free_water ? T(1) : T(P.w1));
    }
  if (P.free_water)
    {
    const T w = fmax((T)state[P.n-1],T(0));
    const T exponent = -(T)b*(u[0]*(T(P.diff_fw)*u[0])+u[1]*(T(P.diff_fw)*u[1])+u[2]*(T(P.diff_fw)*u[2]));
    const T iso = P.eigen_exp_packet > 1 ? (T)d_full2fw_exp(P,(double)exponent,2,q) : d_host_exp(P.libm_exp,exponent);
    signal = w*signal+(T(1)-w)*iso;
    }
  return signal;
}

// Evaluate the complete NODDI observation, including its signal-wide correction
// of nonpositive intracellular values. Original special-function branches use FP64.
__device__ void d_noddi_signal(const GPUDarParams& P, const DevBufs& D, const double* state,
                               double* result, double* work)
{
  const int lane = threadIdx.x & 31, Q2 = 2*P.G;
  double* coeff = work + Q2; // seven SH coefficients, dw[2], normalized direction[3], last positive
  for (int q = lane; q < Q2; q += 32) { result[q] = 0; }
  __syncwarp();
  for (int t = 0; t < P.num_tensors; ++t)
    {
    const double* s = state + t*5;
    if (lane == 0)
      {
      d_WatsonSHCoeff(s[4],coeff);
      d_WatsonHinderedDiffusionCoeff(1.7e-9,1.7e-9*(1-s[3]),s[4],coeff+7);
      coeff[9] = s[0]; coeff[10] = s[1]; coeff[11] = s[2];
      d_initNormalized(coeff[9],coeff[10],coeff[11]);
      if (P.num_tensors == 2 && coeff[9] < 0)
        { coeff[9] = -coeff[9]; coeff[10] = -coeff[10]; coeff[11] = -coeff[11]; }
      }
    __syncwarp();
    for (int q = lane; q < Q2; q += 32)
      {
      double ct = D.gradients[q*3]*coeff[9]+D.gradients[q*3+1]*coeff[10]+D.gradients[q*3+2]*coeff[11];
      if (ct > 1) { ct /= (int)ct; }
      const double* table = D.noddi_table+(q%P.G)*10;
      double v = 0;
      for (int j = 0; j < 7; ++j) { v += sqrt((j+0.25)*7/22)*d_Legendre(2*j,ct)*coeff[j]*table[j+3]; }
      work[q] = 0.5*v;
      }
    __syncwarp();
    if (lane == 0)
      {
      // Preserve the original loop's last-positive behavior, including b=0.
      double last = 0;
      for (int q = 0; q < Q2; ++q) { if (work[q] > 0) { last = work[q]; } }
      coeff[12] = last;
      }
    __syncwarp();
    for (int q = lane; q < Q2; q += 32)
      {
      const double* table = D.noddi_table+(q%P.G)*10;
      const double ct = D.gradients[q*3]*coeff[9]+D.gradients[q*3+1]*coeff[10]+D.gradients[q*3+2]*coeff[11];
      const double ec = exp(-table[0]*((coeff[7]-coeff[8])*ct*ct+coeff[8]));
      const double ic = work[q] > 0 ? work[q] : 0.1*coeff[12];
      const double v = s[3]*ic+(1-s[3])*ec;
      result[q] += v*(P.num_tensors == 1 ? 1.0 : P.w1);
      }
    __syncwarp();
    }
  for (int q = lane; q < Q2; q += 32)
    {
    const double viso = state[P.n-1];
    result[q] = viso*D.noddi_table[(q%P.G)*10+2]+(1-viso)*result[q];
    }
  __syncwarp();
}

template<typename T>
__global__ void k_H_general(GPUDarParams P, DevBufs D, int F)
{
  const int f = D.act_idx[blockIdx.x];
  if (f >= F || !D.active[f]) { return; }
  const int Q2 = 2*P.G, S = 2*P.n+1, lane = threadIdx.x&31, warp = threadIdx.x>>5;
  double* memory = (double*)ukf_smem_raw;
  for (int i = warp; i < S; i += blockDim.x/32)
    {
    const double* state = D.X+((size_t)f*S+i)*P.n;
    double* output = D.Z+((size_t)f*S+i)*Q2;
    if (P.noddi) { d_noddi_signal(P,D,state,output,memory+warp*(Q2+16)); }
    else
      {
      T* matrices = (T*)(memory+warp*32);
      if (lane == 0)
        { for (int t = 0; t < P.num_tensors; ++t) { d_tensor_matrix(P,state+t*(P.full?6:5),matrices+t*9); } }
      __syncwarp();
      const int count = P.eigen_exp_packet > 1 ? Q2 : P.G;
      for (int q = lane; q < count; q += 32)
        {
        output[q] = d_tensor_signal(P,matrices,state,D.gradients,D.bvalues[q],q);
        if (count < Q2) { output[q+P.G] = output[q]; }
        }
      __syncwarp();
      }
    }
}

__device__ void d_extract_tensor(const GPUDarParams& P, const double* state, double* direction, double* lambda)
{
  if (P.full)
    {
    double R[9]; d_rotation(P,state,R);
    for (int c = 0; c < 3; ++c) { direction[c] = R[c*3]; }
    }
  else
    {
    for (int c = 0; c < 3; ++c) { direction[c] = state[c]; }
    d_initNormalized(direction[0],direction[1],direction[2]);
    }
  const double low = P.free_water ? 0 : P.lambda_min;
  lambda[0] = fmax(state[3],low); lambda[1] = fmax(state[4],low);
  lambda[2] = P.full ? fmax(state[5],low) : lambda[1];
}

__device__ void d_swap_tensor(double* state, double* cov, int n, int width, int tensor)
{
  for (int i = 0; i < width; ++i)
    { double v = state[i]; state[i] = state[tensor*width+i]; state[tensor*width+i] = v; }
  for (int i = 0; i < width; ++i)
    { for (int c = 0; c < n; ++c) { double v = cov[i*n+c]; cov[i*n+c] = cov[(tensor*width+i)*n+c]; cov[(tensor*width+i)*n+c] = v; } }
  for (int i = 0; i < width; ++i)
    { for (int r = 0; r < n; ++r) { double v = cov[r*n+i]; cov[r*n+i] = cov[r*n+tensor*width+i]; cov[r*n+tensor*width+i] = v; } }
}

__device__ void d_swap_vec(double* a, double* b)
{
  for (int c = 0; c < 3; ++c) { double v = a[c]; a[c] = b[c]; b[c] = v; }
}

__global__ void k_step_general(GPUDarParams P, DevBufs D, int F, int step)
{
  const int f = D.act_idx[blockIdx.x];
  if (f >= F || !D.active[f]) { return; }
  const int Q2 = 2*P.G, n = P.n, width = P.full ? 6 : 5;
  double* st = D.state+(size_t)f*n;
  double* cov = D.cov+(size_t)f*n*n;
  double* x = D.pos+f*3;
  double* memory = (double*)ukf_smem_raw;
  double m[3][3], l[3][3];
  if (threadIdx.x == 0)
    {
    double fa[3] = {}, traces[3] = {};
    const double* old = D.m1+f*3;
    double dots[3];
    for (int t = 0; t < P.num_tensors; ++t)
      {
      if (P.noddi)
        { for (int c = 0; c < 3; ++c) { m[t][c] = st[t*5+c]; } if (P.num_tensors>1) { d_initNormalized(m[t][0],m[t][1],m[t][2]); } }
      else
        {
        d_extract_tensor(P,st+t*width,m[t],l[t]);
        fa[t] = d_l2fa(l[t][0],l[t][1],l[t][2]);
        traces[t] = l[t][0]+l[t][1]+l[t][2];
        }
      dots[t] = m[t][0]*old[0]+m[t][1]*old[1]+m[t][2]*old[2];
      if (P.num_tensors > 1 && dots[t] < 0)
        { for (int c = 0; c < 3; ++c) { m[t][c] = -m[t][c]; } dots[t] = -dots[t]; }
      }
    const double angle = P.num_tensors == 2 ? acos(m[0][0]*m[1][0]+m[0][1]*m[1][1]+m[0][2]*m[1][2])*180/M_PI : 0;
    D.trace[f] = traces[0]; if (P.num_tensors>1) { D.trace2[f] = traces[1]; }
    int selected = 0;
    if (P.num_tensors == 2 && dots[0] < dots[1]) { selected = 1; }
    if (P.num_tensors == 3)
      {
      if (dots[0]<dots[1] && dots[2]<dots[1]) { selected = 1; }
      else if (dots[0]<dots[2]) { selected = 2; }
      }
    if (selected)
      {
      d_swap_vec(m[0],m[selected]);
      if (!P.noddi) { d_swap_vec(l[0],l[selected]); }
      d_swap_tensor(st,cov,n,width,selected);
      if (P.num_tensors == 2)
        { double v = fa[0]; fa[0] = fa[1]; fa[1] = v; v = D.trace[f]; D.trace[f] = D.trace2[f]; D.trace2[f] = v; }
      }
    if (P.num_tensors == 2 && angle <= 20 && (P.noddi || (fmin(fa[0],fa[1])<=0.2 && fa[0]<=0.2)))
      {
      d_swap_vec(m[0],m[1]);
      if (!P.noddi) { d_swap_vec(l[0],l[1]); }
      d_swap_tensor(st,cov,n,width,1);
      }
    if (P.noddi)
      {
      D.fa[f] = st[3]; D.trace[f] = st[4];
      if (P.num_tensors == 2) { D.fa2[f] = st[8]; D.trace2[f] = st[9]; }
      }
    else
      {
      D.fa[f] = l[0][0]<l[0][1] || l[0][0]<l[0][2] ? 0 : d_l2fa(l[0][0],l[0][1],l[0][2]);
      if (P.num_tensors > 1)
        { if (l[0][0]<l[0][1] || l[0][0]<l[0][2]) { if (P.num_tensors == 2) { D.fa2[f] = 0; } }
          else { D.fa2[f] = d_l2fa(l[1][0],l[1][1],l[1][2]); } }
      }
    for (int c = 0; c < 3; ++c) { D.m1[f*3+c] = m[0][c]; }
    x[0] += m[0][2]/P.vx*P.stepLength; x[1] += m[0][1]/P.vy*P.stepLength; x[2] += m[0][0]/P.vz*P.stepLength;
    }
  __syncthreads();
  if (P.noddi) { d_noddi_signal(P,D,st,memory,memory+Q2); }
  else
    {
    double* matrices = memory+Q2;
    if (threadIdx.x == 0)
      { for (int t = 0; t < P.num_tensors; ++t) { d_tensor_matrix(P,st+t*width,matrices+t*9); } }
    __syncthreads();
    const int count = P.eigen_exp_packet > 1 ? Q2 : P.G;
    for (int q = threadIdx.x; q < count; q += blockDim.x)
      {
      memory[q] = d_tensor_signal(P,matrices,st,D.gradients,D.bvalues[q],q);
      if (count < Q2) { memory[q+P.G] = memory[q]; }
      }
    __syncthreads();
    }
  if (threadIdx.x != 0) { return; }
  double mean = 0;
  for (int q = 0; q < Q2; ++q) { mean += memory[q]; }
  mean /= Q2;
  const int rx = (int)round(x[0]), ry = (int)round(x[1]), rz = (int)round(x[2]);
  const bool in_csf = (!P.use_gmcsf || P.num_tensors == 3) && (mean<P.mean_signal_min || (!P.noddi && D.fa[f]<P.fa_min));
  const bool stop = P.num_tensors != 3 && d_stop_value(D,P,rx,ry,rz)>0;
  const bool low_kappa = P.noddi && (st[4] < (P.num_tensors==1 ? 1.2 : 0.6) || (P.num_tensors==2 && st[9]<0.6));
  if (d_mask_value(D,P,rx,ry,rz)<=0 || in_csf || stop || low_kappa || d_curve_radius(D,P,f)<P.min_radius)
    { D.active[f] = 0; return; }
  if (P.branching)
    {
    double fa[3]; bool anisotropic[3];
    for (int t = 0; t < P.num_tensors; ++t)
      { fa[t]=d_l2fa(l[t][0],l[t][1],l[t][2]); anisotropic[t]=l[t][0]>l[t][1] && l[t][0]>l[t][2] && fa[t]>P.fa_min; }
    bool add[3] = {};
    const double dot12=m[0][0]*m[1][0]+m[0][1]*m[1][1]+m[0][2]*m[1][2];
    const bool branch12=dot12<P.cos_branch_min && dot12>P.cos_branch_max;
    if (P.num_tensors == 2) { add[1]=anisotropic[0] && anisotropic[1] && branch12; }
    else if (anisotropic[0])
      {
      const double dot13=m[0][0]*m[2][0]+m[0][1]*m[2][1]+m[0][2]*m[2][2];
      const double dot23=m[1][0]*m[2][0]+m[1][1]*m[2][1]+m[1][2]*m[2][2];
      const bool branch13=dot13<P.cos_branch_min && dot13>P.cos_branch_max;
      if (anisotropic[1] && branch12)
        {
        if (anisotropic[2] && branch13)
          { if (dot23<P.cos_branch_min) { add[1]=add[2]=true; } else { add[fa[1]>fa[2]?1:2]=true; } }
        else { add[1]=true; }
        }
      else if (anisotropic[2] && branch13) { add[2]=true; }
      }
    for (int t = 1; t < P.num_tensors; ++t)
      {
      if (!add[t]) { continue; }
      const int index=atomicAdd(D.raw.branch_count,1);
      GPUBranchSeed& seed=D.raw.branches[index]; seed=GPUBranchSeed{};
      seed.fiber=f; seed.step=step; seed.tensor=t; seed.fa=fa[t];
      for (int c=0;c<3;++c) { seed.pos[c]=x[c]; seed.direction[c]=m[t][c]; }
      for (int c=0;c<n;++c) { seed.state[c]=st[c]; }
      for (int c=0;c<n*n;++c) { seed.cov[c]=cov[c]; }
      d_swap_tensor(seed.state,seed.cov,n,width,t);
      }
    }
  if ((step+1)%P.steps_per_record != 0) { return; }
  const size_t record = (size_t)f*P.max_rec+D.reccount[f];
  for (int c = 0; c < 3; ++c) { D.rec_pos[record*3+c] = x[c]; }
  D.rec_norm[record] = d_frobnorm_colmajor(cov,n);
  D.rec_nmse[record] = D.nmse[f];
  D.rec_trace[record] = D.trace[f]; D.rec_trace2[record] = D.trace2[f];
  D.rec_fa[record] = D.fa[f]; D.rec_fa2[record] = D.fa2[f];
  double* rs = D.rec_state+record*n;
  for (int c = 0; c < n; ++c) { rs[c] = st[c]; }
  if (n == 5 || n == 10 || n == 15)
    {
    d_initNormalized(rs[0],rs[1],rs[2]);
    if (n == 10) { d_initNormalized(rs[5],rs[6],rs[7]); }
    if (n == 15) { d_initNormalized(rs[10],rs[11],rs[12]); }
    }
  if (P.record_cov) { for (int c = 0; c < n*n; ++c) { D.rec_cov[record*n*n+c] = cov[c]; } }
  ++D.reccount[f];
}
#endif
