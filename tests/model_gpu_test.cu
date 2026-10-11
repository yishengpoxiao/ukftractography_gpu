// Isolate the GPU sigma/constraint/H/update pipeline from trajectory branch decisions.
#include <memory>
#include <random>
#include "filter_Simple1T.h"
#include "filter_Simple2T.h"
#include "filter_Simple3T.h"
#include "filter_Full1T.h"
#include "filter_Full2T.h"
#include "filter_Full3T.h"
#include "filter_Simple1T_FW.h"
#include "filter_Simple2T_FW.h"
#include "filter_Full1T_FW.h"
#include "filter_Full2T_FW.h"
#include "filter_NODDI1F.h"
#include "filter_NODDI2F.h"
#include "utilities.h"
#include "../ukf/gpu/gpu_kernels.cu"

struct Signal : ISignalData
{
  stdVec_t dirs;
  ukfVectorType b;
  Signal() : ISignalData(0,0), dirs(62), b(62)
    {
    std::mt19937 random(41); std::normal_distribution<double> normal(0,1);
    for (int q = 0; q < 31; ++q)
      { dirs[q] << normal(random),normal(random),normal(random); dirs[q].normalize(); dirs[q+31] = -dirs[q]; b[q] = b[q+31] = 1000+(q%3)*1000; }
    }
  void Interp3Signal(const vec3_t&,ukfVectorType&) const override {}
  double Interp3ScalarMask(const vec3_t&) const override { return 1; }
  double ScalarMaskValue(const vec3_t&) const override { return 1; }
  double ScalarStopValue(const std::vector<int>&,double,double,const vec3_t&) const override { return 0; }
  double isGMCSFProvided() const override { return 0; }
  void GetSeeds(const std::vector<int>&,double,stdVec_t&) const override {}
  const stdVec_t& gradients() const override { return dirs; }
  const ukfVectorType& GetBValues() const override { return b; }
  int GetSignalDimension() const override { return 31; }
  bool LoadData(const std::string&,const std::string&,bool,bool,const std::string&,const std::string&,const std::string&,const std::string&,const std::string&) override { return false; }
  vec3_t dim() const override { return vec3_t::Zero(); }
};

__global__ void watson(const double* kappa,double* out,int count)
{
  const int i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i>=count) { return; }
  d_WatsonSHCoeff(kappa[i],out+i*9);
  d_WatsonHinderedDiffusionCoeff(1.7e-9,0.7e-9,kappa[i],out+i*9+7);
}

int main()
{
  const std::vector<double> kappas={0,1e-8,1e-5,0.05,0.1,0.10001,1,4,8,29.999,30,30.001,100};
  double *dk,*dy; cudaMalloc(&dk,kappas.size()*sizeof(double)); cudaMalloc(&dy,kappas.size()*9*sizeof(double));
  cudaMemcpy(dk,kappas.data(),kappas.size()*sizeof(double),cudaMemcpyHostToDevice);
  watson<<<1,32>>>(dk,dy,kappas.size());
  if (cudaDeviceSynchronize()!=cudaSuccess) { return 2; }
  std::vector<double> coefficients(kappas.size()*9);
  cudaMemcpy(coefficients.data(),dy,coefficients.size()*sizeof(double),cudaMemcpyDeviceToHost);
  cudaFree(dk); cudaFree(dy);
  for (size_t i=0;i<kappas.size();++i)
    {
    ukfVectorType sh(7),dw(2); WatsonSHCoeff(kappas[i],sh); WatsonHinderedDiffusionCoeff(1.7e-9,0.7e-9,kappas[i],dw);
    for (int j=0;j<9;++j)
      {
      const double expected=j<7?sh[j]:dw[j-7], actual=coefficients[i*9+j];
      if (!std::isfinite(actual) || fabs(actual-expected)>1e-8*(1+fabs(expected)))
        { fprintf(stderr,"Watson mismatch kappa=%g coefficient=%d CPU=%.17g GPU=%.17g\n",kappas[i],j,expected,actual); return 1; }
      }
    }
  printf("PASS Watson boundaries: %zu kappas, SH and hindered diffusion\n",kappas.size());
  Signal data;
  for (int kind = 0; kind < 12; ++kind)
    {
    const int nt = kind<6 ? kind/2+1 : kind<10 ? (kind-6)/2+1 : kind-9;
    const bool full = kind<10 && kind%2, fw = kind>=6 && kind<10, noddi = kind>=10;
    ukfVectorType weights = ukfVectorType::Constant(nt,1.0/nt);
    std::unique_ptr<FilterModel> model;
    const double qs = 0.0001, ql = 0.01, rs = 0.015, qw = 0.0001;
    switch (kind)
      {
      case 0: model.reset(new Simple1T(qs,ql,rs,weights,false)); break;
      case 1: model.reset(new Full1T(qs,ql,rs,weights,false)); break;
      case 2: model.reset(new Simple2T(qs,ql,rs,weights,false)); break;
      case 3: model.reset(new Full2T(qs,ql,rs,weights,false)); break;
      case 4: model.reset(new Simple3T(qs,ql,rs,weights,false)); break;
      case 5: model.reset(new Full3T(qs,ql,rs,weights,false)); break;
      case 6: model.reset(new Simple1T_FW(qs,ql,qw,rs,weights,true,0.003)); break;
      case 7: model.reset(new Full1T_FW(qs,ql,qw,rs,weights,true,0.003)); break;
      case 8: model.reset(new Simple2T_FW(qs,ql,qw,rs,weights,true,0.003)); break;
      case 9: model.reset(new Full2T_FW(qs,ql,qw,rs,weights,true,0.003)); break;
      case 10: model.reset(new NODDI1F(qs,qw,qw,rs,weights,true)); break;
      case 11: model.reset(new NODDI2F(qs,qw,qw,rs,weights,true)); break;
      }
    model->set_signal_data(&data); model->set_signal_dim(62);
    const int n = model->state_dim(), S = 2*n+1, Q2 = 62, width = full?6:5;
    GPUDarParams p = {};
    p.model = UKF_GPU_MODEL_GENERAL; p.n = n; p.num_tensors = nt;
    p.full = full; p.free_water = fw; p.noddi = noddi; p.G = 31; p.lambda_min = 100; p.diff_fw = 0.003;
    p.Rs = rs; p.w1 = 1.0/nt; p.has_process_noise = 1;
    for (int i = 0; i < n; ++i) { p.process_noise[i] = model->Q()(i,i); }
    if (model->isConstrained())
      {
      p.num_constraints = model->d().size();
      for (int c = 0; c < p.num_constraints; ++c)
        { p.constraint_offset[c] = model->d()[c]; for (int i = 0; i < n; ++i) { if (model->D()(i,c)!=0) { p.constraint_index[c]=i; p.constraint_sign[c]=model->D()(i,c); } } }
      }
    State state(n);
    for (int t = 0; t < nt; ++t)
      {
      if (full) { state.segment(t*width,3) << 0.7+0.2*t,0.4+0.6*t,-0.3+0.4*t; }
      else { vec3_t m(0.7-0.2*t,0.3+0.4*t,0.5); m.normalize(); state.segment(t*width,3)=m; }
      if (noddi) { state[t*width+3]=0.02+0.5*t; state[t*width+4]=8+4*t; }
      else { state[t*width+3]=1700+300*t; state[t*width+4]=400+30*t; if (full) { state[t*width+5]=200+20*t; } }
      }
    if (fw || noddi) { state[n-1] = 0.99; }
    ukfMatrixType covariance = ukfMatrixType::Identity(n,n)*0.001;
    for (int i = 0; i < n; ++i) { for (int j = 0; j < n; ++j) { covariance(i,j)+=0.0001; } }
    ukfMatrixType one(n,1), observation(Q2,1); one.col(0)=state; model->H(one,observation);
    const ukfVectorType z = observation.col(0)*0.99;
    std::vector<double> cov(n*n), gradients(Q2*3), table(31*10);
    for (int i=0;i<n;++i) { for(int j=0;j<n;++j) { cov[i*n+j]=covariance(i,j); } }
    for (int q=0;q<Q2;++q) { for (int c=0;c<3;++c) { gradients[q*3+c]=data.dirs[q][c]; } }
    if (noddi)
      {
      ukfVectorType strength,separation,parallel(Q2),iso;
      createProtocol(data.b,strength,separation);
      for(int q=0;q<Q2;++q) { double m=267598700.0*267598700.0*separation[q]*separation[q]*strength[q]*strength[q]; parallel[q]=m*(separation[q]-separation[q]/3)*1.7e-9; }
      ukfMatrixType integral(Q2,7); legendreGaussianIntegral(parallel,6,integral); IsoModel(3e-9,strength,separation,iso);
      for(int q=0;q<31;++q)
        { double m=267598700.0*267598700.0*separation[q]*separation[q]*strength[q]*strength[q]; table[q*10]=(separation[q]-separation[q]/3)*m; table[q*10+2]=iso[q]; for(int j=0;j<7;++j) { table[q*10+3+j]=integral(q,j); } }
      }
    GPUDevPointers raw = {}; DeviceBuffers storage; std::string err;
#define ALLOC_TEST(field,count) if(!storage.allocate(reinterpret_cast<void**>(&raw.field),sizeof(double)*(count),err)) { return 2; }
    ALLOC_TEST(state,n); ALLOC_TEST(cov,n*n); ALLOC_TEST(z,Q2); ALLOC_TEST(X,S*n); ALLOC_TEST(Z,S*Q2);
    ALLOC_TEST(scratch,2*n*Q2); ALLOC_TEST(nmse,1); ALLOC_TEST(gradients,Q2*3); ALLOC_TEST(bvalues,Q2); ALLOC_TEST(noddi_table,31*10);
    ALLOC_TEST(active,1); ALLOC_TEST(act_idx,1); ALLOC_TEST(invalid,1);
#undef ALLOC_TEST
    cudaMemcpy(raw.state,state.data(),n*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(raw.cov,cov.data(),n*n*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(raw.z,z.data(),Q2*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(raw.gradients,gradients.data(),Q2*3*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(raw.bvalues,data.b.data(),Q2*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(raw.noddi_table,table.data(),31*10*sizeof(double),cudaMemcpyHostToDevice);
    const int active=1; cudaMemcpy(raw.active,&active,sizeof(int),cudaMemcpyHostToDevice);
    cudaMemset(raw.act_idx,0,sizeof(int)); cudaMemset(raw.invalid,0,sizeof(int));
    DevBufs d = DevBufs::from(raw);
    auto sigmas = [&]()
      {
#define SIGMA(N) case N: k_sigmas<N><<<1,32,sizeof(double)*(n*n+n)>>>(p,d,1); break
      switch(n) { SIGMA(5); SIGMA(6); SIGMA(7); SIGMA(10); SIGMA(11); SIGMA(12); SIGMA(13); SIGMA(15); SIGMA(18); }
#undef SIGMA
      if (p.num_constraints)
        {
#define PROJECT_SIGMA(N) case N: k_constrain_sigmas<N><<<(S+31)/32,32>>>(p,d,1,1); break
        switch(n) { PROJECT_SIGMA(6); PROJECT_SIGMA(7); PROJECT_SIGMA(11); PROJECT_SIGMA(13); }
#undef PROJECT_SIGMA
        }
      };
    auto predict = [&]() { k_H_general<double><<<1,32,sizeof(double)*(noddi?Q2+16:32)>>>(p,d,1); };
    const size_t shared=sizeof(double)*(4*n*n+4*n+2*Q2+S+S*n)+sizeof(int)*(n+1);
    auto update = [&]()
      {
#define UPDATE(N) case N: k_update<N><<<1,32,shared>>>(p,d,1); break
      switch(n) { UPDATE(5); UPDATE(6); UPDATE(7); UPDATE(10); UPDATE(11); UPDATE(12); UPDATE(13); UPDATE(15); UPDATE(18); }
#undef UPDATE
      if (p.num_constraints)
        {
#define PROJECT_STATE(N) case N: k_constrain_state<N><<<1,32>>>(p,d,1,1); break
        switch(n) { PROJECT_STATE(6); PROJECT_STATE(7); PROJECT_STATE(11); PROJECT_STATE(13); }
#undef PROJECT_STATE
        }
      };
    sigmas(); predict();
    if(cudaDeviceSynchronize()!=cudaSuccess) { return 2; }
    ukfMatrixType X(n,S), Z(Q2,S), H(Q2,S);
    cudaMemcpy(X.data(),raw.X,n*S*sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(Z.data(),raw.Z,Q2*S*sizeof(double),cudaMemcpyDeviceToHost);
    model->H(X,H);
    const double herror=(H-Z).cwiseAbs().maxCoeff();
    update();
    if(cudaDeviceSynchronize()!=cudaSuccess) { return 2; }
    State gpu(n), expected(n); ukfMatrixType expected_cov(n,n), gpu_cov(n,n); double nmse;
    cudaMemcpy(gpu.data(),raw.state,n*sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(cov.data(),raw.cov,n*n*sizeof(double),cudaMemcpyDeviceToHost);
    for(int i=0;i<n;++i) { for(int j=0;j<n;++j) { gpu_cov(i,j)=cov[i*n+j]; } }
    UnscentedKalmanFilter filter(model.get()); filter.Filter(state,covariance,z,expected,expected_cov,nmse);
    double xerror=(gpu-expected).cwiseAbs().maxCoeff(), perror=(gpu_cov-expected_cov).cwiseAbs().maxCoeff();
    for (int step=1;step<20;++step)
      {
      state=expected; covariance=expected_cov;
      sigmas(); predict(); update();
      if (cudaDeviceSynchronize()!=cudaSuccess) { return 2; }
      filter.Filter(state,covariance,z,expected,expected_cov,nmse);
      cudaMemcpy(gpu.data(),raw.state,n*sizeof(double),cudaMemcpyDeviceToHost);
      cudaMemcpy(cov.data(),raw.cov,n*n*sizeof(double),cudaMemcpyDeviceToHost);
      for(int i=0;i<n;++i) { for(int j=0;j<n;++j) { gpu_cov(i,j)=cov[i*n+j]; } }
      xerror=fmax(xerror,(gpu-expected).cwiseAbs().maxCoeff()); perror=fmax(perror,(gpu_cov-expected_cov).cwiseAbs().maxCoeff());
      }
    int invalid=0; cudaMemcpy(&invalid,raw.invalid,sizeof(int),cudaMemcpyDeviceToHost);
    printf("model=%d n=%d steps=20 H_error=%.3g max_state_error=%.3g max_covariance_error=%.3g invalid=%d\n",kind,n,herror,xerror,perror,invalid);
    if(invalid || !gpu.allFinite() || herror>1e-10 || xerror>1e-7 || perror>1e-8) { return 1; }
    }
}
