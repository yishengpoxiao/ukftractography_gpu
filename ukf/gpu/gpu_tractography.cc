/**
 * \file gpu_tractography.cc
 * \brief Host-side glue: packs NrrdData + seeds + settings into POD buffers,
 *        invokes the CUDA wavefront implementation, and unpacks the results
 *        into UKFFiber exactly as the CPU Follow*T/Record path would.
 */
#include "gpu_tractography.h"

#include <algorithm>
#include <chrono>
#include <memory>
#include <array>
#include <cmath>
#include <cstdio>
#include <thread>
#include <cstring>
#include <vector>
#include <sstream>
#include <cstdlib>

#include "../ISignalData.h"
#include "../filter_model.h"
#include "../tractography.h"
#include "../ukffiber.h"
#include "../seed.h"
#include "../utilities.h"

#include "gpu_device_api.h"
#include "gpu_libm_math.cuh"

namespace ukf
{
namespace gpu
{

bool available(std::string& info)
{
  return gpu_available(info);
}

bool run_tractography(ISignalData* signal_data,
                      const UKFSettings& s,
                      int filter_model_type,
                      const FilterModel* filter_model,
                      const std::vector<SeedPointInfo>& seeds,
                      std::vector<UKFFiber>& fibers,
                      std::string& err,
                      std::vector<SeedPointInfo>* branch_seeds,
                      std::vector<BranchingSeedAffiliation>* affiliations)
{
  const int n = filter_model->state_dim();
  const bool branching = s.num_tensors>1 && s.max_branching_angle>0 && !s.noddi;
  if (branching && (!branch_seeds || !affiliations)) { err = "branch output buffers are required"; return false; }
  const int model = filter_model_type == Tractography::_1T ? UKF_GPU_MODEL_SIMPLE1T :
    filter_model_type == Tractography::_2T ? UKF_GPU_MODEL_SIMPLE2T : UKF_GPU_MODEL_GENERAL;
  if (n > UKF_GPU_MAX_STATE) { err = "state dimension exceeds GPU capacity"; return false; }

  const int G = signal_data->GetSignalDimension();
  if (G <= 0 || signal_data->rawDwiData() == NULL)
    {
    err = "no DWI data available for the GPU path";
    return false;
    }
  const int Q2 = 2 * G;
  const int S = (int)seeds.size();

  /* ---- volumes ---- */
  const vec3_t dim = signal_data->rawDims();
  const vec3_t vox = signal_data->rawVoxel();

  GPUDarParams P;
  memset(&P, 0, sizeof(P));
  P.model = model;
  P.n = n;
  P.num_tensors = s.num_tensors;
  P.full = s.is_full_model && !s.noddi;
  P.free_water = s.free_water && !s.noddi;
  P.noddi = s.noddi;
  P.eigen_packet = Eigen::internal::packet_traits<double>::size;
  P.eigen_gemm_mr = Eigen::internal::gebp_traits<double,double>::mr;
  P.eigen_solve_panel = std::max(P.eigen_gemm_mr, (int)Eigen::internal::gebp_traits<double,double>::nr);
#ifdef EIGEN_VECTORIZE_FMA
  P.eigen_fma = 1;
#endif
  // Full2T_FW::H evaluates a whole Eigen array with packet exp and a scalar tail.
  if (P.full && P.free_water && P.num_tensors == 2 && Eigen::internal::packet_traits<double>::HasExp)
    {
    P.eigen_exp_packet = P.eigen_packet;
    }
  // Select by actual host results, avoiding a dependency on a libc/OS version.
  // Volatile prevents constant folding from substituting compiler math results.
  const volatile double math_samples[] = {-74.211523979052288,20.737142811295371,83.801544053310693,
    40.726217187925158,-3.14,-1.5707963267948966,-0.855469,-0.126,-0.01,0,1e-9,
    0.00333988251,0.126,0.855469,1.5707963267948966,2.426265,3.14,11.5,100,500};
  for (int fused = 0; fused <= 1; ++fused)
    {
    bool same_exp = true, same_rotation = true;
    for (double sample : math_samples)
      {
      double sine, cosine;
      d_libm_sincos(sample,sine,cosine,fused);
      const double host_sine = std::sin(sample), host_cosine = std::cos(sample);
      same_rotation = same_rotation && sine == host_sine && cosine == host_cosine;
      same_exp = same_exp && d_libm_exp(-std::abs(sample),fused) == std::exp(-std::abs(sample));
      }
    if (same_exp) { P.libm_exp = fused+1; }
    if (same_rotation) { P.libm_rotation = fused+1; }
    }
  P.diff_fw = D_ISO;
  P.branching = branching;
  P.cos_branch_min = cos(s.min_branching_angle*M_PI/180);
  P.cos_branch_max = cos(s.max_branching_angle*M_PI/180);
  P.has_process_noise = 1;
  for (int d = 0; d < n; ++d) { P.process_noise[d] = filter_model->Q()(d,d); }
  if (filter_model->isConstrained())
    {
    P.num_constraints = filter_model->d().size();
    for (int c = 0; c < P.num_constraints; ++c)
      {
      P.constraint_offset[c] = filter_model->d()[c];
      for (int d = 0; d < n; ++d)
        {
        if (filter_model->D()(d,c) != 0)
          {
          P.constraint_index[c] = d;
          P.constraint_sign[c] = filter_model->D()(d,c);
          }
        }
      }
    }
  P.G = G;
  P.nx = (int)dim[0];
  P.ny = (int)dim[1];
  P.nz = (int)dim[2];
  P.vx = vox[0];
  P.vy = vox[1];
  P.vz = vox[2];
  P.sigma_signal = s.sigma_signal;
  P.Qm = s.Qm;
  P.Ql = s.Ql;
  P.Rs = s.Rs;
  P.w1 = 1.0 / s.num_tensors;
  P.w2 = P.w1;
  P.lambda_min = 100.0;
  P.stepLength = s.stepLength;
  P.mean_signal_min = s.mean_signal_min;
  P.fa_min = s.fa_min;
  P.min_radius = s.min_radius;
  P.max_length = (int)std::ceil(s.maxHalfFiberLength / s.stepLength);
  P.steps_per_record = (int)(s.recordLength / s.stepLength);
  if (P.steps_per_record < 1)
    {
    P.steps_per_record = 1;
    }
  P.record_nmse = s.record_nmse ? 1 : 0;
  P.record_trace = s.record_trace || s.record_kappa;
  P.record_fa = s.record_fa || s.record_Vic;
  P.record_cov = s.record_cov ? 1 : 0;
  P.gm_thr = s.gm_prob_threshold;
  P.csf_thr = s.csf_prob_threshold;
  P.use_gmcsf = (int)signal_data->isGMCSFProvided();
  // Three-component direction selection amplifies FP32 observation errors.
  P.use_float = s.gpu_precision == 1 && P.num_tensors <= 2 && !P.full && !P.free_water && !P.noddi;
  P.num_stop_labels = (int)s.stopLabels.size();
  if (P.num_stop_labels > UKF_GPU_MAX_STOP_LABELS)
    {
    err = "too many stop labels for the GPU path";
    return false;
    }
  for (int i = 0; i < P.num_stop_labels; ++i)
    {
    P.stop_labels[i] = s.stopLabels[i];
    }
  P.max_rec = P.max_length / P.steps_per_record + 2;
  P.step1 = (std::int64_t)P.nz * P.ny * G;
  P.step2 = (std::int64_t)P.nz * G;
  P.mask_type = signal_data->rawMaskType();
  P.stop_type = signal_data->rawStopType();
  P.gm_type = signal_data->rawGmType();
  P.csf_type = signal_data->rawCsfType();

  /* ---- constant tables ---- */
  const stdVec_t& grads = signal_data->gradients();
  const ukfVectorType& bvals = signal_data->GetBValues();
  if ((int)grads.size() != Q2 || (int)bvals.size() != Q2)
    {
    err = "gradient table size mismatch";
    return false;
    }
  std::vector<double> grad_flat((size_t)Q2 * 3);
  std::vector<double> b_flat(Q2);
  for (int i = 0; i < Q2; ++i)
    {
    grad_flat[(size_t)i * 3 + 0] = grads[i][0];
    grad_flat[(size_t)i * 3 + 1] = grads[i][1];
    grad_flat[(size_t)i * 3 + 2] = grads[i][2];
    b_flat[i] = bvals[i];
    }

  std::vector<double> noddi_table;
  if (P.noddi)
    {
    ukfVectorType strength, separation, parallel(Q2), iso;
    createProtocol(bvals,strength,separation);
    for (int q = 0; q < Q2; ++q)
      {
      const double modq = 267598700.0*267598700.0*separation[q]*separation[q]*strength[q]*strength[q];
      parallel[q] = modq*(separation[q]-separation[q]/3)*1.7e-9;
      }
    ukfMatrixType integrals(Q2,7);
    legendreGaussianIntegral(parallel,6,integrals);
    IsoModel(3e-9,strength,separation,iso);
    noddi_table.resize(G*10);
    for (int q = 0; q < G; ++q)
      {
      const double modq = 267598700.0*267598700.0*separation[q]*separation[q]*strength[q]*strength[q];
      noddi_table[q*10] = (separation[q]-separation[q]/3)*modq;
      noddi_table[q*10+1] = parallel[q]; noddi_table[q*10+2] = iso[q];
      for (int j = 0; j < 7; ++j) { noddi_table[q*10+3+j] = integrals(q,j); }
      }
    }

  /* ---- seeds ---- */
  std::vector<double> sd_pos((size_t)S * 3), sd_state((size_t)S * n),
    sd_cov((size_t)S * n * n), sd_fa(S), sd_fa2(S), sd_trace(S), sd_trace2(S),
    sd_dir((size_t)S * 3);
  for (int f = 0; f < S; ++f)
    {
    const SeedPointInfo& si = seeds[f];
    for (int c = 0; c < 3; ++c)
      {
      sd_pos[(size_t)f * 3 + c] = si.point[c];
      sd_dir[(size_t)f * 3 + c] = si.start_dir[c];
      }
    for (int d = 0; d < n; ++d)
      {
      sd_state[(size_t)f * n + d] = si.state[d];
      }
    for (int a = 0; a < n; ++a)
      {
      for (int b = 0; b < n; ++b)
        {
        sd_cov[((size_t)f * n + a) * n + b] = si.covariance(a, b);
        }
      }
    sd_fa[f] = si.fa;
    sd_fa2[f] = si.fa2;
    sd_trace[f] = si.trace;
    sd_trace2[f] = si.trace2;
    }

  const auto staging_start = std::chrono::steady_clock::now();
  /* ---- output staging ---- */
  const size_t total_rec = (size_t)S * P.max_rec;
  std::vector<int> reccount(S, 0);
  auto allocate = [](size_t count) { return std::unique_ptr<double[]>(count ? new double[count] : nullptr); };
  auto rec_pos = allocate(total_rec*3), rec_norm = allocate(total_rec), rec_state = allocate(total_rec*n);
  auto rec_nmse = allocate(P.record_nmse ? total_rec : 0);
  auto rec_trace = allocate(P.record_trace ? total_rec : 0), rec_trace2 = allocate(P.record_trace ? total_rec : 0);
  auto rec_fa = allocate(P.record_fa ? total_rec : 0), rec_fa2 = allocate(P.record_fa ? total_rec : 0);
  auto rec_cov = allocate(P.record_cov ? total_rec*n*n : 0);
  const int host_workers = S < 1024 ? 1 : (int)std::min<size_t>(8, std::max<size_t>(1, s.num_threads));
  // Touch output pages in parallel. CUDA overwrites all downloaded rows;
  // only recorded rows are read, so zero-initializing every value is unnecessary.
  const std::array<std::pair<double*, size_t>, 9> outputs = {{{rec_pos.get(), total_rec*3},
    {rec_norm.get(), total_rec}, {rec_state.get(), total_rec*n},
    {rec_nmse.get(), P.record_nmse ? total_rec : 0},
    {rec_trace.get(), P.record_trace ? total_rec : 0}, {rec_trace2.get(), P.record_trace ? total_rec : 0},
    {rec_fa.get(), P.record_fa ? total_rec : 0}, {rec_fa2.get(), P.record_fa ? total_rec : 0},
    {rec_cov.get(), P.record_cov ? total_rec*n*n : 0}}};
  auto touch = [&](int t)
    {
    for (const auto& output : outputs)
      {
      const size_t lo = output.second*t/host_workers, hi = output.second*(t+1)/host_workers;
      for (size_t i = lo; i < hi; i += 4096/sizeof(double)) { output.first[i] = 0; }
      if (hi > lo) { output.first[hi-1] = 0; }
      }
    };
  std::vector<std::thread> touch_pool;
  for (int t = 1; t < host_workers; ++t) { touch_pool.emplace_back(touch, t); }
  touch(0);
  for (auto& th : touch_pool) { th.join(); }
  if (getenv("UKF_GPU_PROFILE"))
    {
    fprintf(stderr, "[GPU host] staging=%.3f s\n",
      std::chrono::duration<double>(std::chrono::steady_clock::now()-staging_start).count());
    }
  const std::size_t dwi_elems = (std::size_t)P.nx * P.ny * P.nz * G;

  /* ---- distribute seeds across GPUs (seeds are embarrassingly parallel).
   *      gpuDevice < 0 -> auto-select devices with enough free memory
   *      (shared-machine friendly); >= 0 -> that single device. ---- */
  int ndev = 1;
  cuda_count_devices(ndev);
  if (ndev < 1) { err = "no CUDA device"; return false; }
  if (s.gpu_device >= ndev) { err = "gpuDevice index out of range"; return false; }
  std::vector<int> devices;
  if (s.gpu_device >= 0)
    {
    devices.push_back(s.gpu_device);
    }
  else if (const char* requested = std::getenv("UKF_GPU_DEVICES"))
    {
    std::istringstream input(requested);
    std::string token;
    try
      {
      while (std::getline(input, token, ','))
        {
        size_t used = 0;
        const int device = std::stoi(token, &used);
        if (used != token.size() || device < 0 || device >= ndev ||
            std::find(devices.begin(), devices.end(), device) != devices.end())
          { err = "Invalid or duplicate explicit GPU device"; return false; }
        devices.push_back(device);
        }
      }
    catch (const std::exception&) { err = "Invalid explicit GPU device list"; return false; }
    if (devices.empty()) { err = "No explicit GPU devices selected"; return false; }
    }
  else
    {
    /* keep devices that (a) have at least 2GB free and (b) can hold a
     * meaningful fiber batch; always keep at least the emptiest one */
    std::vector<std::pair<size_t, int>> free_sorted;   // (free bytes, device)
    for (int d = 0; d < ndev; ++d)
      {
      size_t f = 0, t = 0;
      cuda_device_mem_info(d, f, t);
      free_sorted.emplace_back(f, d);
      }
    std::sort(free_sorted.rbegin(), free_sorted.rend());
    for (auto& ft : free_sorted)
      {
      if (ft.first >= (size_t)2 << 30)
        {
        devices.push_back(ft.second);
        }
      }
    if (devices.empty())
      {
      devices.push_back(free_sorted.front().second);
      }
    }
  const int ngpu = (int)devices.size();
  if (getenv("UKF_GPU_PROGRESS")) { fprintf(stderr, "[UKF_PROGRESS_PASS] %d\n", S); }
  std::vector<std::string> errs(ngpu);
  std::vector<int> ok(ngpu, false);
  std::vector<std::vector<GPUBranchSeed>> events(ngpu);
  auto worker = [&](int t)
    {
    const int begin = (int)((std::int64_t)S * t / ngpu);
    const int end = (int)((std::int64_t)S * (t + 1) / ngpu);
    const int ns = end - begin;
    if (ns <= 0)
      {
      ok[t] = true;
      return;
      }
    const size_t rec_off = (size_t)begin * P.max_rec;
    fprintf(stderr, "[GPU split] worker %d -> device %d, seeds [%d,%d) of %d\n",
            t, devices[t], begin, end, S);
    ok[t] = gpu_compute(P,
                        signal_data->rawDwiData(),
                        signal_data->rawMaskData(), signal_data->rawStopData(),
                        signal_data->rawGmData(), signal_data->rawCsfData(),
                        grad_flat.data(), b_flat.data(), dwi_elems,
                        sd_pos.data() + (size_t)begin * 3,
                        sd_state.data() + (size_t)begin * n,
                        sd_cov.data() + (size_t)begin * n * n,
                        sd_fa.data() + begin, sd_fa2.data() + begin,
                        sd_trace.data() + begin, sd_trace2.data() + begin,
                        sd_dir.data() + (size_t)begin * 3, ns,
                        reccount.data() + begin,
                        rec_pos.get() + rec_off * 3, rec_norm.get() + rec_off,
                        rec_state.get() + rec_off * n,
                        P.record_cov ? rec_cov.get() + rec_off * n * n : NULL,
                        P.record_nmse ? rec_nmse.get() + rec_off : NULL,
                        P.record_trace ? rec_trace.get() + rec_off : NULL,
                        P.record_trace ? rec_trace2.get() + rec_off : NULL,
                        P.record_fa ? rec_fa.get() + rec_off : NULL,
                        P.record_fa ? rec_fa2.get() + rec_off : NULL,
                        devices[t], 0, errs[t], P.noddi ? noddi_table.data() : NULL, branching ? &events[t] : NULL);
    };
  if (ngpu == 1)
    {
    worker(0);
    }
  else
    {
    std::vector<std::thread> pool;
    for (int t = 0; t < ngpu; ++t)
      {
      pool.emplace_back(worker, t);
      }
    for (auto& th : pool)
      {
      th.join();
      }
    }
  for (int t = 0; t < ngpu; ++t)
    {
    if (!ok[t])
      {
      err = errs[t].empty() ? "gpu worker failed" : errs[t];
      return false;
      }
    }

  if (branching)
    {
    std::vector<GPUBranchSeed> merged;
    for (int t = 0; t < ngpu; ++t)
      {
      const int begin = (int)((std::int64_t)S*t/ngpu);
      for (auto& event : events[t]) { event.fiber += begin; }
      merged.insert(merged.end(),events[t].begin(),events[t].end());
      }
    const int workers = std::min<int>(s.num_threads,S);
    std::sort(merged.begin(),merged.end(),[&](const GPUBranchSeed& a,const GPUBranchSeed& b)
      {
      if (a.fiber%workers != b.fiber%workers) { return a.fiber%workers < b.fiber%workers; }
      if (a.fiber != b.fiber) { return a.fiber < b.fiber; }
      if (a.step != b.step) { return a.step < b.step; }
      return a.tensor < b.tensor;
      });
    for (const auto& event : merged)
      {
      branch_seeds->emplace_back(); SeedPointInfo& seed = branch_seeds->back();
      seed.state.resize(n); seed.covariance.resize(n,n);
      for (int d = 0; d < n; ++d)
        { seed.state[d] = event.state[d]; for (int c = 0; c < n; ++c) { seed.covariance(d,c) = event.cov[d*n+c]; } }
      for (int c = 0; c < 3; ++c) { seed.point[c]=event.pos[c]; seed.start_dir[c]=event.direction[c]; }
      seed.fa=event.fa;
      BranchingSeedAffiliation affiliation; affiliation.fiber_index_=event.fiber; affiliation.position_on_fiber_=(event.step+1)/P.steps_per_record-1/P.steps_per_record;
      affiliations->push_back(affiliation);
      }
    fprintf(stderr,"[GPU branches] generated=%zu\n",merged.size());
    }

  const auto unpack_start = std::chrono::steady_clock::now();
  /* ---- unpack into UKFFiber (mirrors Record()/Follow*T) ---- */
  fibers.resize(S);
  const bool record_trace2 = P.record_trace && s.num_tensors >= 2;
  const bool record_fa2 = P.record_fa && s.num_tensors >= 2;
  const int unpack_workers = host_workers;
  auto unpack = [&](int unpack_worker)
    {
    const int begin = (int)((std::int64_t)S * unpack_worker / unpack_workers);
    const int end = (int)((std::int64_t)S * (unpack_worker + 1) / unpack_workers);
    for (int f = begin; f < end; ++f)
    {
    UKFFiber& fiber = fibers[f];
    const int r = reccount[f];
    fiber.position.reserve(r); fiber.norm.reserve(r); fiber.state.reserve(r);
    if (P.record_nmse) { fiber.normMSE.reserve(r); }
    if (P.record_trace) { fiber.trace.reserve(r); if (record_trace2) { fiber.trace2.reserve(r); } }
    if (P.record_fa) { fiber.fa.reserve(r); if (record_fa2) { fiber.fa2.reserve(r); } }
    if (P.record_cov) { fiber.covariance.reserve(r); }
    for (int i = 0; i < r; ++i)
      {
      const size_t idx = (size_t)f * P.max_rec + i;
      vec3_t p;
      p << rec_pos[idx * 3 + 0], rec_pos[idx * 3 + 1], rec_pos[idx * 3 + 2];
      fiber.position.push_back(p);
      fiber.norm.push_back(rec_norm[idx]);
      if (s.record_nmse)
        {
        fiber.normMSE.push_back(rec_nmse[idx]);
        }
      if (P.record_trace)
        {
        fiber.trace.push_back(P.noddi ? 2*(atan(1/rec_trace[idx])/3.14) : rec_trace[idx] / 1.0e6);
        if (record_trace2)
          {
          fiber.trace2.push_back(P.noddi ? 2*(atan(1/rec_trace2[idx])/3.14) : rec_trace2[idx] / 1.0e6);
          }
        }
      if (P.record_fa)
        {
        fiber.fa.push_back(rec_fa[idx]);
        if (record_fa2)
          {
          fiber.fa2.push_back(rec_fa2[idx]);
          }
        }
      fiber.state.emplace_back(n);
      State& st = fiber.state.back();
      for (int d = 0; d < n; ++d)
        {
        st[d] = rec_state[idx * n + d];
        }
      if (s.record_Viso) { fiber.free_water.push_back(std::max(0.0, st[n-1])); }
      if (s.record_free_water) { fiber.free_water.push_back(std::max(0.0, 1-st[n-1])); }
      if (s.record_cov)
        {
        fiber.covariance.emplace_back(n, n);
        ukfMatrixType& cov = fiber.covariance.back();
        for (int a = 0; a < n; ++a)
          {
          for (int b = 0; b < n; ++b)
            {
            cov(a, b) = rec_cov[idx * n * n + a * n + b];
            }
          }
        }
      }
    }
    };
  std::vector<std::thread> unpack_pool;
  for (int t = 1; t < unpack_workers; ++t) { unpack_pool.emplace_back(unpack, t); }
  unpack(0);
  for (auto& th : unpack_pool) { th.join(); }
  if (getenv("UKF_GPU_PROFILE"))
    {
    fprintf(stderr, "[GPU host] unpack=%.3f s\n",
      std::chrono::duration<double>(std::chrono::steady_clock::now()-unpack_start).count());
    }
  return true;
}

}
}
