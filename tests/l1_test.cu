/*
 * Usage: l1_test [model: 0=1T, 1=2T] [precision: 0=FP64, 1=mixed FP32]
 * L1 consistency test: GPU single-fiber pipeline vs an Eigen reference that
 * is a direct copy of the CPU code (NrrdData::Interp3Signal + the body of
 * UnscentedKalmanFilter::Filter + Step1T/2T tails).
 *
 * Build:
 *   nvcc -w -std=c++11 -O2 -fmad=false -arch=sm_89 \
 *        -I<ukf/gpu> -I/usr/include/eigen3 \
 *        tests/l1_test.cu ukf/gpu/gpu_kernels.cu -o l1_test
 */
#include <Eigen/Dense>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>

#include "gpu_device_api.h"

typedef double ukfPrecisionType;
typedef Eigen::Matrix<double, Eigen::Dynamic, 1> ukfVectorType;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> ukfMatrixType;

/* ------- direct copy of linalg.h helpers (double) ------- */
struct V3 { double a[3]; };
static void initNormalized(V3& m, double a, double b, double c) { m.a[0]=a; m.a[1]=b; m.a[2]=c; double n=sqrt(m.a[0]*m.a[0]+m.a[1]*m.a[1]+m.a[2]*m.a[2]); m.a[0]/=n; m.a[1]/=n; m.a[2]/=n; }

static Eigen::Matrix3d diffusion(const V3& m, double l1, double l2)
{
  Eigen::Matrix3d R;
  R << m.a[0], m.a[1], m.a[2],
    m.a[1], m.a[1]*m.a[1]/(1+m.a[0]) - 1, m.a[1]*m.a[2]/(1+m.a[0]),
    m.a[2], m.a[1]*m.a[2]/(1+m.a[0]), m.a[2]*m.a[2]/(1+m.a[0]) - 1;
  Eigen::DiagonalMatrix<double,3> lam; lam.diagonal() << l1, l2, l2;
  return R * lam * R.transpose() * 1e-6;
}

/* ------- direct copy of NrrdData::Interp3Signal (G gradients) ------- */
static void interp3(const double* pos, const float* data, int nx, int ny, int nz, int G,
                    double vx, double vy, double vz, double sigma, ukfVectorType& signal)
{
  if (sigma == 0) { sigma = std::min(std::min(vx, vy), vz); }
  double w_sum = 1e-16;
  const int step1 = nz * ny * G, step2 = nz * G;
  for (int i = 0; i < 2*G; ++i) { signal[i] = 0.0; }
  for (int xx = -1; xx <= 1; ++xx)
    {
    const int x = (int)round(pos[0]) + xx;
    if (x < 0 || nx <= x) { continue; }
    const double dx = (x - pos[0]) * vx; const double dxx = dx*dx;
    for (int yy = -1; yy <= 1; ++yy)
      {
      const int y = (int)round(pos[1]) + yy;
      if (y < 0 || ny <= y) { continue; }
      const double dy = (y - pos[1]) * vy; const double dyy = dy*dy;
      for (int zz = -1; zz <= 1; ++zz)
        {
        const int z = (int)round(pos[2]) + zz;
        if (z < 0 || nz <= z) { continue; }
        const double dz = (z - pos[2]) * vz; const double dzz = dz*dz;
        const double w = exp(-(dxx+dyy+dzz)/sigma);
        for (int i = 0; i < G; ++i)
          {
          signal[i] += w * data[step1*x + step2*y + z*G + i];
          }
        w_sum += w;
        }
      }
    }
  for (int i = 0; i < G; ++i) { signal[i] /= w_sum; signal[i+G] = signal[i]; }
}

/* ------- direct copy of UnscentedKalmanFilter (weights/SigmaPoints/Filter) ------- */
struct RefUKF
{
  int dim, signal_dim;
  double scale;
  ukfVectorType W;
  ukfMatrixType Wrep;
  ukfMatrixType Q;
  double Rs;
  int model;   // 0 = Simple1T, 1 = Simple2T
  RefUKF(int dim_, int signal_dim_, int model_, double Qm, double Ql, double Rs_)
    : dim(dim_), signal_dim(signal_dim_), model(model_), Rs(Rs_)
  {
    scale = sqrt(dim + 0.01);
    W.resize(2*dim+1);
    Wrep.resize(2*dim+1, 2*dim+1);
    Wrep.setConstant(0.0);
    W(0) = 0.01 / (dim + 0.01);
    Wrep(0,0) = W(0);
    for (unsigned i = 1; i < 2*dim+1; ++i)
      {
      W(i) = 0.5 / (dim + 0.01);
      Wrep(i,i) = W(i);
      }
    Q.resize(dim, dim);
    Q.setConstant(0.0);
    if (model == 0)
      {
      Q(0,0)=Q(1,1)=Q(2,2)=Qm; Q(3,3)=Q(4,4)=Ql;
      }
    else
      {
      Q(0,0)=Q(1,1)=Q(2,2)=Q(5,5)=Q(6,6)=Q(7,7)=Qm;
      Q(3,3)=Q(4,4)=Q(8,8)=Q(9,9)=Ql;
      }
  }

  void F(ukfMatrixType& X) const
  {
    for (unsigned i = 0; i < X.cols(); ++i)
      {
      double ninv = X(0,i)*X(0,i) + X(1,i)*X(1,i) + X(2,i)*X(2,i);
      ninv = 1.0/sqrt(ninv);
      X(0,i)*=ninv; X(1,i)*=ninv; X(2,i)*=ninv;
      X(3,i) = std::max(X(3,i), 100.0);
      X(4,i) = std::max(X(4,i), 100.0);
      if (model == 1)
        {
        ninv = X(5,i)*X(5,i) + X(6,i)*X(6,i) + X(7,i)*X(7,i);
        ninv = 1.0/sqrt(ninv);
        X(5,i)*=ninv; X(6,i)*=ninv; X(7,i)*=ninv;
        X(8,i) = std::max(X(8,i), 100.0);
        X(9,i) = std::max(X(9,i), 100.0);
        }
      }
  }

  void H(const ukfMatrixType& X, ukfMatrixType& Y, const std::vector<V3>& grads, const ukfVectorType& b) const
  {
    for (unsigned i = 0; i < X.cols(); ++i)
      {
      V3 m1; initNormalized(m1, X(0,i), X(1,i), X(2,i));
      double l10 = std::max(X(3,i), 100.0), l11 = std::max(X(4,i), 100.0);
      if (m1.a[0] < 0) { m1.a[0]=-m1.a[0]; m1.a[1]=-m1.a[1]; m1.a[2]=-m1.a[2]; }
      const Eigen::Matrix3d D1 = diffusion(m1, l10, l11);
      Eigen::Matrix3d D2;
      double w1v = 1.0, w2v = 0.0;
      if (model == 1)
        {
        V3 m2; initNormalized(m2, X(5,i), X(6,i), X(7,i));
        const double l20 = std::max(X(8,i), 100.0), l21 = std::max(X(9,i), 100.0);
        if (m2.a[0] < 0) { m2.a[0]=-m2.a[0]; m2.a[1]=-m2.a[1]; m2.a[2]=-m2.a[2]; }
        D2 = diffusion(m2, l20, l21);
        w1v = 0.5; w2v = 0.5;
        }
      for (int j = 0; j < signal_dim; ++j)
        {
        const V3& u = grads[j];
        Eigen::Vector3d uu(u.a[0], u.a[1], u.a[2]);
        double v = exp(-b[j] * uu.dot(D1*uu)) * w1v;
        if (model == 1) { v += exp(-b[j] * uu.dot(D2*uu)) * w2v; }
        Y(j, i) = v;
        }
      }
  }

  void SigmaPoints(const ukfVectorType& x, const ukfMatrixType& p, ukfMatrixType& x_spread) const
  {
    Eigen::LLT<ukfMatrixType> llt(p);
    ukfMatrixType NewM = llt.matrixL();
    NewM *= scale;
    x_spread.col(0) = x;
    x_spread.block(0, 1, dim, dim) = x.replicate(1, dim) + NewM;
    x_spread.block(0, dim+1, dim, dim) = x.replicate(1, dim) - NewM;
  }

  void Filter(const ukfVectorType& x, const ukfMatrixType& p, const ukfVectorType& z,
              ukfVectorType& x_new, ukfMatrixType& p_new, double& nmse,
              const std::vector<V3>& grads, const ukfVectorType& b) const
  {
    ukfMatrixType X(dim, 2*dim+1);
    SigmaPoints(x, p, X);
    F(X);
    const ukfVectorType X_hat = X * W;
    const ukfMatrixType X_ = X - X_hat.replicate(1, 2*dim+1);
    p_new = X_ * Wrep * X_.transpose() + Q;
    const ukfMatrixType Yk = p_new.inverse();
    const ukfVectorType yhat = Yk * X_hat;
    ukfMatrixType Z(signal_dim, 2*dim+1);
    H(X, Z, grads, b);
    const ukfVectorType Z_hat = Z * W;
    const ukfMatrixType Z_ = Z - Z_hat.replicate(1, 2*dim+1);
    const double R = 1.0 / Rs;
    const ukfMatrixType Pxz = X_ * Wrep * Z_.transpose();
    const ukfMatrixType Ht = Yk * Pxz;
    nmse = (z - Z_hat).squaredNorm() / z.squaredNorm();
    const ukfMatrixType I = Ht * R * Ht.transpose();
    const ukfMatrixType iv = Ht * R * ((z - Z_hat) + (Pxz.transpose() * yhat));
    p_new = (Yk + I).inverse();
    x_new = p_new * (iv + yhat);
  }
};

/* ---------------- test driver ---------------- */
int main(int argc, char** argv)
{
  const int model = (argc > 1) ? atoi(argv[1]) : 1;   // 0 or 1
  const int n = (model == 0) ? 5 : 10;
  const int G = (getenv("L1_G") ? atoi(getenv("L1_G")) : 15);  // gradients -> signal_dim 2G
  const int Q2 = 2 * G;
  const int nx = (getenv("L1_NX") ? atoi(getenv("L1_NX")) : 12);
  const int ny = (getenv("L1_NY") ? atoi(getenv("L1_NY")) : 11);
  const int nz = (getenv("L1_NZ") ? atoi(getenv("L1_NZ")) : 10);
  const int S = 3;                                      // test fibers
  const int max_length = (getenv("L1_MAX") ? atoi(getenv("L1_MAX")) : 40);
  const int steps_per_record = (getenv("L1_SPR") ? atoi(getenv("L1_SPR")) : 3);

  srand(7);
  /* synthetic dwi volume with smooth structure */
  std::vector<float> dwi((size_t)nx*ny*nz*G);
  for (int x = 0; x < nx; ++x)
    for (int y = 0; y < ny; ++y)
      for (int z = 0; z < nz; ++z)
        for (int g = 0; g < G; ++g)
          {
          dwi[((size_t)x*ny*nz + y*nz + z)*G + g] =
            100.0f + 30.0f*sin(0.3*x + 0.1*g) + 20.0f*cos(0.2*y + 0.05*g) + 10.0f*sin(0.4*z);
          }
  std::vector<unsigned char> maskv((size_t)nx*ny*nz, 1);
  std::vector<double> grads(Q2*3), bvals(Q2);
  for (int j = 0; j < G; ++j)
    {
    double gx = sin(0.7*j+0.3), gy = sin(1.3*j+1.1), gz = cos(0.5*j+0.7);
    double nn = sqrt(gx*gx+gy*gy+gz*gz);
    grads[(size_t)j*3+0] = gx/nn; grads[(size_t)j*3+1] = gy/nn; grads[(size_t)j*3+2] = gz/nn;
    bvals[j] = 1000.0;
    }
  for (int j = 0; j < G; ++j)
    {
    for (int c = 0; c < 3; ++c) { grads[(size_t)(j+G)*3+c] = -grads[(size_t)j*3+c]; }
    bvals[j+G] = bvals[j];
    }

  /* seeds (deterministic) */
  std::vector<double> sd_pos(S*3), sd_state(S*n), sd_cov(S*n*n), sd_fa(S), sd_fa2(S),
    sd_tr(S), sd_tr2(S), sd_dir(S*3);
  for (int f = 0; f < S; ++f)
    {
    sd_pos[f*3+0] = nx/2.0 + 0.27*f; sd_pos[f*3+1] = ny/2.0 + 0.13*f; sd_pos[f*3+2] = nz/2.0 - 0.21*f;
    for (int d = 0; d < n; ++d)
      {
      sd_state[(size_t)f*n+d] = (d < 3) ? (0.5 + 0.1*f) : (300.0 + 50.0*d + 10.0*f);
      }
    for (int i = 0; i < n*n; ++i) { sd_cov[(size_t)f*n*n+i] = 0.0; }
    for (int d = 0; d < n; ++d) { sd_cov[(size_t)f*n*n + d*n + d] = 0.01; }
    sd_fa[f] = 0.7; sd_fa2[f] = 0.5; sd_tr[f] = 2.1e6; sd_tr2[f] = 1.9e6;
    sd_dir[f*3+0] = 1.0; sd_dir[f*3+1] = 0.1; sd_dir[f*3+2] = -0.2;
    }

  GPUDarParams P;
  memset(&P, 0, sizeof(P));
  P.model = model; P.n = n; P.G = G;
  P.nx = nx; P.ny = ny; P.nz = nz;
  P.vx = 1.0; P.vy = 1.2; P.vz = 1.1;
  P.sigma_signal = 0.0;
  P.Qm = 0.003; P.Ql = (getenv("L1_QL") ? atof(getenv("L1_QL")) : 0.008); P.Rs = 100.0;
  P.w1 = P.w2 = (model == 0) ? 1.0 : 0.5;
  P.lambda_min = 100.0;
  P.stepLength = getenv("L1_STEP") ? atof(getenv("L1_STEP")) : 0.3;
  P.mean_signal_min = 0.0; P.fa_min = 0.0; P.min_radius = 0.0;   /* never stop by thresholds */
  P.max_length = max_length; P.steps_per_record = steps_per_record;
  P.record_nmse = 1; P.record_trace = 1; P.record_fa = 1; P.record_cov = 1;
  P.gm_thr = 0.99; P.csf_thr = 0.5; P.use_gmcsf = 0;
  P.use_float = argc > 2 ? atoi(argv[2]) : 0;
  P.num_stop_labels = 0;
  P.max_rec = max_length/steps_per_record + 2;
  P.mask_type = 2; P.stop_type = 0; P.gm_type = 0; P.csf_type = 0;
  P.step1 = (long)nz*ny*G; P.step2 = (long)nz*G;

  std::vector<int> reccount(S, 0);
  const size_t tr = (size_t)S * P.max_rec;
  // Canary tails detect a final partial chunk downloading CF rows instead of F.
  const double canary = -123456789.0;
  const size_t guard = 16;
  std::vector<double> rp(tr*3+guard, canary), rn(tr+guard, canary),
    rs(tr*n+guard, canary), rc(tr*n*n+guard, canary), rm(tr+guard, canary),
    rt(tr+guard, canary), rt2(tr+guard, canary), rf(tr+guard, canary), rf2(tr+guard, canary);
  std::string err;
  if (!gpu_compute(P, dwi.data(), maskv.data(), NULL, NULL, NULL,
                   grads.data(), bvals.data(), (long)dwi.size(),
                   sd_pos.data(), sd_state.data(), sd_cov.data(), sd_fa.data(), sd_fa2.data(),
                   sd_tr.data(), sd_tr2.data(), sd_dir.data(), S,
                   reccount.data(), rp.data(), rn.data(), rs.data(), rc.data(), rm.data(),
                   rt.data(), rt2.data(), rf.data(), rf2.data(), 0, 0, err))
    {
    printf("GPU FAILED: %s\n", err.c_str());
    return 1;
    }

  for (const auto* output : {&rp, &rn, &rs, &rc, &rm, &rt, &rt2, &rf, &rf2})
    {
    for (size_t i = output->size() - guard; i < output->size(); ++i)
      {
      if ((*output)[i] != canary) { printf("Host output tail overwritten\n"); return 1; }
      }
    }

  /* ---------------- CPU reference ---------------- */
  const std::vector<V3> gradsv = [&]()
    {
    std::vector<V3> v(Q2);
    for (int j = 0; j < Q2; ++j)
      {
      v[j].a[0] = grads[(size_t)j*3+0]; v[j].a[1] = grads[(size_t)j*3+1]; v[j].a[2] = grads[(size_t)j*3+2];
      }
    return v;
    }();
  ukfVectorType bref(Q2);
  for (int j = 0; j < Q2; ++j) { bref[j] = bvals[j]; }

  RefUKF ref(n, Q2, model, P.Qm, P.Ql, P.Rs);
  int mismatches = 0;
  double max_dx = 0.0, max_dp = 0.0, max_dn = 0.0;
  int total_rec = 0;
  for (int f = 0; f < S; ++f)
    {
    V3 x; x.a[0]=sd_pos[f*3+0]; x.a[1]=sd_pos[f*3+1]; x.a[2]=sd_pos[f*3+2];
    V3 m1; m1.a[0]=sd_dir[f*3+0]; m1.a[1]=sd_dir[f*3+1]; m1.a[2]=sd_dir[f*3+2];
    ukfVectorType state(n);
    for (int d = 0; d < n; ++d) { state[d] = sd_state[(size_t)f*n+d]; }
    ukfMatrixType p(n, n);
    for (int a = 0; a < n; ++a)
      for (int bb = 0; bb < n; ++bb)
        {
        p(a,bb) = sd_cov[((size_t)f*n+a)*n+bb];
        }
    double fa = sd_fa[f], fa2 = sd_fa2[f], trace = sd_tr[f], trace2 = sd_tr2[f], nmse = 0;
    /* records: CPU reference stores pos/state/cov/norm each steps_per_record */
    std::vector<V3> recp; std::vector<ukfVectorType> recs; std::vector<ukfMatrixType> recC;
    auto record = [&](void)
      {
      /* mirror Record(): normalize direction(s) for simple models */
      ukfVectorType st = state;
      {
      V3 mm; initNormalized(mm, st[0], st[1], st[2]);
      st[0]=mm.a[0]; st[1]=mm.a[1]; st[2]=mm.a[2];
      }
      if (n == 10)
        {
        V3 mm; initNormalized(mm, st[5], st[6], st[7]);
        st[5]=mm.a[0]; st[6]=mm.a[1]; st[7]=mm.a[2];
        }
      recp.push_back(x);
      recs.push_back(st);
      recC.push_back(p);
      };
    record();
    for (int step = 1; step <= max_length; ++step)
      {
      /* Step*T */
      ukfVectorType z(Q2);
      interp3(x.a, dwi.data(), nx, ny, nz, G, P.vx, P.vy, P.vz, P.sigma_signal, z);
      ukfVectorType x_new(n); ukfMatrixType p_new(n, n);
      ref.Filter(state, p, z, x_new, p_new, nmse, gradsv, bref);
      state = x_new; p = p_new;
      if (model == 0)
        {
        V3 mm; initNormalized(mm, state[0], state[1], state[2]);
        double l0 = std::max(state[3], 100.0), l1 = std::max(state[4], 100.0);
        trace = l0 + l1 + l1;
        fa = (l0 < l1) ? 0.0 : fabs(l0-l1)/sqrt(l0*l0+2.0*l1*l1);
        (void)fa2;
        m1 = mm;
        }
      else
        {
        /* full Step2T tail: State2Tensor2T + swap heuristics (tractography.cc) */
        const V3 old = m1;
        V3 mm1; initNormalized(mm1, state[0], state[1], state[2]);
        V3 mm2; initNormalized(mm2, state[5], state[6], state[7]);
        double l10 = std::max(state[3], 100.0), l11 = std::max(state[4], 100.0), l12 = l11;
        double l20 = std::max(state[8], 100.0), l21 = std::max(state[9], 100.0), l22 = l21;
        auto dot3 = [](const V3& a, const V3& b) { return a.a[0]*b.a[0] + a.a[1]*b.a[1] + a.a[2]*b.a[2]; };
        if (dot3(mm1, old) < 0) { mm1.a[0]=-mm1.a[0]; mm1.a[1]=-mm1.a[1]; mm1.a[2]=-mm1.a[2]; }
        if (dot3(mm2, old) < 0) { mm2.a[0]=-mm2.a[0]; mm2.a[1]=-mm2.a[1]; mm2.a[2]=-mm2.a[2]; }
        auto fa_simple = [](double a, double b) { return fabs(a-b)/sqrt(a*a+2.0*b*b); };
        trace = l10 + l11 + l12;
        trace2 = l20 + l21 + l22;
        double fa_t1 = fa_simple(l10, l11);
        double fa_t2 = fa_simple(l20, l21);
        const double ang = acos(fmax(-1.0, fmin(1.0, dot3(mm1, mm2)))) * (180.0 / 3.141592653589793238462643383279502884);
        auto swap_state = [&](void)
          {
          /* SwapState2T for even n: swap the two halves of state and cov blocks */
          const int h = n / 2;
          ukfVectorType ts(n);
          for (int a = 0; a < h; ++a) { ts[a] = state[a]; state[a] = state[h+a]; state[h+a] = ts[a]; }
          ukfMatrixType tp = p;
          for (int a = 0; a < h; ++a)
            for (int b = 0; b < h; ++b)
              {
              p(a,b) = tp(h+a, h+b);
              p(h+a, h+b) = tp(a,b);
              p(a,h+b) = tp(h+a, b);
              p(h+a,b) = tp(a, h+b);
              }
          };
        if (dot3(mm1, old) < dot3(mm2, old))
          {
          std::swap(mm1, mm2); std::swap(l10, l20); std::swap(l11, l21); std::swap(l12, l22);
          std::swap(fa_t1, fa_t2); std::swap(trace, trace2);
          swap_state();
          }
        if (ang <= 20.0 && std::min(fa_t1, fa_t2) <= 0.2 && fa_t1 <= 0.2)
          {
          std::swap(mm1, mm2); std::swap(l10, l20); std::swap(l11, l21); std::swap(l12, l22);
          swap_state();
          }
        if (l10 < l11 || l10 < l12) { fa = 0.0; fa2 = 0.0; }
        else { fa = fa_simple(l10, l11); fa2 = fa_simple(l20, l21); }
        m1 = mm1;
        }
      x.a[0] = x.a[0] + (m1.a[2] / P.vx) * P.stepLength;
      x.a[1] = x.a[1] + (m1.a[1] / P.vy) * P.stepLength;
      x.a[2] = x.a[2] + (m1.a[0] / P.vz) * P.stepLength;
      const int rx = (int)round(x.a[0]), ry = (int)round(x.a[1]), rz = (int)round(x.a[2]);
      if (rx < 0 || rx >= nx || ry < 0 || ry >= ny || rz < 0 || rz >= nz) { break; }
      if ((step + 1) % steps_per_record == 0) { record(); }
      }
    /* compare */
    for (size_t i = 0; i < recp.size() && i < (size_t)reccount[f]; ++i)
      {
      ++total_rec;
      const size_t gi = (size_t)f * P.max_rec + i;
      double dx = 0;
      for (int c = 0; c < 3; ++c)
        {
        if (!std::isfinite(rp[gi*3+c])) { ++mismatches; }
        dx = fmax(dx, fabs(rp[gi*3+c] - recp[i].a[c]));
        }
      max_dx = fmax(max_dx, dx);
      double ds = 0;
      for (int d = 0; d < n; ++d)
        {
        if (!std::isfinite(rs[gi*n+d])) { ++mismatches; }
        ds = fmax(ds, fabs(rs[gi*n+d] - recs[i][d]));
        }
      double dc = 0;
      for (int e = 0; e < n*n; ++e)
        {
        if (!std::isfinite(rc[gi*n*n+e])) { ++mismatches; }
        dc = fmax(dc, fabs(rc[gi*n*n+e] - recC[i](e/n, e%n)));
        }
      max_dp = fmax(max_dp, ds);
      max_dn = fmax(max_dn, dc);
      if (getenv("L1_VERBOSE") != NULL)
        {
        printf("f%d rec%02zu: dx=%.3e ds=%.3e dc=%.3e\n", f, i, dx, ds, dc);
        }
      const double pos_tol = P.use_float ? 1e-5 : 1e-6;
      const double state_tol = P.use_float ? 1e-4 : 1e-8;
      const double cov_tol = 1e-6;
      if (dx > pos_tol || ds > state_tol || dc > cov_tol)
        {
        ++mismatches;
        }
      }
    if (reccount[f] != (int)recp.size())
      {
      printf("f%d record count mismatch: gpu %d vs cpu %zu\n", f, reccount[f], recp.size());
      ++mismatches;
      }
    }
  printf("model=%s: %d fibers, %d records compared, mismatches=%d | max |dx|=%.3e max |dstate|=%.3e max |dcov|=%.3e\n",
         model == 0 ? "Simple1T" : "Simple2T", S, total_rec, mismatches, max_dx, max_dp, max_dn);
  return mismatches == 0 ? 0 : 1;
}
