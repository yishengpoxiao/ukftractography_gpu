// Fixed-capacity CUDA port of ukf/QuadProg++_Eigen.cc.
// Goldfarb-Idnani, Luca Di Gaspero (2007-2009), Eric Moyer (2009), LGPL-3.0-or-later.
// See the original file for attribution and license. Equality placeholders are ignored.
#ifndef UKF_GPU_QP_CUH
#define UKF_GPU_QP_CUH
#include <float.h>
template<int N>
struct QPVector
{
  int length;
  double values[N+2] = {};
  __device__ explicit QPVector(int n) : length(n) {}
  __device__ int size() const { return length; }
  __device__ double& operator[](int i) { return values[i]; }
  __device__ double operator[](int i) const { return values[i]; }
};
template<int N>
struct QPMatrix
{
  int width, height;
  double values[N*N] = {};
  __device__ QPMatrix(int n, int m) : width(m), height(n) {}
  __device__ int rows() const { return height; }
  __device__ double& operator()(int i, int j) { return values[i*width+j]; }
  __device__ double operator()(int i, int j) const { return values[i*width+j]; }
};
template<int N>
__device__ void qp_compute_d(QPVector<N>& d, const QPMatrix<N>& J, const QPVector<N>& np)
{
  const int    n = static_cast<int>(d.size());
  for( int c = 0; c < n; ++c )
    {
    double sum = 0.0;
    for( int r = 0; r < n; ++r )
      {
      sum += J(r,c) * np[r];
      }
    d[c] = sum;
    }
}
template<int N>
__device__ void qp_update_z(QPVector<N>& z, const QPMatrix<N>& J, const QPVector<N>& d, int iq)
{
  const int n = static_cast<int>(z.size());
  for( int i = 0; i < n; ++i )
    {
    z[i] = 0.0;
    for( int j = iq; j < n; ++j )
      {
      z[i] += J(i,j) * d[j];
      }
    }
}
template<int N>
__device__ void qp_update_r(const QPMatrix<N>& R, QPVector<N>& r, const QPVector<N>& d, int iq)
{
  for( int i = iq - 1; i >= 0; i-- )
    {
    double sum = 0.0;
    for( int j = i + 1; j < iq; ++j )
      {
      sum += R(i,j) * r[j];
      }
    r[i] = (d[i] - sum) / R(i,i);
    }
}
__device__ double qp_distance(double a, double b)
{
  const double a1 = fabs(a);
  const double b1 = fabs(b);
  if( a1 > b1 )
    {
    double t = (b1 / a1);
    return a1 * sqrt(1.0 + t * t);
    }
  else if( b1 > a1 )
    {
    double t = (a1 / b1);
    return b1 * sqrt(1.0 + t * t);
    }
  return a1 * sqrt(2.0);
}
template<int N>
__device__ bool qp_add_constraint(QPMatrix<N>& R, QPMatrix<N>& J, QPVector<N>& d, int& iq, double& R_norm)
{
  const int n = static_cast<int>(d.size());
  for( int j = n - 1; j >= iq + 1; j-- )
    {
    double cc = d[j - 1];
    double ss = d[j];
    double h = qp_distance(cc, ss);
    if( fabs(h) < DBL_EPSILON ) 
      {
      continue;
      }
    d[j] = 0.0;
    ss = ss / h;
    cc = cc / h;
    if( cc < 0.0 )
      {
      cc = -cc;
      ss = -ss;
      d[j - 1] = -h;
      }
    else
      {
      d[j - 1] = h;
      }
    double xny = ss / (1.0 + cc);
    for( int k = 0; k < n; k++ )
      {
      double t1 = J(k,j - 1);
      double t2 = J(k,j);
      J(k,j - 1) = t1 * cc + t2 * ss;
      J(k,j) = xny * (t1 + J(k,j - 1)) - t2;
      }
    }
  iq++;
  for( int i = 0; i < iq; ++i )
    {
    R(i,iq - 1) = d[i];
    }
  if( fabs(d[iq - 1]) <= DBL_EPSILON * R_norm )
    {
    return false;
    }
  R_norm = fmax(R_norm, fabs(d[iq - 1]) );
  return true;
}
template<int N>
__device__ void qp_delete_constraint(QPMatrix<N>& R, QPMatrix<N>& J, QPVector<N>& A, QPVector<N>& u, int n,
                       int p, int& iq, int l)
{
  int qq = -1; 
  for( int i = p; i < iq; ++i )
    {
    if( A[i] == l )
      {
      qq = i;
      break;
      }
    }
  for( int i = qq; i < iq - 1; ++i )
    {
    A[i] = A[i + 1];
    u[i] = u[i + 1];
    for( int j = 0; j < n; ++j )
      {
      R(j,i) = R(j,i + 1);
      }
    }
  A[iq - 1] = A[iq];
  u[iq - 1] = u[iq];
  A[iq] = 0;
  u[iq] = 0.0;
  for( int j = 0; j < iq; ++j )
    {
    R(j,iq - 1) = 0.0;
    }
  iq--;
  if( iq == 0 )
    {
    return;
    }
  for( int j = qq; j < iq; ++j )
    {
    double cc = R(j,j);
    double ss = R(j + 1,j);
    double h = qp_distance(cc, ss);
    if( fabs(h) < DBL_EPSILON ) 
      {
      continue;
      }
    cc = cc / h;
    ss = ss / h;
    R(j + 1,j) = 0.0;
    if( cc < 0.0 )
      {
      R(j,j) = -h;
      cc = -cc;
      ss = -ss;
      }
    else
      {
      R(j,j) = h;
      }
    double xny = ss / (1.0 + cc);
    for( int k =j + 1; k < iq; ++k )
      {
      double t1 = R(j,k);
      double t2 = R(j + 1,k);
      R(j,k) = t1 * cc + t2 * ss;
      R(j + 1,k) = xny * (t1 + R(j,k)) - t2;
      }
    for( int k = 0; k < n; ++k )
      {
      double t1 = J(k,j);
      double t2 = J(k,j + 1);
      J(k,j) = t1 * cc + t2 * ss;
      J(k,j + 1) = xny * (J(k,j) + t1) - t2;
      }
    }
}
template<int N>
__device__ void qp_cholesky_decomposition(QPMatrix<N>& A)
{
  const int n = static_cast<int>(A.rows());
  for( int i = 0; i < n; ++i )
    {
    for( int j = i; j < n; ++j )
      {
      double sum = A(i,j);
      for( int k =i - 1; k >= 0; k-- )
        {
        sum -= A(i,k) * A(j,k);
        }
      if( i == j )
        {
        if( sum <= 0.0 )
          {
          A(i,i) = NAN; return;
          }
        A(i,i) = sqrt(sum);
        }
      else
        {
        A(j,i) = sum / A(i,i);
        }
      }
    for( int k =i + 1; k < n; ++k )
      {
      A(i,k) = A(k,i);
      }
    }
}
template<int N>
__device__ void qp_forward_elimination(const QPMatrix<N>& L, QPVector<N>& y, const QPVector<N>& b)
{
  const int n = static_cast<int>(L.rows());
  y[0] = b[0] / L(0,0);
  for( int i = 1; i < n; ++i )
    {
    y[i] = b[i];
    for( int j = 0; j < i; ++j )
      {
      y[i] -= L(i,j) * y[j];
      }
    y[i] = y[i] / L(i,i);
    }
}
template<int N>
__device__ void qp_backward_elimination(const QPMatrix<N>& U, QPVector<N>& x, const QPVector<N>& y)
{
  const int n = static_cast<int>(U.rows());
  x[n - 1] = y[n - 1] / U(n - 1,n - 1);
  for( int i = n - 2; i >= 0; i-- )
    {
    x[i] = y[i];
    for( int j = i + 1; j < n; ++j )
      {
      x[i] -= U(i,j) * x[j];
      }
    x[i] = x[i] / U(i,i);
    }
}
template<int N>
__device__ void qp_cholesky_solve(const QPMatrix<N>& L, QPVector<N>& x, const QPVector<N>& b)
{
  const int n = static_cast<int>(L.rows());
  QPVector<N> y(n);
  qp_forward_elimination(L, y, b);
  qp_backward_elimination(L, x, y);
}
template<int N>
__device__ double qp_dot_product(const QPVector<N>& x, const QPVector<N>& y)
{
  int    n = static_cast<int>(x.size());
  double sum;
  sum = 0.0;
  for( int i = 0; i < n; ++i )
    {
    sum += x[i] * y[i];
    }
  return sum;
}
template<int N>
__device__ __noinline__ bool d_constrain(const GPUDarParams& P, double* values, const double* W)
{
  QPVector<N> x(P.n);
  bool violated = false;
  for (int i = 0; i < P.n; ++i) { x[i] = values[i]; }
  for (int c = 0; c < P.num_constraints; ++c)
    { violated = violated || -P.constraint_sign[c]*x[P.constraint_index[c]] > P.constraint_offset[c]; }
  if (!violated) { return true; }
  const bool valid = [&]() {
  const int n = P.n, p = 0, m = P.num_constraints;
  QPVector<N> g0(n), ci0(m);
  QPMatrix<N> CI(n, m), G(n,n);

  for (int i = 0; i < n; ++i)
    {
    for (int j = 0; j < n; ++j) { G(i,j) = 0.5*(W[i*n+j]+W[j*n+i]); }
    }
  for (int i = 0; i < n; ++i)
    {
    g0[i] = 0;
    for (int j = 0; j < n; ++j) { g0[i] -= G(j,i)*x[j]; }

    }
  for (int c = 0; c < m; ++c)
    {
    ci0[c] = P.constraint_offset[c];
    CI(P.constraint_index[c],c) = P.constraint_sign[c];
    }
  QPMatrix<N> R(n, n), J(n, n);
  QPVector<N> s(m + p), z(n), r(m + p), d(n), np(n), u(m + p), x_old(n), u_old(m + p);
  const double inf = INFINITY;
  QPVector<N>          A(m + p), A_old(m + p), iai(m + p);
  int                      iq, iter = 0;
  QPVector<N> iaexcl(m + p);
  double c1 = 0.0;
  for(int i = 0; i < n; ++i )
    {
    c1 += G(i,i);
    }
  qp_cholesky_decomposition(G);
  for( int i = 0; i < n; ++i )
    {
    d[i] = 0.0;
    for( int j = 0; j < n; ++j )
      {
      R(i,j) = 0.0;
      }
    }
  double R_norm = 1.0; 
  double c2 = 0.0;
  for( int i = 0; i < n; ++i )
    {
    d[i] = 1.0;
    qp_forward_elimination(G, z, d);
    for( int j = 0; j < n; ++j )
      {
      J(i,j) = z[j];
      }
    c2 += z[i];
    d[i] = 0.0;
    }
  qp_cholesky_solve(G, x, g0);
  for( int i = 0; i < n; ++i )
    {
    x[i] = -x[i];
    }
  double f_value = 0.5 * qp_dot_product(g0, x);
  iq = 0;
  for( int i = 0; i < m; ++i )
    {
    iai[i] = i;
    }
l1:
  if (++iter > 256) { return false; }
  for( int i = p; i < iq; ++i )
    {
    int ip = A[i];
    iai[ip] = -1;
    }
  double ss = 0.0;
  double psi = 0.0; 
  int ip = 0; 
  for( int i = 0; i < m; ++i )
    {
    iaexcl[i] = true;
    double sum = 0.0;
    for( int j = 0; j < n; ++j )
      {
      sum += CI(j,i) * x[j];
      }
    sum += ci0[i];
    s[i] = sum;
    psi += fmin(0.0, sum);
    }
  if( fabs(psi) <= m * DBL_EPSILON * c1 * c2 * 100.0 )
    {
    return isfinite(f_value) && f_value <= 0.01;
    }
  for( int i = 0; i < iq; ++i )
    {
    u_old[i] = u[i];
    A_old[i] = A[i];
    }
  for( int i = 0; i < n; ++i )
    {
    x_old[i] = x[i];
    }
l2: 
  for( int i = 0; i < m; ++i )
    {
    if( s[i] < ss && iai[i] != -1 && iaexcl[i] )
      {
      ss = s[i];
      ip = i;
      }
    }
  if( ss >= 0.0 )
    {
    return isfinite(f_value) && f_value <= 0.01;
    }
  for( int i = 0; i < n; ++i )
    {
    np[i] = CI(i,ip);
    }
  u[iq] = 0.0;
  A[iq] = ip;
l2a: 
  qp_compute_d(d, J, np);
  qp_update_z(z, J, d, iq);
  qp_update_r(R, r, d, iq);
  unsigned int l = 0;
  double t1 = inf; 
  for( int k = p; k < iq; k++ )
    {
    if( r[k] > 0.0 )
      {
      if( u[k] / r[k] < t1 )
        {
        t1 = u[k] / r[k];
        l = A[k];
        }
      }
    }
  double t2 = inf;
  if( fabs(qp_dot_product(z, z) )  > DBL_EPSILON ) 
    {
    t2 = -s[ip] / qp_dot_product(z, np);
    }
  double t = fmin(t1, t2);
  if( t >= inf )
    {
    return false;
    }
  if( t2 >= inf )
    {
    for( int k = 0; k < iq; k++ )
      {
      u[k] -= t * r[k];
      }
    u[iq] += t;
    iai[l] = l;
    qp_delete_constraint(R, J, A, u, n, p, iq, l);
    goto l2a;
    }
  for( int k = 0; k < n; k++ )
    {
    x[k] += t * z[k];
    }
  f_value += t * qp_dot_product(z, np) * (0.5 * t + u[iq]);
  for( int k = 0; k < iq; k++ )
    {
    u[k] -= t * r[k];
    }
  u[iq] += t;
  if( fabs(t - t2) < DBL_EPSILON )
    {
    if( !qp_add_constraint(R, J, d, iq, R_norm) )
      {
      iaexcl[ip] = false;
      qp_delete_constraint(R, J, A, u, n, p, iq, ip);
      for( int i = 0; i < m; ++i )
        {
        iai[i] = i;
        }
      for( int i = p; i < iq; ++i )
        {
        A[i] = A_old[i];
        u[i] = u_old[i];
        iai[A[i]] = -1;
        }
      for( int i = 0; i < n; ++i )
        {
        x[i] = x_old[i];
        }
      goto l2; 
      }
    else
      {
      iai[ip] = -1;
      }
    goto l1;
    }
  iai[l] = l;
  qp_delete_constraint(R, J, A, u, n, p, iq, l);
  double sum = 0.0;
  for( int k = 0; k < n; k++ )
    {
    sum += CI(k,ip) * x[k];
    }
  s[ip] = sum + ci0[ip];
  goto l2a;
}();
  if (valid) { for (int i = 0; i < P.n; ++i) { values[i] = x[i]; } }
  return valid;
}

#endif
