// Compile with the backend's C++ compiler/flags, independently of nvcc/Eigen's
// __CUDACC__ vectorization choices. Produce the original CPU tensor reference.
#include <random>
#include "linalg.h"

extern "C" void tensor_reference(double* inputs, double* expected, int count)
{
  std::mt19937 random(47);
  std::normal_distribution<double> normal;
  for (int i = 0; i < count; ++i)
    {
    vec3_t m(normal(random), normal(random), normal(random));
    m.normalize();
    if (m[0] < 0) { m = -m; }
    diagmat3_t l;
    l.diagonal() << 1500+100*normal(random), 400+30*normal(random), 400+30*normal(random);
    const mat33_t D = diffusion(m, l);
    vec3_t u(normal(random), normal(random), normal(random));
    u.normalize();
    for (int c = 0; c < 3; ++c)
      { inputs[i*9+c] = m[c]; inputs[i*9+3+c] = l.diagonal()[c]; inputs[i*9+6+c] = u[c]; }
    for (int r = 0; r < 3; ++r)
      { for (int c = 0; c < 3; ++c) { expected[i*10+r*3+c] = D(r,c); } }
    expected[i*10+9] = std::exp(-1000*u.dot(D*u));
    }
}
