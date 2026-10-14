// Compare actual GPU helpers with the separately compiled upstream Eigen code.
#include <vector>
#include "../ukf/gpu/gpu_kernels.cu"
#define CUDA_CHECK(call) do { const auto status = (call); if (status != cudaSuccess) { fprintf(stderr, "%s\n", cudaGetErrorString(status)); return 2; } } while (0)
extern "C" void tensor_reference(double*, double*, int);

__global__ void tensors(const double* inputs, double* output, int count)
{
  const int i = blockIdx.x*blockDim.x+threadIdx.x;
  if (i >= count) { return; }
  const double* s = inputs+i*9;
  double* d = output+i*10;
  d_diffusion(s[0], s[1], s[2], s[3], s[4], s[5], d);
  const double b = 1000;
  d[9] = d_H_grad(UKF_GPU_MODEL_SIMPLE1T, d, d, 1.0, 0.0, s+6, &b, 0);
}

int main()
{
  const int count = 1000;
  std::vector<double> inputs(count*9), expected(count*10), actual(count*10);
  tensor_reference(inputs.data(), expected.data(), count);
  double *source, *result;
  CUDA_CHECK(cudaMalloc(&source, inputs.size()*sizeof(double)));
  CUDA_CHECK(cudaMalloc(&result, actual.size()*sizeof(double)));
  CUDA_CHECK(cudaMemcpy(source, inputs.data(), inputs.size()*sizeof(double), cudaMemcpyHostToDevice));
  tensors<<<(count+127)/128,128>>>(source, result, count);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(actual.data(), result, actual.size()*sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(source)); CUDA_CHECK(cudaFree(result));
  int matrix_errors = 0, signal_errors = 0;
  double max_signal_error = 0;
  for (int i = 0; i < count; ++i)
    {
    for (int c = 0; c < 9; ++c) { matrix_errors += expected[i*10+c] != actual[i*10+c]; }
    const double error = fabs(expected[i*10+9]-actual[i*10+9]);
    max_signal_error = fmax(max_signal_error,error);
    signal_errors += !std::isfinite(actual[i*10+9]) || error > 3e-16;
    }
  printf("CPU/GPU tensor matrices: %d unequal elements; signal max error %.17g (%d failures)\n",
         matrix_errors,max_signal_error,signal_errors);
  return matrix_errors || signal_errors ? 1 : 0;
}
