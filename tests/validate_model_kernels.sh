#!/usr/bin/env bash
# Build CPU/CUDA model and QP comparisons; optional CUDA sanitizers.
set -euo pipefail
cd "$(dirname "$0")/.."
out=${1:-/tmp/ukf-model-kernels}
mkdir -p "$out"
nvcc=${NVCC:-nvcc}
flags=(-w -std=c++17 -O2 -fmad=false -arch=sm_89 -Iukf/gpu -Iukf -Icommon
       -I"${EIGEN_INCLUDE:-/usr/include/eigen3}"
       -I"${TEEM_INCLUDE:-../build-cpu/teem-build/include}")
"$nvcc" "${flags[@]}" tests/qp_gpu_test.cu ukf/QuadProg++_Eigen.cc -o "$out/qp_gpu_test"
"$nvcc" "${flags[@]}" tests/model_gpu_test.cu ukf/filter_*.cc \
  ukf/unscented_kalman_filter.cc ukf/utilities.cc ukf/QuadProg++_Eigen.cc -o "$out/model_gpu_test"
"$out/qp_gpu_test"
"$out/model_gpu_test"
if [[ ${SANITIZE:-0} == 1 ]]; then
  compute-sanitizer --tool memcheck --leak-check full --error-exitcode 99 "$out/qp_gpu_test"
  for tool in memcheck racecheck synccheck; do
    compute-sanitizer --tool "$tool" --error-exitcode 99 "$out/model_gpu_test"
  done
fi
