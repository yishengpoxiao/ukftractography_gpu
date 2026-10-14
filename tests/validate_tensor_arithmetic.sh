#!/usr/bin/env bash
# Use the same Eigen and host ISA as the native reference backend.
set -euo pipefail
cd "$(dirname "$0")/.."
out=${1:-/tmp/ukf-tensor-arithmetic}
mkdir -p "$out"
"${CXX:-c++}" -std=c++17 -O3 -march=corei7-avx -ffp-contract=off -Icommon \
  -I"${EIGEN_INCLUDE:-/usr/include/eigen3}" -c tests/tensor_reference.cc -o "$out/reference.o"
"${NVCC:-nvcc}" -std=c++17 -O2 -fmad=false -arch="${CUDA_TEST_ARCH:-sm_89}" -Icommon -Iukf/gpu \
  tests/tensor_arithmetic_test.cu "$out/reference.o" -o "$out/tensor-arithmetic"
"$out/tensor-arithmetic"
