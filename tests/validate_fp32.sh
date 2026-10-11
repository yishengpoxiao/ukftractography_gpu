#!/usr/bin/env bash
# Build the API-level FP64/mixed-FP32 regression and check partial chunks.
# Usage: tests/validate_fp32.sh [output-directory]
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:-/tmp/ukf-fp32-validation}
mkdir -p "$OUT"
NVCC=${NVCC:-nvcc}
EIGEN_INCLUDE=${EIGEN_INCLUDE:-/usr/include/eigen3}
"$NVCC" -w -std=c++11 -O2 -fmad=false -arch=sm_89 -I ukf/gpu -I "$EIGEN_INCLUDE" \
  tests/l1_test.cu ukf/gpu/gpu_kernels.cu -o "$OUT/l1_test"
for model in 0 1; do
  for precision in 0 1; do
    # Three seeds / capacity two exercises the final partial batch and canaries.
    UKF_GPU_CHUNK=2 "$OUT/l1_test" "$model" "$precision"
    L1_MAX=800 L1_SPR=1 L1_STEP=0.001 UKF_GPU_CHUNK=2 \
      "$OUT/l1_test" "$model" "$precision"
  done
done
# Exercise the cached 51-gradient update, including a partial final batch.
for precision in 0 1; do
  L1_G=51 L1_MAX=800 L1_SPR=3 L1_STEP=0.001 UKF_GPU_CHUNK=2 \
    "$OUT/l1_test" 1 "$precision"
done
if [[ ${SANITIZE:-0} == 1 ]]; then
  for tool in memcheck racecheck synccheck; do
    args=(--tool "$tool" --error-exitcode 99)
    if [[ $tool == memcheck ]]; then args+=(--leak-check full); fi
    L1_G=51 UKF_GPU_CHUNK=2 compute-sanitizer "${args[@]}" "$OUT/l1_test" 1 1
  done
fi
