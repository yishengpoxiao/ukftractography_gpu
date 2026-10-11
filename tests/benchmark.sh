#!/bin/bash
# UKFTractography CPU/GPU 双方案对照基准
# 用法: bash tests/benchmark.sh <GPU二进制> [工作目录]
# CPU 方案: 上游二进制多线程; GPU 方案: --useGPU 单卡
set -u
GPU_BIN=${1:?usage: benchmark.sh /path/to/UKFTractography_gpu [cpu_bin]}
CPU_BIN=${2:-/data/yijie/ukf-GPU/build-cpu/UKFTractography-build/UKFTractography/bin/UKFTractography}
cd "$(dirname "$0")/.." || exit 1
D=UKFTractography/Data/Input
OUT=${3:-/data/yijie/ukf-GPU/bench}
mkdir -p "$OUT"
SEEDS=1; NT=2
COMMON="--dwiFile $D/dwi.nhdr --maskFile $D/dwi-mask.nhdr --seedsPerVoxel $SEEDS --numTensor $NT --numThreads 1 --minBranchingAngle 0.0 --maxBranchingAngle 0.0 --stoppingFA 0.10 --stoppingThreshold 0.05 --Qm 0.001 --Ql 10 --Rs 0.015"

echo "== 方案一: CPU (多线程, 原版代码路径) =="
/usr/bin/time -f "CPU_WALL %e s" "$CPU_BIN" $COMMON --numThreads 16 --tracts "$OUT/bench_cpu.vtk" 2>&1 | grep -E "CPU_WALL"

echo "== 方案二: GPU (单卡, --useGPU) ="
/usr/bin/time -f "GPU_WALL %e s" "$GPU_BIN" $COMMON --useGPU --gpuDevice 0 --tracts "$OUT/bench_gpu.vtk" 2>&1 | grep -E "GPU_WALL"

echo "== 输出一致性 ="
$(dirname "$0")/../tests/compare_fibers.py "$OUT/bench_cpu.vtk" "$OUT/bench_gpu.vtk" 2>&1 | grep -E "point-count|wDice" || true
