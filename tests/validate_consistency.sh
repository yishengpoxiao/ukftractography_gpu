#!/bin/bash
# CPU vs GPU 一致性验证：复现仓库官方基线参数（generate_ground_truth.sh），
# 对每个用例分别跑 CPU 与 GPU，再用 compare_fibers.py 对比三组结果：
#   CPU vs 仓库自带基线、GPU vs CPU
set -e
UKF_BIN=${1:?usage: validate_consistency.sh /path/to/UKFTractography-with-GPU}
OUT=${2:-/data/yijie/ukf-GPU/validation}
PY=/data/yijie/miniconda3/envs/tractography/bin/python
DATA=UKFTractography/Data/Input
BASE=UKFTractography/Data/Baseline
mkdir -p "$OUT"
cd "$(dirname "$0")/.."   # repo root

run_case () {
  local name=$1; shift
  echo "=== case $name : CPU ==="
  /usr/bin/time -f "cpu wall %e s" "$UKF_BIN" "$@" --numThreads 1 --tracts "$OUT/${name}_cpu.vtk" > "$OUT/${name}_cpu.log" 2>&1 || { cat "$OUT/${name}_cpu.log"; return 1; }
  echo "=== case $name : GPU ==="
  /usr/bin/time -f "gpu wall %e s" "$UKF_BIN" "$@" --numThreads 1 --useGPU 1 --tracts "$OUT/${name}_gpu.vtk" > "$OUT/${name}_gpu.log" 2>&1 || { cat "$OUT/${name}_gpu.log"; return 1; }
  grep -h "wall" "$OUT/${name}_cpu.log" "$OUT/${name}_gpu.log"
  echo "=== case $name : GPU vs CPU ==="
  "$PY" tests/compare_fibers.py "$OUT/${name}_cpu.vtk" "$OUT/${name}_gpu.vtk"
}

# 官方 1T 参数（基线：$BASE/1T_fiber.vtk）
run_case 1T --dwiFile $DATA/single_tensor.nhdr --maskFile $DATA/mask.nhdr --seedsFile $DATA/seed.nhdr \
  --seedsPerVoxel 1 --numTensor 1 --minBranchingAngle 0.0 --maxBranchingAngle 0.0 --recordNMSE

# 官方 2T 参数（基线：$BASE/2T_fiber.vtk）
run_case 2T --dwiFile $DATA/two_tensor.nhdr --maskFile $DATA/mask.nhdr --seedsFile $DATA/seed.nhdr \
  --seedsPerVoxel 1 --numTensor 2 --minBranchingAngle 0.0 --maxBranchingAngle 0.0 --recordNMSE \
  --stoppingFA 0.10 --stoppingThreshold 0.05 --Qm 0.001 --Ql 10 --Rs 0.015

echo "=== reference: CPU vs shipped baseline (1T/2T) ==="
"$PY" tests/compare_fibers.py $BASE/1T_fiber.vtk "$OUT/1T_cpu.vtk" || true
"$PY" tests/compare_fibers.py $BASE/2T_fiber.vtk "$OUT/2T_cpu.vtk" || true
