#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
compare_fibers.py — CPU vs GPU 输出一致性评估工具

读取两个 UKFTractography 输出的 legacy VTK polydata 纤维文件，按种子顺序逐条
比较：纤维条数、逐点数、位置偏差、标量数组（fa/trace/normMSE）偏差，以及
基于体素化的 tract 覆盖 Dice 重叠度。

用法：
  python compare_fibers.py ref.vtk test.vtk [--tol-pos 1e-3] [--verbose-limit 20]
"""
import argparse
import sys

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy


def load_fibers(path):
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(path)
    reader.ReadAllScalarsOn()
    reader.ReadAllVectorsOn()
    reader.ReadAllTensorsOn()
    reader.Update()
    pd = reader.GetOutput()

    pts = vtk_to_numpy(pd.GetPoints().GetData())
    lines = vtk_to_numpy(pd.GetLines().GetData())
    fibers = []
    i = 0
    while i < len(lines):
        n = lines[i]
        ids = lines[i + 1:i + 1 + n]
        fibers.append(pts[ids])
        i += 1 + n

    scalars = {}
    for k in range(pd.GetPointData().GetNumberOfArrays()):
        arr = pd.GetPointData().GetArray(k)
        scalars[arr.GetName()] = vtk_to_numpy(arr)
    return fibers, scalars


def voxelize(fibers, spacing=1.0):
    """把所有记录点 bin 到 spacing(mm) 网格，返回 dict[(i,j,k)] -> count"""
    grid = {}
    for f in fibers:
        idx = np.round(f / spacing).astype(np.int64)
        for ijk in idx:
            t = (int(ijk[0]), int(ijk[1]), int(ijk[2]))
            grid[t] = grid.get(t, 0) + 1
    return grid


def w_dice(ga, gb):
    """加权 Dice（权重为体素内点数）与普通 Dice（占用即 1）"""
    keys = set(ga) | set(gb)
    inter_w = sum(min(ga[k], gb[k]) for k in keys if k in ga and k in gb)
    sum_w = sum(ga.values()) + sum(gb.values())
    wdice = 2.0 * inter_w / sum_w if sum_w else 1.0
    occ_a = set(ga)
    occ_b = set(gb)
    inter = len(occ_a & occ_b)
    dice = 2.0 * inter / (len(occ_a) + len(occ_b)) if (occ_a or occ_b) else 1.0
    return wdice, dice


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("test")
    ap.add_argument("--tol-pos", type=float, default=1e-3, help="位置逐点容差 (mm)")
    ap.add_argument("--verbose-limit", type=int, default=10)
    args = ap.parse_args()

    fibers_a, scalars_a = load_fibers(args.ref)
    fibers_b, scalars_b = load_fibers(args.test)

    print(f"ref : {args.ref}  ->  {len(fibers_a)} fibers")
    print(f"test: {args.test}  ->  {len(fibers_b)} fibers")

    ok = True
    if len(fibers_a) != len(fibers_b):
        print(f"[MISMATCH] fiber count differs: {len(fibers_a)} vs {len(fibers_b)}")
        ok = False

    n = min(len(fibers_a), len(fibers_b))
    count_diff = 0
    pos_devs = []
    len_diff_fibers = []
    exact_len = 0
    for i in range(n):
        a = fibers_a[i]
        b = fibers_b[i]
        if a.shape != b.shape:
            count_diff += 1
            if count_diff <= args.verbose_limit:
                print(f"  fiber {i}: points {a.shape[0]} vs {b.shape[0]}")
            # compare the common prefix
            m = min(a.shape[0], b.shape[0])
        else:
            m = a.shape[0]
            exact_len += 1
        if m:
            d = np.linalg.norm(a[:m].astype(np.float64) - b[:m].astype(np.float64), axis=1).max()
            pos_devs.append(d)
        if a.shape != b.shape:
            len_diff_fibers.append(i)

    pos_devs = np.array(pos_devs) if pos_devs else np.array([0.0])
    print(f"point-count identical fibers : {exact_len}/{n} ({100.0*exact_len/max(n,1):.2f}%)")
    print(f"per-point position deviation : max={pos_devs.max():.3e} mean={pos_devs.mean():.3e} mm")
    print(f"fibers exceeding tol {args.tol_pos} mm : {(pos_devs > args.tol_pos).sum()}")

    for name in sorted(set(scalars_a) | set(scalars_b)):
        if name in scalars_a and name in scalars_b:
            a = scalars_a[name]
            b = scalars_b[name]
            m = min(len(a), len(b))
            if m:
                d = np.abs(np.asarray(a[:m], float) - np.asarray(b[:m], float))
                print(f"scalar '{name}' (first {m} pts): max|d|={d.max():.3e} mean|d|={d.mean():.3e}")
        else:
            print(f"scalar '{name}': missing in {'ref' if name not in scalars_a else 'test'}")

    ga = voxelize(fibers_a)
    gb = voxelize(fibers_b)
    wdice, dice = w_dice(ga, gb)
    print(f"tract overlap: wDice={wdice:.6f} Dice={dice:.6f}")

    if not ok or count_diff > 0 or (pos_devs > args.tol_pos).any():
        sys.exit(1)
    print("PASS: outputs are consistent within tolerance")


if __name__ == "__main__":
    main()
