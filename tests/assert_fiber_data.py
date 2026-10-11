#!/usr/bin/env python3
"""Require identical VTK geometry, connectivity and every recorded point array."""
import sys

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy


def read(path):
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(path)
    reader.ReadAllScalarsOn()
    reader.ReadAllVectorsOn()
    reader.ReadAllTensorsOn()
    reader.Update()
    data = reader.GetOutput()
    if data.GetPoints() is None:
        raise ValueError(f"No points in {path}")
    arrays = {"points": vtk_to_numpy(data.GetPoints().GetData()),
              "lines": vtk_to_numpy(data.GetLines().GetData())}
    for i in range(data.GetPointData().GetNumberOfArrays()):
        array = data.GetPointData().GetArray(i)
        arrays["point:" + array.GetName()] = vtk_to_numpy(array)
    return arrays


def main():
    left, right = map(read, sys.argv[1:3])
    if left.keys() != right.keys():
        raise AssertionError(f"Array names differ: {left.keys()} vs {right.keys()}")
    for name, reference in left.items():
        actual = right[name]
        if not np.isfinite(actual).all():
            raise AssertionError(f"{name}: non-finite output")
        if not np.array_equal(reference, actual):
            raise AssertionError(f"{name}: arrays differ ({reference.shape} vs {actual.shape})")
    print("PASS: geometry, connectivity and all recorded arrays are identical")


if __name__ == "__main__":
    main()
