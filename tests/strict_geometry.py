#!/usr/bin/env python3
"""Audit ordered fibers using Euclidean distances in millimetres."""
import argparse
import json

import numpy as np


def fibers(arrays):
    lines = arrays['lines']
    offset = 0
    while offset < len(lines):
        count = int(lines[offset])
        yield arrays['points'][lines[offset+1:offset+1+count]]
        offset += count+1


def compare(reference, actual, tolerance=0.001):
    left, right = list(fibers(reference)), list(fibers(actual))
    nonfinite = not np.isfinite(reference['points']).all() or not np.isfinite(actual['points']).all()
    counts, positions = [], []
    maximum = 0.0
    for index, (a, b) in enumerate(zip(left, right)):
        if len(a) != len(b):
            counts.append({'fiber': index, 'cpu_points': len(a), 'gpu_points': len(b)})
        common = min(len(a), len(b))
        distances = np.linalg.norm(a[:common].astype(np.float64)-b[:common].astype(np.float64), axis=1)
        if common and np.isfinite(distances).all():
            maximum = max(maximum, float(distances.max()))
            failed = np.flatnonzero(distances > tolerance)
            if len(failed):
                first = int(failed[0])
                positions.append({'fiber': index, 'first_point': first,
                                  'first_distance_mm': float(distances[first]),
                                  'max_distance_mm': float(distances.max())})
    passed = bool(left) and len(left) == len(right) and not counts and not positions and not nonfinite
    return {'passed': passed, 'tolerance_mm': tolerance, 'cpu_fibers': len(left),
            'gpu_fibers': len(right), 'nonfinite': nonfinite, 'max_distance_mm': maximum,
            'point_count_mismatches': counts, 'position_mismatches': positions}


def main():
    from assert_fiber_data import read
    parser = argparse.ArgumentParser()
    parser.add_argument('reference')
    parser.add_argument('actual')
    parser.add_argument('--output')
    args = parser.parse_args()
    result = compare(read(args.reference), read(args.actual))
    output = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        with open(args.output, 'w') as stream:
            stream.write(output+'\n')
    print(output)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
