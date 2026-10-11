import unittest
import numpy as np
from strict_geometry import compare


def fixture(points, counts):
    lines, offset = [], 0
    for count in counts:
        lines += [count]+list(range(offset, offset+count))
        offset += count
    return {'points': np.asarray(points, dtype=float), 'lines': np.asarray(lines, dtype=int)}


class StrictGeometryTests(unittest.TestCase):
    def test_euclidean_distance_rejects_small_component_errors(self):
        a = fixture([[0, 0, 0]], [1])
        b = fixture([[0.0008, 0.0008, 0]], [1])
        self.assertFalse(compare(a, b)['passed'])

    def test_threshold_includes_exactly_one_micron(self):
        a = fixture([[0, 0, 0]], [1])
        b = fixture([[0.001, 0, 0]], [1])
        self.assertTrue(compare(a, b)['passed'])

    def test_equal_total_points_does_not_hide_different_fiber_lengths(self):
        points = [[0, 0, 0]]*4
        result = compare(fixture(points, [1, 3]), fixture(points, [2, 2]))
        self.assertFalse(result['passed'])
        self.assertEqual(len(result['point_count_mismatches']), 2)

    def test_missing_fibers_and_empty_outputs_fail(self):
        a = fixture([[0, 0, 0], [1, 0, 0]], [1, 1])
        self.assertFalse(compare(a, fixture([[0, 0, 0]], [1]))['passed'])
        self.assertFalse(compare(fixture([], []), fixture([], []))['passed'])

    def test_nonfinite_and_first_deviation(self):
        a = fixture([[0, 0, 0], [1, 0, 0]], [2])
        self.assertFalse(compare(a, fixture([[0, 0, 0], [np.nan, 0, 0]], [2]))['passed'])
        result = compare(a, fixture([[0, 0, 0], [1.002, 0, 0]], [2]))
        self.assertEqual(result['position_mismatches'][0]['first_point'], 1)


if __name__ == '__main__':
    unittest.main()
