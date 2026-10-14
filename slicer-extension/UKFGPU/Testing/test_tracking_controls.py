"""Input validation and progress/device contracts without a Slicer process."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from UKFGPULib import TrackingProgress, process_environment, validate_options


class TrackingControlsTest(unittest.TestCase):
    def test_invalid_sampling_and_thresholds(self):
        for options in ({'stepLength': 0}, {'Rs': float('nan')}, {'Ql': -1},
                        {'seedsPerVoxel': 1.5}, {'numThreads': 0}, {'stoppingFA': 1.1},
                        {'maxBranchingAngle': 91}, {'stepLength': .3, 'recordLength': .1},
                        {'minBranchingAngle': 20, 'maxBranchingAngle': 10}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                validate_options(options)
        validate_options({'stepLength': .15, 'recordLength': .45, 'Rs': 0,
                          'seedsPerVoxel': 1, 'minBranchingAngle': 20, 'maxBranchingAngle': 60})

    def test_fragmented_concurrent_progress_and_branch_pass(self):
        progress = TrackingProgress()
        progress.feed('[UKF_PROGRESS_PASS] 10\n[GPU split] device 0: 5 of 10\n[UKF_PROG')
        result = progress.feed('RESS] {"device":1,"completed":3,"total":5}\n'
                               '[UKF_PROGRESS] {"device":0,"completed":4,"total":5}\n')
        self.assertEqual((result['completed'], result['total'], result['pass_number']), (7, 10, 1))
        progress.feed('[UKF_PROGRESS] {"device":1,"completed":5,"total":5}\n')
        result = progress.feed('[UKF_PROGRESS_PASS] 4\n[UKF_PROGRESS] {"device":0,"completed":1,"total":2}\n')
        self.assertEqual((result['completed'], result['total'], result['pass_number']), (1, 4, 2))
        self.assertEqual(progress.feed('[phase] tractography: 1s\n')['stage'], 'Preparing fibers')

    def test_explicit_physical_gpus_replace_stale_child_settings(self):
        startup = dict(CUDA_VISIBLE_DEVICES='2,0', NVIDIA_VISIBLE_DEVICES='container',
                       UKF_GPU_DEVICES='9', UKF_GPU_PROGRESS='1')
        env = process_environment(startup, gpu_uuids=['GPU-a', 'GPU-b'], progress=True)
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], 'GPU-a,GPU-b')
        self.assertEqual(env['UKF_GPU_DEVICES'], '0,1')
        self.assertEqual(env['NVIDIA_VISIBLE_DEVICES'], 'container')
        self.assertEqual(startup['CUDA_VISIBLE_DEVICES'], '2,0')
        inherited = process_environment(startup)
        self.assertNotIn('UKF_GPU_DEVICES', inherited)
        self.assertNotIn('UKF_GPU_PROGRESS', inherited)


if __name__ == '__main__':
    unittest.main()
