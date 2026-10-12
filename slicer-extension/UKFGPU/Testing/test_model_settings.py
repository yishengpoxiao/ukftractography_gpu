"""Model applicability and preset contracts, independent of Slicer/CUDA."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from UKFGPULib import MODELS, NUMERIC_OPTIONS, RECORD_OPTIONS, model_options, preset_values


class ModelSettingsTest(unittest.TestCase):
    def test_all_models_gate_inapplicable_parameters(self):
        for model, (tensors, flags) in MODELS.items():
            with self.subTest(model=model):
                numeric, records = model_options(model, True)
                noddi, water = '--noddi' in flags, '--freeWater' in flags
                self.assertEqual(set(numeric), set(NUMERIC_OPTIONS))
                self.assertEqual(set(records), set(RECORD_OPTIONS))
                self.assertEqual(numeric['Ql'], not noddi)
                self.assertEqual(numeric['Qvic'], noddi)
                self.assertEqual(numeric['Qw'], water)
                self.assertEqual(records['recordFreeWater'], water)
                self.assertEqual(records['recordTensors'], not noddi)
                self.assertEqual(records['recordKappa'], noddi)
                self.assertEqual(numeric['maxBranchingAngle'], tensors > 1 and not noddi)
                self.assertFalse(model_options(model, False)[0]['maxBranchingAngle'])

    def test_presets_keep_filter_defaults_and_valid_sampling(self):
        for model in MODELS:
            standard = preset_values('Standard', model)
            preview = preset_values('Quick preview', model)
            fine = preset_values('Fine sampling', model)
            self.assertEqual(set(standard), set(NUMERIC_OPTIONS))
            self.assertEqual(preview['maxHalfFiberLength'], 50)
            self.assertEqual(fine['stepLength'], standard['stepLength']/2)
            self.assertEqual(fine['recordLength'], standard['recordLength']/2)
            for values in (standard, preview, fine):
                self.assertGreaterEqual(values['recordLength'], values['stepLength'])
                self.assertEqual(values['maxBranchingAngle'], 0)
                for noise in ('Qm', 'Ql', 'Qw', 'Rs', 'Qkappa', 'Qvic'):
                    self.assertEqual(values[noise], 0)
        self.assertEqual(preset_values('Standard', 'Simple3T')['stepLength'], 0.15)


if __name__ == '__main__':
    unittest.main()
