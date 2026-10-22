"""Native adapters preserve actual distributions and unsupported-case semantics."""
import tempfile
import unittest
from unittest.mock import patch
from comparison.native import NativeBackend, distribution


class NativeTests(unittest.TestCase):
    def test_native_vector_requires_exact_labels_and_real_mass(self):
        self.assertEqual(distribution({'A': .25, 'B': .75}, ['B', 'A']), {'B': .75, 'A': .25})
        for probabilities in ({'A': 1.}, {'A': .2, 'B': .2}, {'A': float('nan'), 'B': 0.}):
            with self.subTest(probabilities=probabilities), self.assertRaises(ValueError):
                distribution(probabilities, ['A', 'B'])

    def test_native_method_and_menu_limits_never_load_or_shrink_the_model(self):
        with tempfile.TemporaryDirectory() as root, patch('comparison.native._allocation'):
            backend = NativeBackend({'family': 'apus', 'model_path': root})
            labels = [str(index) for index in range(17)]
            with patch.object(backend, '_load', side_effect=AssertionError('should not load')):
                row = backend.decide([], labels, option_descriptions={label: label for label in labels})
                self.assertEqual(row['status'], 'error')
                self.assertIn('at most 16', row['error'])
                row = backend.decide([], ['A', 'B'], option_descriptions={'A': 'A', 'B': 'B'}, method='sampling')
                self.assertEqual(row['status'], 'error')
                self.assertIsNone(row['probabilities'])

    def test_native_single_option_is_recorded_as_deterministic(self):
        with tempfile.TemporaryDirectory() as root, patch('comparison.native._allocation'):
            backend = NativeBackend({'family': 'openjev', 'model_path': root})
            with patch.object(backend, '_load', side_effect=AssertionError('should not load')):
                row = backend.decide([], ['A'], option_descriptions={'A': 'Only allowed action'})
            self.assertEqual(row['status'], 'ok')
            self.assertEqual(row['probabilities'], {'A': 1.})
            self.assertTrue(row['metadata']['single_option_deterministic'])

    def test_missing_criteria_is_an_error_instead_of_label_only_native_scoring(self):
        with tempfile.TemporaryDirectory() as root, patch('comparison.native._allocation'):
            backend = NativeBackend({'family': 'openjev', 'model_path': root})
            row = backend.decide([], ['A', 'B'])
            self.assertEqual(row['status'], 'error')
            self.assertIsNone(row['selected_option_id'])


if __name__ == '__main__':
    unittest.main()
