"""Regression fixtures for copied confidence/resource arithmetic; Slurm only."""
from copy import deepcopy
import os
from pathlib import Path
import sys
import unittest

if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURMD_NODENAME'):
    raise RuntimeError('A genuine Slurm compute allocation is required')
PACKAGE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PACKAGE / 'mcqa'), str(PACKAGE.parents[0])]
from metrics import summarize_extra
from comparison.study_metrics import calibration, correctness_auc, selective_curve, mean, quantile

CANONICAL = dict(calibration=calibration, correctness_auc=correctness_auc,
                 selective_curve=selective_curve, mean=mean, quantile=quantile)


class ConfidenceAndResources(unittest.TestCase):
    def setUp(self):
        self.rows = [
            dict(correct=True, confidence=.9, attempted=True, valid=True, status='ok', latency_s=.1, input_tokens=10, output_tokens=1),
            dict(correct=True, confidence=.8, attempted=True, valid=True, status='ok', latency_s=.2, input_tokens=20, output_tokens=2),
            dict(correct=False, confidence=.8, attempted=True, valid=True, status='ok', latency_s=None, input_tokens=None, output_tokens=None),
            dict(correct=False, confidence=.1, attempted=True, valid=True, status='ok', latency_s=0., input_tokens=0, output_tokens=0)]

    def test_binary_brier_and_auroc_ties(self):
        result = summarize_extra(self.rows, 'fixture', CANONICAL)
        self.assertAlmostEqual(result['binary_brier_selected_correctness'], .175)
        self.assertAlmostEqual(result['correctness_auroc'], .875)

    def test_selective_threshold_includes_the_entire_tie(self):
        points = summarize_extra(self.rows, 'fixture', CANONICAL)['selective_curve']
        self.assertEqual([point['accepted'] for point in points], [1, 3, 4])
        self.assertEqual([point['coverage'] for point in points], [.25, .75, 1.])

    def test_unknown_usage_stays_uncovered_and_known_zero_is_retained(self):
        resources = summarize_extra(self.rows, 'fixture', CANONICAL)['resources']
        self.assertEqual(resources['tokens']['input']['known_total'], 30)
        self.assertEqual(resources['tokens']['input']['coverage'], .75)
        self.assertEqual(resources['latency']['observed_n'], 3)

    def test_reference_zero_timings_are_placeholders(self):
        rows = [dict(row, latency_s=0.) for row in self.rows]
        latency = summarize_extra(rows, 'reference-uniform-seed0', CANONICAL)['resources']['latency']
        self.assertEqual(latency['placeholder_n'], 4)
        self.assertEqual(latency['observed_n'], 0)
        self.assertIsNone(latency['total_s'])

    def test_no_confidence_single_class_and_empty_have_no_auroc(self):
        for rows in ([], self.rows[:2], [dict(self.rows[0], confidence=None)]):
            self.assertIsNone(summarize_extra(rows, 'fixture', CANONICAL)['correctness_auroc'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
