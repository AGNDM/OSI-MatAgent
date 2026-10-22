"""Meaningful saved-report projection checks; run only through Slurm."""
from copy import deepcopy
import importlib.util
import os
from pathlib import Path
import unittest

if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURMD_NODENAME'):
    raise RuntimeError('A genuine Slurm compute allocation is required')

spec = importlib.util.spec_from_file_location('compact_status', Path(__file__).resolve().parents[1] / 'workflow/compact_status.py')
status = importlib.util.module_from_spec(spec)
spec.loader.exec_module(status)
STUDY = '/tmp/evaluation-compact-status-fixture'


def action_metrics(observed=0, correct=0, corrections=0, accuracy=None):
    return {'observed_original_proposals': observed, 'correct_original_proposals': correct,
        'original_proposal_accuracy_micro': accuracy, 'reference_corrections': corrections,
        'proposal_count_coverage': 1.0, 'correction_count_coverage': 1.0,
        'known_partial_observed_original_proposals': observed,
        'known_partial_reference_corrections': corrections}


def fixture():
    cells = []
    for index in range(22):
        first = index == 0
        cells.append({'experiment': 'cell-' + str(index), 'expert': 'expert', 'chooser': 'chooser',
            'batch': 'core' if index < 8 else 'support', 'mode': 'teacher_forced', 'seed': 0,
            'expected_cases': 120, 'episode_records': 2 if first else 0,
            'valid_native_episodes': int(first), 'verified_outcome_successes': int(first),
            'aborted_episodes': int(first), 'missing_episode_records': 118 if first else 120,
            'deadline_not_started': 0, 'unusable_records': 0,
            'native_score': {'coverage': 1/120 if first else 0},
            'original_action_metrics_all_closed_attempts': action_metrics(8, 3, 5, 0.375) if first else action_metrics(),
            'cases': [{'slurm': {'state': 'PENDING'}} for _ in range(120)]})
    cells[0]['cases'][:2] = [{'slurm': {'state': 'COMPLETED'}}] * 2
    cells[1]['cases'][0] = {'slurm': {'state': 'RUNNING'}}
    return {'schema_version': 1, 'study': STUDY, 'mode': 'teacher_forced', 'protocol': 'frozen-test-profile',
        'generated_utc': '2026-10-22T17:00:00+00:00', 'status': 'provisional_partial_coverage',
        'study_stopped': False, 'outcome_semantics': 'Assisted native goals; not autonomous success.',
        'cohort': {'mode': 'teacher_forced', 'seed': 0, 'cells': 22, 'cases_per_cell': 120,
            'declared_episode_count': 2640, 'development_smokes_included': False,
            'deadline_utc': '2026-10-22T20:43:00+00:00'},
        'progress': {'declared_episode_count': 2640, 'episode_records': 2, 'valid_native_episodes': 1,
            'verified_outcome_successes': 1, 'aborted_episodes': 1, 'missing_episode_records': 2638,
            'deadline_not_started': 0, 'unusable_records': 0, 'attempt_record_coverage': 2/2640,
            'valid_native_episode_coverage': 1/2640},
        'original_action_metrics_all_closed_attempts': action_metrics(8, 3, 5, 0.375),
        'scheduler': {'status': 'available', 'states': {'COMPLETED': 2, 'RUNNING': 1, 'PENDING': 2637},
            'matched_declared_slots': 2640}, 'cells': cells,
        'provenance': {'source_manifest_sha256': 'a'*64, 'configuration_sha256': 'b'*64,
            'jobs_sha256': 'c'*64, 'deadline_scope_sha256': 'd'*64,
            'neural_inference_performed': False, 'export_helper': {'path': 'authenticated-reporter.py', 'sha256': 'e'*64},
            'integrity_scope': 'Authenticated frozen saved-report inputs.'}}


class CompactStatusChecks(unittest.TestCase):
    def test_partial_progress_and_pending_are_distinct_from_failures(self):
        source = fixture()
        original = deepcopy(source)
        result = status.compact(source, STUDY)
        self.assertEqual(len(result['cells']), 22)
        self.assertEqual(result['totals']['processed'], 2)
        self.assertEqual(result['totals']['aborted'], 1)
        self.assertEqual(result['totals']['missing'], 2638)
        self.assertEqual(result['cells'][1]['scheduler_states'], {'RUNNING': 1, 'PENDING': 119})
        self.assertEqual(result['cells'][1]['aborted'], 0)
        self.assertEqual(result['totals']['original']['accuracy'], 0.375)
        self.assertEqual(result['totals']['original']['corrections'], 5)
        self.assertEqual(source, original)

    def test_unknown_action_totals_stay_null_with_known_partial_counts(self):
        source = fixture()
        for metrics in (source['original_action_metrics_all_closed_attempts'],
                        source['cells'][0]['original_action_metrics_all_closed_attempts']):
            metrics.update(observed_original_proposals=None, correct_original_proposals=None,
                original_proposal_accuracy_micro=None, reference_corrections=None,
                proposal_count_coverage=0.5, correction_count_coverage=0.5)
        result = status.compact(source, STUDY)
        self.assertIsNone(result['totals']['original']['accuracy'])
        self.assertIsNone(result['cells'][0]['original']['corrections'])
        self.assertEqual(result['totals']['original']['known_partial_observed'], 8)
        self.assertEqual(result['totals']['original']['known_partial_corrections'], 5)

    def test_unavailable_scheduler_never_publishes_partial_cached_state_counts(self):
        source = fixture()
        source['scheduler'].update(status='query_failed', error='accounting unavailable')
        result = status.compact(source, STUDY)
        self.assertFalse(result['scheduler']['available'])
        self.assertIsNone(result['scheduler']['states'])
        self.assertTrue(all(row['scheduler_states'] is None for row in result['cells']))

    def test_wrong_mode_study_pilot_and_missing_integrity_are_rejected(self):
        for field, value in [('mode', 'free_run'), ('study', STUDY+'-other')]:
            source = fixture()
            source[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                status.compact(source, STUDY)
        source = fixture()
        source['cohort']['development_smokes_included'] = True
        with self.assertRaises(ValueError):
            status.compact(source, STUDY)
        source = fixture()
        source['provenance']['source_manifest_sha256'] = 'invalid'
        with self.assertRaises(ValueError):
            status.compact(source, STUDY)

    def test_duplicate_cells_and_inconsistent_or_boolean_counts_are_rejected(self):
        for change in ('duplicate', 'count', 'bool'):
            source = fixture()
            if change == 'duplicate':
                source['cells'][1]['experiment'] = source['cells'][0]['experiment']
            elif change == 'count':
                source['progress']['episode_records'] = 3
            else:
                source['cells'][0]['aborted_episodes'] = True
            with self.subTest(change=change), self.assertRaises(ValueError):
                status.compact(source, STUDY)

    def test_completed_native_episode_does_not_imply_assisted_goal(self):
        source = fixture()
        source['cells'][0]['verified_outcome_successes'] = 0
        source['progress']['verified_outcome_successes'] = 0
        result = status.compact(source, STUDY)
        self.assertEqual(result['totals']['native_completed'], 1)
        self.assertEqual(result['totals']['assisted_goals'], 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
