"""Focused final-audit regressions; saved fixtures only, no scheduler/models."""
from copy import deepcopy
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURMD_NODENAME'):
    raise RuntimeError('A genuine Slurm allocation is required')

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'workflow'))
import final_audit as audit

audit.STUDY = Path(os.environ.get('EVALUATION_STUDY', '/tmp/evaluation-study-fixture')).resolve()


class MetricReconciliation(unittest.TestCase):
    def fixture(self):
        return {'totals': {'original': {'observed': 100, 'correct': 42, 'accuracy': .42}, 'reference_corrections': 58},
            'cells': [{'experiment': 'example', 'scores': {'overall': {'accuracy': .42, 'denominator': 100},
                'by_instrument': {'furnace': {'accuracy': .5, 'denominator': 20}}},
                'checkpoint': {'model_path': '/released/checkpoint', 'sha256': 'abc'}}]}

    def test_identical_nested_metrics_pass(self):
        expected = self.fixture()
        self.assertEqual(audit.differences(deepcopy(expected), expected), [])

    def test_wrong_compact_overall_original_accuracy_rejected(self):
        expected = self.fixture()
        actual = deepcopy(expected)
        actual['totals']['original']['accuracy'] = .43
        self.assertTrue(audit.differences(actual, expected))

    def test_changed_nested_group_score_rejected(self):
        expected = self.fixture()
        actual = deepcopy(expected)
        actual['cells'][0]['scores']['by_instrument']['furnace']['accuracy'] = .55
        self.assertTrue(audit.differences(actual, expected))

    def test_changed_checkpoint_identity_rejected(self):
        expected = self.fixture()
        actual = deepcopy(expected)
        actual['cells'][0]['checkpoint']['sha256'] = 'different'
        self.assertTrue(audit.differences(actual, expected))

    def test_missing_or_additional_score_is_not_silently_ignored(self):
        expected = self.fixture()
        for mutation in ('missing', 'extra'):
            with self.subTest(mutation=mutation):
                actual = deepcopy(expected)
                if mutation == 'missing':
                    del actual['totals']['original']['correct']
                else:
                    actual['totals']['invented_score'] = 1
                self.assertTrue(audit.differences(actual, expected))


class SourceBindings(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='tf-final-source-fixture-', dir='/tmp')
        self.study = Path(self.temporary.name)
        self.study_patch = patch.object(audit, 'STUDY', self.study)
        self.study_patch.start()
        self.path = self.study / 'runs/example/sequential/episodes.jsonl'
        self.path.parent.mkdir(parents=True)
        self.path.write_bytes(b'{"correct":1}\n')

    def tearDown(self):
        self.study_patch.stop()
        self.temporary.cleanup()

    def test_exact_current_source_binding_passes(self):
        source = audit.file_proof(self.path)
        self.assertEqual(audit.verify_source_bindings([source], {str(self.path): source}), [])

    def test_same_count_same_length_stale_source_fails(self):
        old = audit.file_proof(self.path)
        self.path.write_bytes(b'{"correct":0}\n')
        current = audit.file_proof(self.path)
        self.assertEqual(old['bytes'], current['bytes'])
        self.assertTrue(audit.verify_source_bindings([old], {str(self.path): current}))

    def test_missing_source_cannot_pass_on_count_alone(self):
        expected = audit.file_proof(self.path)
        other = self.path.with_name('different.jsonl')
        other.write_bytes(self.path.read_bytes())
        self.assertTrue(audit.verify_source_bindings([audit.file_proof(other)], {str(self.path): expected}))

    def test_source_outside_production_runs_is_rejected(self):
        other = self.study / 'outside.jsonl'
        other.write_bytes(self.path.read_bytes())
        source = audit.file_proof(other)
        self.assertTrue(audit.verify_source_bindings([source], {str(other): source}))


class PerCaseOutcomes(unittest.TestCase):
    def test_exact_completed_default_and_aborted_null_semantics(self):
        for episode, expected in (({'status': 'ok', 'trajectory_outcome_met': True}, True),
                ({'trajectory_outcome_met': False}, False),
                ({'status': 'infrastructure_error', 'trajectory_outcome_met': True}, None)):
            with self.subTest(episode=episode):
                self.assertIs(audit.native_case_outcome(episode), expected)
                self.assertEqual(audit.case_outcome_errors({'native_outcome_met': expected}, expected), [])

    def test_swapped_case_flags_fail_despite_equal_aggregate_goal_count(self):
        episodes = [{'status': 'ok', 'trajectory_outcome_met': value} for value in (True, False)]
        expected = [audit.native_case_outcome(row) for row in episodes]
        swapped = [{'native_outcome_met': value} for value in reversed(expected)]
        self.assertEqual(sum(expected), sum(row['native_outcome_met'] for row in swapped))
        self.assertTrue(all(audit.case_outcome_errors(row, target) for row, target in zip(swapped, expected)))

    def test_missing_abort_null_and_boolean_count_substitution_fail(self):
        self.assertTrue(audit.case_outcome_errors({}, None))
        self.assertTrue(audit.case_outcome_errors({'native_outcome_met': 1}, True))


class FinalFailureEvidence(unittest.TestCase):
    def fixture(self):
        hashes = {'source_manifest_sha256': 'a' * 64, 'configuration_sha256': 'b' * 64,
            'jobs_sha256': 'c' * 64, 'deadline_scope_sha256': 'd' * 64}
        document = {'study': str(audit.STUDY), 'mode': 'teacher_forced', 'seed': 0,
            **{key: value for key, value in hashes.items() if key != 'deadline_scope_sha256'},
            'read_errors': [], 'trace_read_error_count': 0, 'cells': []}
        cells = {}
        for cell_index in range(22):
            name = f'cell-{cell_index:02d}'
            source_cases = []
            failures = []
            for shard in range(120):
                index = cell_index * 120 + shard
                task_id = f'public-task-{shard:03d}'
                path = str(audit.STUDY / f'runs/{name}/shard-{shard:03d}/episodes.jsonl')
                episode = {'path': path, 'sha256': f'{index:064x}', 'bytes': 100}
                source_cases.append({'index': index, 'task_id': task_id, 'episode': episode,
                    'record_status': 'infrastructure_error' if shard < 2 else 'ok'})
                if shard < 2:
                    category = 'expert_schema_error' if shard == 0 else 'terminal_deadline'
                    events = [{'categories': ['expert_schema_error'], 'request_sent': True,
                        'generated_response': {'raw_response_characters': 40, 'raw_response_sha256': 'e' * 64}}] if shard == 0 else []
                    failures.append({'task_id': task_id, 'job_index': index, 'episode_path': path,
                        'episode_sha256': episode['sha256'], 'trace_read_errors': [],
                        'observed_categories': [category], 'graph_error_events': events})
            cell = {'closed': 120, 'native_completed': 118, 'assisted_goals': 118, 'aborted': 2,
                'explicit_initialization_failures': 0, 'source_cases': source_cases}
            cells[name] = cell
            document['cells'].append({'experiment': name, 'closed': 120, 'native_completed': 118,
                'assisted_goals': 118, 'aborted': 2, 'missing': 0, 'unusable': 0, 'in_progress_unclosed': 0,
                'failures': failures, 'abort_categories': {'expert_schema_error': 1, 'terminal_deadline': 1}})
        return document, cells, hashes

    def test_full_22_by_120_model_and_budget_failures_are_valid_closed_results(self):
        document, cells, hashes = self.fixture()
        self.assertEqual(audit.failure_evidence_errors(document, cells, hashes), [])

    def test_actual_technical_and_unknown_categories_block_completion(self):
        for category in ('context_overflow', 'cuda_out_of_memory', 'cuda_runtime_error',
                'ipc_or_worker_failure', 'missing_file_or_artifact_identity', 'library_or_api_compatibility', 'unknown'):
            with self.subTest(category=category):
                document, cells, hashes = self.fixture()
                row = document['cells'][0]
                row['failures'][0]['observed_categories'] = [category]
                row['abort_categories'] = {category: 1, 'terminal_deadline': 1}
                self.assertTrue(audit.failure_evidence_errors(document, cells, hashes))

    def test_budget_label_cannot_mask_an_unmapped_actual_error_event(self):
        document, cells, hashes = self.fixture()
        failure = document['cells'][0]['failures'][1]
        failure['graph_error_events'] = [{'kind': 'operation_error', 'categories': [],
            'error': 'RuntimeError: unknown worker fault'}]
        self.assertEqual(failure['observed_categories'], ['terminal_deadline'])
        self.assertTrue(audit.failure_evidence_errors(document, cells, hashes))

    def test_initialization_failure_blocks_even_when_other_counts_match(self):
        document, cells, hashes = self.fixture()
        cells['cell-00']['explicit_initialization_failures'] = 1
        self.assertTrue(audit.failure_evidence_errors(document, cells, hashes))

    def test_same_count_stale_aborted_episode_identity_rejected(self):
        document, cells, hashes = self.fixture()
        document['cells'][0]['failures'][0]['episode_sha256'] = 'f' * 64
        self.assertTrue(audit.failure_evidence_errors(document, cells, hashes))

    def test_totals_alone_cannot_hide_incomplete_cells(self):
        document, cells, hashes = self.fixture()
        document['cells'][0]['closed'] = 119
        document['cells'][1]['closed'] = 121
        self.assertEqual(sum(row['closed'] for row in document['cells']), 2640)
        self.assertTrue(audit.failure_evidence_errors(document, cells, hashes))

    def test_missing_aborted_case_is_rejected(self):
        document, cells, hashes = self.fixture()
        document['cells'][0]['failures'].pop()
        self.assertTrue(audit.failure_evidence_errors(document, cells, hashes))

    def test_model_error_label_requires_real_generated_response_proof(self):
        document, cells, hashes = self.fixture()
        document['cells'][0]['failures'][0]['graph_error_events'][0]['request_sent'] = False
        self.assertTrue(audit.failure_evidence_errors(document, cells, hashes))

    def test_wrong_declared_seed_or_source_hash_is_rejected(self):
        for field, value in (('seed', 1), ('configuration_sha256', 'f' * 64)):
            with self.subTest(field=field):
                document, cells, hashes = self.fixture()
                document[field] = value
                self.assertTrue(audit.failure_evidence_errors(document, cells, hashes))


class FrozenSavedScoreReconciliation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory = (audit.STUDY / 'snapshot/instrument/eval' if os.environ.get('EVALUATION_STUDY')
                     else Path(__file__).resolve().parents[2] / 'instrument/eval')
        with patch.object(sys, 'path', [str(directory), *sys.path]):
            cls.scorer = audit.load_module('final_audit_saved_score_fixture', directory / 'sequential.py')

    def scored_fixture(self):
        tasks = [{'task_id': f'task-{index}', 'workflow_id': f'workflow-{index}',
            'instrument_id': instrument, 'condition': 'nominal'}
            for index, instrument in enumerate(('furnace', 'oven'))]
        episodes = []
        for index, grades in enumerate(((True, False, True), (True, True, False))):
            decisions = [{'decision_index': position, 'proposal_correct': correct,
                'corrected': not correct, 'auxiliary_query': position == 3,
                'checkpoint_id': 'checkpoint-a' if position < 3 else None,
                'stage': 'prepare' if position < 3 else None,
                'proposed_action': {'command': f'offered-{position}'}, 'response': {}}
                for position, correct in enumerate(grades, 1)]
            episodes.append({'task_id': f'task-{index}', 'mode': 'teacher_forced', 'status': 'ok',
                'trajectory_outcome_met': True, 'menu_membership_verified': True,
                'decision_records': decisions})
        with patch.object(self.scorer, '_runtime', side_effect=AssertionError('Native replay is prohibited')):
            return self.scorer.score_episodes(tasks, episodes, modes=('teacher_forced',))['by_mode']['teacher_forced']

    def test_frozen_scorer_aggregates_saved_controls_queries_and_checkpoint_grades(self):
        score = self.scored_fixture()
        self.assertEqual(score['decisions_observed'], 6)
        self.assertEqual(score['control_decisions_observed'], 4)
        self.assertEqual(score['query_decisions_observed'], 2)
        self.assertEqual(score['observed_checkpoint_accuracy_episode_macro'], .5)
        self.assertEqual(audit.differences(deepcopy(score), score), [])

    def test_changed_computed_native_checkpoint_score_rejected(self):
        score = self.scored_fixture()
        changed = deepcopy(score)
        changed['observed_checkpoint_accuracy_episode_macro'] = 1.0
        self.assertTrue(audit.differences(changed, score, 'native_score'))

    def test_changed_computed_native_group_score_rejected(self):
        score = self.scored_fixture()
        changed = deepcopy(score)
        changed['by_instrument_id']['furnace']['observed_control_accuracy_micro'] = 1.0
        self.assertTrue(audit.differences(changed, score, 'native_score'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
