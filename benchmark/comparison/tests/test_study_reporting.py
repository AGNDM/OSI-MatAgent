"""Statistical and taxonomy contracts; execution belongs in Slurm."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from comparison.study_metrics import (cluster_interval, correctness_auc, ece_interval, holm_adjust,
                                      paired_permutation, selective_curve,
                                      summarize_decisions, validation_thresholds)
from comparison.study_report import (analyze, apply_plan, discover_experiments,
                                     fingerprint, mcqa_rows, render_html, sha256,
                                     sequential_rows, summarize_sequential)
from comparison.taxonomy import classify_mcqa, classify_sequential


def task(identity="one", split="test", tier="T1"):
    return {"task_id": identity, "group_id": "paired-group", "instrument_id": "instrument",
            "split": split, "tier": tier, "difficulty": "L1", "task_type": "predict",
            "options": [{"option_id": "A", "steps": []}, {"option_id": "B", "steps": []}]}


def taxonomy(tasks):
    return {"labels": [classify_mcqa(row, {"fault_family": "motor_failure", "healthy_control": True})
                       for row in tasks], "mcqa_evaluator_metadata": {
                           row["task_id"]: {"label_class": "healthy", "causes": [{"state_id": "power.on"}]}
                           for row in tasks}}


def decision(identity, correct=True, confidence=.8, group="g", **kwargs):
    return {"task_id": identity, "group_id": group, "correct": correct,
            "confidence": confidence, "probabilities": None, "gold_labels": ["A"],
            "brier": None, "semantic_label": "label", "status": "ok", "attempted": True,
            "valid": True, **kwargs}


class NativeEpisodeAccountingTests(unittest.TestCase):
    """Episode contract verified against completed r2 native Ollama shard004.

    The durable CLI's committed_policy_call_wall_time_s is a separate summary;
    native engine episode rows expose policy_latency_s and token totals.
    """
    def rows(self, token_complete=True):
        case = {'task_id': 'case', 'workflow_id': 'workflow', 'instrument_id': 'instrument'}
        record = {'auxiliary_query': False, 'proposal_correct': True, 'corrected': False,
                  'hierarchy_selections': 1, 'node_confidence_rows': 1, 'node_probability_rows': 0}
        episode = {'status': 'ok', 'trajectory_outcome_met': True,
                   'decision_records': [record], 'policy_latency_s': 35.68072825972922,
                   'input_tokens': 90130, 'output_tokens': 657,
                   'token_accounting_complete': token_complete}
        taxonomy = {'labels': [{'track': 'sequential', 'task_id': 'case',
                               'primary_failure_mode': 'interlock', 'condition': 'fault'}]}
        return sequential_rows([case], {('case', 'free_run'): episode}, taxonomy, 'free_run')

    def test_actual_native_episode_field_names_reach_report(self):
        rows = self.rows()
        self.assertEqual(rows[0]['latency_s'], 35.68072825972922)
        self.assertEqual(rows[0]['input_tokens'], 90130)
        self.assertEqual(rows[0]['output_tokens'], 657)
        summary = summarize_sequential(rows, bootstrap=5)
        self.assertEqual(summary['observed_policy_wall_s'], 35.68072825972922)
        self.assertEqual(summary['known_input_tokens'], 90130)
        self.assertEqual(summary['known_output_tokens'], 657)

    def test_partial_token_accounting_remains_unknown(self):
        rows = self.rows(token_complete=False)
        self.assertIsNone(rows[0]['input_tokens'])
        self.assertIsNone(rows[0]['output_tokens'])


class TaxonomyTests(unittest.TestCase):
    def test_healthy_matched_control_retains_target_mechanism(self):
        result = classify_mcqa(task(), {"fault_family": "motor_failure", "healthy_control": True})
        self.assertEqual(result["primary_failure_mode"], "actuator_motor")
        self.assertEqual(result["condition"], "healthy_control")
        self.assertEqual(result["legacy_level"], "L1")

    def test_unknown_metadata_does_not_guess_from_question(self):
        row = task()
        row["question"] = "The motor is broken; a lamp is also broken"
        self.assertEqual(classify_mcqa(row)["primary_failure_mode"], "unknown")

    def test_schema_decision_overrides_generic_precondition_default(self):
        row = task()
        row["decision_kind"] = "argument_validity"
        result = classify_mcqa(row, {"fault_family": "precondition"})
        self.assertEqual(result["primary_failure_mode"], "configuration_argument")

    def test_sequential_healthy_and_failed_pair_share_mechanism(self):
        base = {"task_id": "a", "workflow_id": "hotplate_motor_diagnostic", "checkpoints": [],
                "expansion": {"condition": "healthy", "pair_id": "pair"}}
        healthy = classify_sequential(base)
        base["expansion"]["condition"] = "motor_blocked"
        failed = classify_sequential(base)
        self.assertEqual(healthy["primary_failure_mode"], failed["primary_failure_mode"])
        self.assertEqual(healthy["condition"], "healthy_control")
        self.assertEqual(failed["condition"], "injected_failure")

    def test_uncertainty_groups_entire_sequential_workflow(self):
        row = classify_sequential({"task_id": "x", "workflow_id": "furnace_program",
                                   "expansion": {"condition": "healthy", "pair_id": "pair"}})
        self.assertEqual(row["group_id"], "furnace_program")


class ResearchMetricTests(unittest.TestCase):
    def test_auroc_requires_both_correctness_classes(self):
        self.assertIsNone(correctness_auc([decision("a"), decision("b")]))
        self.assertEqual(correctness_auc([decision("a", True, .8), decision("b", False, .2)]), 1.)
        self.assertEqual(correctness_auc([decision("a", True, .8), decision("b", False, .8)]), .5)

    def test_selective_threshold_keeps_tied_confidence_group(self):
        curve = selective_curve([decision("a", True, .9), decision("b", False, .9)])
        self.assertEqual(len(curve), 1)
        self.assertEqual(curve[0]["coverage"], 1)
        self.assertEqual(curve[0]["accuracy"], .5)

    def test_single_correlated_group_has_no_fabricated_interval(self):
        value = cluster_interval([decision("a", True), decision("b", False)], samples=20)
        self.assertEqual(value["groups"], 1)
        self.assertEqual(value["estimate"], .5)
        self.assertIsNone(value["low"])
        self.assertIsNone(value["high"])

    def test_verbal_confidence_never_creates_multiclass_brier(self):
        result = summarize_decisions([decision("a")], bootstrap=20)
        self.assertIsNone(result["multiclass_brier"])
        self.assertEqual(result["calibration"]["confidence_coverage"], 1)
        self.assertAlmostEqual(result["calibration"]["binary_brier_selected_correctness"], .04)

    def test_ece_cluster_point_matches_reported_calibration(self):
        rows = [decision("a", True, .8, group="g1"), decision("b", False, .2, group="g2")]
        result = summarize_decisions(rows, bootstrap=20)
        self.assertAlmostEqual(result["ece_ci95"]["estimate"], result["calibration"]["ece"])
        self.assertEqual(result["ece_ci95"]["groups"], 2)
        self.assertIsNotNone(result["ece_ci95"]["low"])

    def test_brier_interval_only_uses_available_genuine_vectors(self):
        rows = [decision("a", brier=.1, uniform_brier=.5, group="g1"),
                decision("b", brier=.3, uniform_brier=.5, group="g2"), decision("c", group="g3")]
        result = summarize_decisions(rows, bootstrap=20)
        self.assertEqual(result["multiclass_brier_ci95"]["groups"], 2)
        self.assertAlmostEqual(result["multiclass_brier_ci95"]["estimate"], .2)

    def test_validation_threshold_is_not_fitted_on_heldout(self):
        rows = [decision("v1", True, .8, split="validation"),
                decision("v2", False, .4, split="validation"),
                decision("test", False, .9, split="test")]
        result = validation_thresholds(rows)["0.95"]
        self.assertEqual(result["threshold"], .8)
        self.assertEqual(result["heldout_precision"], 0)

    def test_holm_adjustment_is_monotone_and_bounded(self):
        rows = holm_adjust([{"p_value": .01}, {"p_value": .04}, {"p_value": .03}, {"p_value": None}])
        self.assertEqual([row.get("holm_p_value") for row in rows], [.03, .06, .06, None])

    def test_paired_permutation_preserves_cluster_unit(self):
        left = [decision("a", True, group="g1"), decision("b", True, group="g1"), decision("c", False, group="g2")]
        right = [decision("a", False, group="g1"), decision("b", False, group="g1"), decision("c", False, group="g2")]
        result = paired_permutation(left, right, samples=50)
        self.assertEqual(result["groups"], 2)
        self.assertEqual(result["paired_tasks"], 3)
        self.assertEqual(result["difference_group_macro"], .5)


class StudyAssemblyTests(unittest.TestCase):
    def test_missing_and_invalid_outputs_stay_in_cohort(self):
        tasks = [task("a"), task("b"), task("c")]
        predictions = {"a": {"task_id": "a", "status": "ok", "selected_option_id": "A"},
                       "b": {"task_id": "b", "status": "model_output_invalid"}}
        rows = mcqa_rows(tasks, {row["task_id"]: ["A"] for row in tasks}, predictions, taxonomy(tasks))
        result = summarize_decisions(rows, bootstrap=20)
        self.assertEqual(result["n"], 3)
        self.assertEqual(result["accuracy"], 1 / 3)
        self.assertEqual(result["missing"], 1)
        self.assertEqual(result["model_output_invalid"], 1)

    def test_agent_shortlist_coverage_and_conditional_confidence_are_separate(self):
        tasks = [task()]
        predictions = {"one": {"task_id": "one", "status": "ok", "selected_option_id": "B",
                               "backend_result": {"confidence": .9, "probabilities": None,
                                                  "metadata": {"agentic_trace": {"shortlist_ids": ["B"]}}}}}
        row = mcqa_rows(tasks, {"one": ["A"]}, predictions, taxonomy(tasks))[0]
        self.assertFalse(row["shortlist_covers_gold"])
        self.assertIsNone(row["probabilities"])
        self.assertIsNone(row["brier"])

    def test_expert_identifier_mentions_are_labelled_as_lexical_agreement(self):
        tasks = [task()]
        predictions = {"one": {"task_id": "one", "status": "ok", "selected_option_id": "A",
                       "backend_result": {"causes_cited": [], "metadata": {
                           "evidence_citations_model_generated_unverified": ["power.on is false"]}}}}
        row = mcqa_rows(tasks, {"one": ["A"]}, predictions, taxonomy(tasks))[0]
        self.assertEqual(row["causes_cited"], [])
        self.assertEqual(row["expected_cause_identifier_mentions"], ["power.on"])
        self.assertEqual(row["cause_citation_source"], "explicit_identifier_list")
        result = summarize_decisions([row], bootstrap=20)
        self.assertIsNone(result["cause_citation_exact_match"])
        self.assertEqual(result["expert_expected_identifier_mention_recall"], 1.)

    def test_plan_retains_not_started_experiments(self):
        entries = apply_plan([], {"experiments": [{"name": "pending", "method": "sampling", "seed": 2,
                                                   "tracks": ["mcqa"], "analysis_family": "sampling"}]})
        self.assertEqual(entries[0]["requested_tracks"], {"mcqa"})
        self.assertEqual(entries[0]["mcqa"], {})
        result, rows = analyze(entries, [task()], {"one": ["A"]}, [], taxonomy([task()]),
                               bootstrap=20, permutations=20)
        self.assertEqual(result["experiments"][0]["mcqa"]["all_388"]["missing"], 1)
        self.assertFalse(result["seed_summaries"]["sampling"]["three_seed_sampling_complete"])

    def test_reference_does_not_enter_competitor_permutation_tests(self):
        plan = {"experiments": [{"name": "rules", "method": "choice", "tracks": ["mcqa"],
                                  "reporting_scope": "T1_reference", "reporting_role": "reference"}]}
        entries = apply_plan([], plan)
        result, _ = analyze(entries, [task("a", tier="T1"), task("b", tier="T2")],
                            {"a": ["A"], "b": ["A"]}, [], taxonomy([task("a"), task("b")]),
                            bootstrap=20, permutations=20)
        self.assertEqual(result["experiments"][0]["mcqa"]["all_388"]["n"], 1)
        self.assertEqual(result["paired_comparisons"], {})

    def test_discovery_rejects_release_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle, run = root / "instrument", root / "run"
            bundle.mkdir(); run.mkdir()
            (bundle / "manifest.json").write_text("{}")
            (run / "configuration.json").write_text(json.dumps({
                "resolved_configuration": {"experiments": [{"name": "a"}]},
                "source_sha256": {"instrument/manifest.json": "mismatch"}}))
            experiments, sources, warnings = discover_experiments([root], bundle)
            self.assertEqual(experiments, [])
            self.assertIn("incompatible", warnings[0])

    def test_discovery_merges_disjoint_shards_and_rejects_duplicate_tasks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "instrument"; bundle.mkdir()
            (bundle / "manifest.json").write_text("{}")
            specification = {"name": "method", "seed": 0, "tracks": ["mcqa"]}
            for name, task_id in (("shard-a", "a"), ("shard-b", "b")):
                run = root / name; run.mkdir()
                (run / "configuration.json").write_text(json.dumps({
                    "resolved_configuration": {"experiments": [specification]},
                    "source_sha256": {"instrument/manifest.json": sha256(bundle / "manifest.json")}}))
                data = run / "method/mcqa"; data.mkdir(parents=True)
                (data / "predictions.jsonl").write_text(json.dumps({"task_id": task_id, "status": "ok"}) + "\n")
            experiments, sources, warnings = discover_experiments([root], bundle)
            self.assertEqual(set(experiments[0]["mcqa"]), {"a", "b"})
            (root / "shard-b/method/mcqa/predictions.jsonl").write_text('{"task_id":"a"}\n')
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                discover_experiments([root], bundle)

    def test_discovery_authenticates_feature_artifacts_across_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "instrument"; bundle.mkdir()
            (bundle / "manifest.json").write_text("{}")
            for index in (0, 1):
                run = root / str(index); run.mkdir()
                (run / "configuration.json").write_text(json.dumps({
                    "resolved_configuration": {"experiments": [{"name": "feature"}]},
                    "source_sha256": {"instrument/manifest.json": sha256(bundle / "manifest.json")}}))
                target = run / "feature"; target.mkdir()
                (target / "identity.json").write_text(json.dumps({"artifact_sha256": str(index)}))
            with self.assertRaisesRegex(ValueError, "Incompatible"):
                discover_experiments([root], bundle)


if __name__ == "__main__":
    unittest.main()
