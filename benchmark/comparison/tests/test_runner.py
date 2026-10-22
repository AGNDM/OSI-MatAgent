"""Runner contracts and confidence semantics; execute through Slurm."""
import math
from contextlib import redirect_stderr
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from comparison.aggregate import calibration, mcqa_calibration, sequential_calibration, flatten_report
from comparison.policies import decision_parameters, validated_result
from comparison.run import validate_config, select_mcqa, resolve_configuration, run_sequential, main
from comparison.policies import load_bundle


class ConfigurationTests(unittest.TestCase):
    def config(self, **experiment):
        return {"schema_version": 1, "experiments": [{"name": "one", "backend": {"kind": "hf"}, **experiment}]}

    def test_minimal_config(self):
        self.assertEqual(validate_config(self.config())["schema_version"], 1)

    def test_distinct_safe_names_required(self):
        config = self.config()
        config["experiments"] *= 2
        with self.assertRaises(ValueError):
            validate_config(config)
        with self.assertRaises(ValueError):
            validate_config(self.config(name="../outside"))

    def test_expert_architecture_not_silently_direct(self):
        with self.assertRaises(ValueError):
            validate_config(self.config(architecture="expert_decision"))

    def test_small_sampling_not_misrepresented_as_confidence(self):
        with self.assertRaises(ValueError):
            validate_config(self.config(method="sampling", samples=2))

    def test_invalid_temperature(self):
        for value in (False, 0, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_config(self.config(temperature=value))

    def test_heldout_cohort_preserves_all_test_types(self):
        tasks = [{"split": split} for split in ("train", "test", "test_fault", "test_template", "test_instrument", "validation")]
        self.assertEqual(len(select_mcqa(tasks, "heldout")), 4)

    def test_decision_parameters_are_explicit(self):
        self.assertEqual(decision_parameters({"method": "native", "seed": 7})["seed"], 7)

    def test_environment_resolution_does_not_require_unselected_models(self):
        config = self.config()
        config["experiments"][0]["backend"]["model_path"] = "${COMPARISON_TEST_MODEL}/weights"
        config["experiments"].append({"name": "other", "backend": {"model_path": "${MISSING_UNUSED_COMPARISON_MODEL}"}})
        with patch.dict(os.environ, {"COMPARISON_TEST_MODEL": "/tmp/local"}):
            resolved = resolve_configuration(config, "/tmp/configs/test.json", {"one"})
        self.assertEqual(resolved["experiments"][0]["backend"]["model_path"], "/tmp/local/weights")
        self.assertEqual(resolved["experiments"][1]["backend"]["model_path"], "${MISSING_UNUSED_COMPARISON_MODEL}")

    def test_unresolved_selected_model_path_rejected(self):
        config = self.config()
        config["experiments"][0]["backend"]["model_path"] = "${MISSING_SELECTED_COMPARISON_MODEL}"
        with self.assertRaises(ValueError):
            resolve_configuration(config, "/tmp/config.json", {"one"})

    @unittest.skipUnless(os.environ.get('SLURMD_NODENAME') and os.environ.get('SLURM_JOB_ID'),
                         'Runner checks must run inside a real Slurm allocation')
    def test_output_cannot_be_nested_in_comparison_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / 'comparison'
            package.mkdir()
            config = root / 'config.json'
            config.write_text(json.dumps(self.config()))
            output = package / 'experiments' / 'new-run'
            error = io.StringIO()
            with patch('comparison.run.PACKAGE', package), \
                 patch('comparison.run.load_bundle') as load, redirect_stderr(error):
                status = main(['--config', str(config), '--output', str(output), '--preflight'])
            self.assertEqual(status, 2)
            self.assertIn('comparison source', error.getvalue())
            self.assertFalse(output.exists())
            load.assert_not_called()


class BackendResultTests(unittest.TestCase):
    def test_choice_without_confidence_is_valid(self):
        row = {"status": "ok", "selected_option_id": "A", "probabilities": None, "confidence": None}
        self.assertIs(validated_result(row, ["A", "B"]), row)

    def test_foreign_label_is_invalid(self):
        with self.assertRaises(ValueError):
            validated_result({"status": "ok", "selected_option_id": "Z"}, ["A", "B"])

    def test_probability_vector_must_cover_whole_menu(self):
        for probabilities in ({"A": 1.0}, {"A": .8, "B": .8}, {"A": float("nan"), "B": 1.0}):
            with self.subTest(probabilities=probabilities), self.assertRaises(ValueError):
                validated_result({"status": "ok", "selected_option_id": "A", "probabilities": probabilities}, ["A", "B"])

    def test_backend_error_is_not_a_choice(self):
        with self.assertRaises(RuntimeError):
            validated_result({"status": "error", "error": "unsupported native menu"}, ["A"])


class CalibrationTests(unittest.TestCase):
    def test_known_calibration_and_brier(self):
        rows = [{"confidence": .8, "correct": True, "probabilities": {"A": .8, "B": .2}, "gold_labels": ["A"]},
                {"confidence": .8, "correct": False, "probabilities": {"A": .8, "B": .2}, "gold_labels": ["B"]}]
        report = calibration(rows, denominator=4)
        self.assertAlmostEqual(report["ece"], .3)
        self.assertAlmostEqual(report["multiclass_brier"], .68)
        self.assertEqual(report["confidence_coverage"], .5)

    def test_confidence_unavailable_remains_null(self):
        report = calibration([{"correct": True, "confidence": None}])
        self.assertIsNone(report["ece"])
        self.assertIsNone(report["multiclass_brier"])
        self.assertEqual(report["confidence_coverage"], 0)

    def test_verbalized_scalar_does_not_create_multiclass_brier(self):
        report = calibration([{"correct": True, "confidence": .9}])
        self.assertAlmostEqual(report["ece"], .1)
        self.assertIsNone(report["multiclass_brier"])

    def test_multiple_gold_does_not_invent_target_distribution(self):
        report = calibration([{"correct": True, "confidence": .9,
                               "probabilities": {"A": .9, "B": .1}, "gold_labels": ["A", "B"]}])
        self.assertIsNone(report["multiclass_brier"])

    def test_confidence_one_is_in_final_bin(self):
        self.assertEqual(calibration([{"correct": True, "confidence": 1.0}])["ece"], 0)

    def test_missing_predictions_stay_in_calibration_denominator(self):
        tasks = [{"task_id": "one", "options": [{"option_id": "A"}, {"option_id": "B"}]},
                 {"task_id": "two", "options": [{"option_id": "A"}, {"option_id": "B"}]}]
        answers = {name: {"correct_option_ids": ["A"]} for name in ("one", "two")}
        result = mcqa_calibration(tasks, answers, [{"task_id": "one", "status": "ok", "selected_option_id": "A", "confidence": .9}])
        self.assertEqual(result["confidence_coverage"], .5)

    def test_hierarchy_probabilities_not_presented_as_action_calibration(self):
        report = sequential_calibration([{"mode": "teacher_forced", "status": "ok", "decision_records": [
            {"proposal_correct": False, "usage": {"selection_events": [{"label_probabilities": {"A": .9, "B": .1}, "confidence": .9}]}}]}])
        self.assertIsNone(report["teacher_forced"]["action_ece"])
        self.assertEqual(report["teacher_forced"]["selections_with_probability_vector"], 1)


class NativeErrorClassificationTests(unittest.TestCase):
    """Exercise actual simulator corrections, with no model or network calls."""
    @classmethod
    def setUpClass(cls):
        cls.bundle = Path(os.environ.get("COMPARISON_TEST_BUNDLE", Path(__file__).resolve().parents[2] / "instrument")).resolve()
        if not cls.bundle.is_dir():
            raise unittest.SkipTest("Set COMPARISON_TEST_BUNDLE to the frozen instrument release")
        cls.helpers = load_bundle(cls.bundle)

    def fake_backend(self, kind):
        class Backend:
            identity = {"kind": "test-double", "diagnostic": True}
            def decide(self, *args, **kwargs):
                return {"status": "error", "selected_option_id": None, "probabilities": None,
                        "confidence": None, "error": "malformed reply" if kind == "model_output" else "unsupported context",
                        "input_tokens": 25, "output_tokens": 3,
                        "metadata": {"error_kind": kind, "raw_completion": "not a label"}}
        return Backend()

    def test_malformed_completion_is_wrong_then_corrected_and_continues(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            summary = run_sequential(self.bundle, self.helpers, self.fake_backend("model_output"),
                {"name": "invalid", "backend": {}, "method": "choice"}, {"mode": "teacher_forced"}, output, 1)
            episodes = self.helpers["common"].load_jsonl(output / "episodes.jsonl")
        episode = episodes[0]
        self.assertEqual(episode["status"], "ok")
        self.assertTrue(episode["trajectory_outcome_met"])
        self.assertGreater(episode["interventions"], 1)
        self.assertEqual(episode["control_decision_accuracy"], 0)
        self.assertEqual(episode["decision_records"][0]["proposed_action"], {"kind": "invalid_model_reply"})
        self.assertFalse(episode["decision_records"][0]["proposed_action_is_offered"])
        self.assertTrue(episode["decision_records"][0]["corrected"])
        self.assertIsNotNone(episode["decision_records"][0]["executed_menu_path"])
        self.assertEqual(episode["decision_records"][0]["usage"]["input_tokens"], 25)
        self.assertEqual(episode["decision_records"][0]["usage"]["output_tokens"], 3)
        self.assertTrue(episode["decision_records"][0]["usage"]["input_tokens_known"])
        self.assertEqual(summary["full_dataset_report"]["by_mode"]["teacher_forced"]["missing"], 119)

    def test_unsupported_backend_remains_unscored_infrastructure(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = run_sequential(self.bundle, self.helpers, self.fake_backend("infrastructure"),
                {"name": "unsupported", "backend": {}, "method": "choice"}, {"mode": "teacher_forced"}, Path(directory) / "run", 1)
        mode = summary["by_mode"]["teacher_forced"]
        self.assertEqual(mode["completed"], 0)
        self.assertEqual(mode["infrastructure_failures"], 1)
        self.assertEqual(mode["interventions"], 0)


if __name__ == "__main__":
    unittest.main()
