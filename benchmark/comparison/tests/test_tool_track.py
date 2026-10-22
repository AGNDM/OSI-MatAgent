"""Conditional labels and missing coverage must stay distinct from static keys."""
import os
from pathlib import Path
import unittest

from comparison.policies import load_bundle
from comparison.tool_track import LocalInformationPolicy, conditional_score


class ConditionalToolScoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = Path(os.environ.get("COMPARISON_TEST_BUNDLE", Path(__file__).resolve().parents[2] / "instrument")).resolve()
        if not cls.bundle.is_dir():
            raise unittest.SkipTest("Set COMPARISON_TEST_BUNDLE to the frozen release")
        cls.helpers = load_bundle(cls.bundle)

    def tasks(self):
        return [{"task_id": name, "split": split, "tier": "T1", "question": "Use the public evidence.",
                 "options": [{"option_id": "A", "text": "one"}, {"option_id": "B", "text": "two"}]}
                for name, split in (("a", "train"), ("b", "test"), ("c", "test"), ("d", "validation"))]

    def test_conditional_gold_scores_and_full_source_missing_denominator(self):
        tasks = self.tasks()
        checkpoints = []
        for task, gold, changed in ((tasks[0], "B", True), (tasks[1], "A", False)):
            checkpoints.append({"derived_task": task,
                "conditional_oracle": {"task_id": task["task_id"], "correct_option_ids": [gold],
                                       "conditional_evaluation": {"label_changed": changed}},
                "prediction": {"task_id": task["task_id"], "status": "ok", "selected_option_id": "B",
                               "confidence": .8, "probabilities": {"A": .2, "B": .8}}})
        report = conditional_score(self.bundle, tasks, checkpoints)
        self.assertEqual(report["conditional_cohort_score"]["accuracy"], .5)
        self.assertEqual(report["full_source_accuracy"], .25)
        self.assertEqual(report["conditional_label_coverage"], .5)
        self.assertEqual(report["full_source_prediction_coverage"], .5)
        self.assertEqual(report["full_source_heldout_prediction_coverage"], .5)
        self.assertEqual(report["missing_task_ids"], ["c", "d"])
        self.assertEqual(report["labels_changed_after_information"], 1)
        self.assertFalse(report["static_answer_key_used_for_grading"])
        self.assertFalse(report["complete_source_cohort"])

    def test_unobserved_conditional_keys_are_not_invented(self):
        report = conditional_score(self.bundle, self.tasks(), [])
        self.assertEqual(report["full_source_accuracy"], 0)
        self.assertEqual(report["conditional_label_rows"], 0)
        self.assertIsNone(report["conditional_cohort_score"])
        self.assertIsNone(report["conditional_calibration"])

    def test_information_policy_receives_only_public_projection(self):
        self.helpers["sequential"]._runtime()
        class Backend:
            identity = {"kind": "test-double"}
            def _generate(self, messages, **parameters):
                self.messages = messages
                return '{"kind":"decide"}', 30, 5
        backend = Backend()
        policy = LocalInformationPolicy(backend, None, "logprobs")
        task = {**self.tasks()[0], "correct_option_ids": ["B"], "private_fixture": "do-not-leak", "tier": "T3"}
        action = policy.next_information_action(task, [])
        self.assertEqual(action, {"kind": "decide"})
        self.assertNotIn("correct_option_ids", backend.messages[1]["content"])
        self.assertNotIn("do-not-leak", backend.messages[1]["content"])
        self.assertNotIn('"tier"', backend.messages[1]["content"])
        self.assertEqual(policy.last_information_usage["input_tokens"], 30)
        self.assertEqual(policy.information_generations[0]["raw_completion"], '{"kind":"decide"}')


if __name__ == "__main__":
    unittest.main()
