"""Scoring-contract tests. Execute these only within a Slurm allocation."""
from __future__ import annotations

import copy
import gzip
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

BUNDLE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUNDLE_ROOT))

from eval.common import load_dataset, load_jsonl
from eval.mcqa import CATEGORY_NAMES, score_predictions


def fixture():
    tasks = []
    answers = {}
    for task_id, split, difficulty, tier, count, correct in (
        ("train-1", "train", "L1", "T1", 2, ["A"]),
        ("test-1", "test", "L1", "T1", 2, ["B"]),
        ("template-1", "test_template", "L4", "T3", 4, ["D"]),
        ("fault-1", "test_fault", "L3", "T2", 3, ["A"]),
    ):
        tasks.append({
            "task_id": task_id,
            "options": [{"option_id": letter, "text": letter} for letter in "ABCD"[:count]],
            "split": split,
            "difficulty": difficulty,
            "tier": tier,
            "instrument_id": "instrument-a" if count < 4 else "instrument-b",
            "domain": "lab",
            "task_type": "predict" if count < 4 else "recover",
        })
        answers[task_id] = {"task_id": task_id, "correct_option_ids": correct}
    return tasks, answers


def write_jsonl(path, rows):
    opener = gzip.open if Path(path).suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.tasks, self.answers = fixture()

    def test_missing_and_invalid_keep_full_denominator(self):
        report = score_predictions(self.tasks, self.answers, [
            {"task_id": "test-1", "selected_option_id": "B"},
            {"task_id": "template-1", "selected_option_id": "Z"},
        ])
        self.assertEqual((report["n"], report["correct"], report["missing"], report["invalid"]), (3, 1, 1, 1))
        self.assertEqual(report["accuracy"], 1 / 3)
        self.assertEqual(report["coverage"], 1 / 3)
        self.assertEqual(report["submitted"], 2)
        self.assertEqual(report["by_option_count"]["4"]["accuracy"], 0)
        self.assertEqual(report["by_category"][CATEGORY_NAMES["L3"]]["missing"], 1)

    def test_known_outside_cohort_is_ignored(self):
        report = score_predictions(self.tasks, self.answers, [
            {"task_id": "test-1", "selected_option_id": "B"},
            {"task_id": "train-1", "selected_option_id": "A"},
        ], split="test")
        self.assertEqual(report["n"], 1)
        self.assertEqual(report["accuracy"], 1)
        self.assertEqual(report["ignored_out_of_cohort"], 1)
        self.assertEqual(report["dataset_size"], 4)

    def test_unknown_and_duplicate_ids_are_rejected_even_outside_cohort(self):
        for predictions, message in (
            ([{"task_id": "typo", "selected_option_id": "A"}], "unknown prediction"),
            ([{"task_id": "train-1"}, {"task_id": "train-1"}], "duplicate prediction"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                score_predictions(self.tasks, self.answers, predictions)

    def test_non_string_ids_are_not_coerced_into_choices(self):
        for choice in (None, True, 1, ["B"], {"option_id": "B"}):
            with self.subTest(choice=choice):
                report = score_predictions(self.tasks, self.answers, [
                    {"task_id": "test-1", "selected_option_id": choice}
                ], "test")
                self.assertEqual(report["invalid"], 1)
                self.assertEqual(report["correct"], 0)

    def test_status_error_is_wrong_even_with_correct_choice(self):
        report = score_predictions(self.tasks, self.answers, [
            {"task_id": "test-1", "selected_option_id": "B", "status": "error"}
        ], "test")
        self.assertEqual(report["invalid"], 1)
        self.assertEqual(report["correct"], 0)

    def test_missing_choice_is_invalid_submission(self):
        report = score_predictions(self.tasks, self.answers, [{"task_id": "test-1"}], "test")
        self.assertEqual(report["invalid"], 1)
        self.assertEqual(report["missing"], 0)

    def test_probabilities_do_not_choose_answer_or_gate_validity(self):
        for probabilities in (None, {"A": 1, "B": 0}, "not a probability vector"):
            with self.subTest(probabilities=probabilities):
                report = score_predictions(self.tasks, self.answers, [{
                    "task_id": "test-1", "selected_option_id": "B", "probabilities": probabilities,
                }], "test")
                self.assertEqual(report["correct"], 1)
                self.assertEqual(report["valid"], 1)
        report = score_predictions(self.tasks, self.answers, [{
            "task_id": "test-1", "selected_option_id": "A", "probabilities": {"A": 0, "B": 1},
        }], "test")
        self.assertEqual(report["correct"], 0)

    def test_random_baseline_and_any_valid_answer(self):
        self.answers["test-1"]["correct_option_ids"] = ["A", "B"]
        report = score_predictions(self.tasks, self.answers, [
            {"task_id": "test-1", "selected_option_id": "A"}
        ])
        self.assertEqual(report["correct"], 1)
        self.assertAlmostEqual(report["uniform_random_accuracy"], (1 + 1 / 4 + 1 / 3) / 3)

    def test_tier_macro_uses_equal_tier_weight(self):
        # Two T1 tasks, one T2 task and one T3 task; only T3 is correct.
        report = score_predictions(self.tasks, self.answers, [
            {"task_id": "template-1", "selected_option_id": "D"}
        ], "all")
        self.assertEqual(report["accuracy"], 1 / 4)
        self.assertEqual(report["tier_macro_accuracy"], 1 / 3)

    def test_all_heldout_subsets_are_included(self):
        extra = copy.deepcopy(self.tasks[1])
        extra["task_id"] = "instrument-1"
        extra["split"] = "test_instrument"
        self.tasks.append(extra)
        self.answers["instrument-1"] = {"correct_option_ids": ["B"]}
        report = score_predictions(self.tasks, self.answers, [])
        self.assertEqual(report["n"], 4)
        self.assertEqual(set(report["by_split"]), {"test", "test_template", "test_fault", "test_instrument"})

    def test_empty_or_unknown_cohorts_do_not_produce_a_score(self):
        for split in ("typo", "validation"):
            with self.subTest(split=split), self.assertRaises(ValueError):
                score_predictions(self.tasks, self.answers, [], split)

    def test_answer_coverage_and_correct_id_validation(self):
        for mutate in (
            lambda answers: answers.pop("test-1"),
            lambda answers: answers.update({"unknown": {"correct_option_ids": ["A"]}}),
            lambda answers: answers["test-1"].update(correct_option_ids=[]),
            lambda answers: answers["test-1"].update(correct_option_ids=["Z"]),
            lambda answers: answers["test-1"].update(correct_option_ids=["A", "A"]),
            lambda answers: answers["test-1"].update(correct_option_ids=[True]),
        ):
            answers = copy.deepcopy(self.answers)
            mutate(answers)
            with self.assertRaises(ValueError):
                score_predictions(self.tasks, answers, [])

    def test_duplicate_and_invalid_question_options_are_rejected(self):
        for options in ([{"option_id": "A"}, {"option_id": "A"}], [{"option_id": 1}, {"option_id": "B"}]):
            tasks = copy.deepcopy(self.tasks)
            tasks[0]["options"] = options
            with self.assertRaises(ValueError):
                score_predictions(tasks, self.answers, [])
        self.tasks.append(copy.deepcopy(self.tasks[0]))
        with self.assertRaisesRegex(ValueError, "duplicate question"):
            score_predictions(self.tasks, self.answers, [])


class FileAndCLITests(unittest.TestCase):
    def test_plain_gzip_and_minimal_answers(self):
        tasks, answers = fixture()
        with tempfile.TemporaryDirectory() as directory:
            questions_path = Path(directory) / "questions.jsonl.gz"
            answers_path = Path(directory) / "answers.jsonl"
            write_jsonl(questions_path, tasks)
            write_jsonl(answers_path, answers.values())
            actual_tasks, actual_answers = load_dataset(questions_path, answers_path)
            self.assertEqual(actual_tasks, tasks)
            self.assertEqual(actual_answers, answers)

    def test_ambiguous_or_malformed_json_is_rejected_with_location(self):
        invalid_lines = (
            '{"task_id": "a", "task_id": "b"}',
            '{"nested": {"choice": "A", "choice": "B"}}',
            '{"value": NaN}',
            '{"value": Infinity}',
            '{"value": -Infinity}',
            '{"value": 1e400}',
            '{"task_id":',
            '["not", "an", "object"]',
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            for line in invalid_lines:
                path.write_text(line + "\n", encoding="utf-8")
                with self.subTest(line=line), self.assertRaisesRegex(ValueError, "predictions.jsonl:1:"):
                    load_jsonl(path)

    def test_duplicate_answer_records_are_rejected(self):
        tasks, answers = fixture()
        with tempfile.TemporaryDirectory() as directory:
            questions_path = Path(directory) / "questions.jsonl"
            answers_path = Path(directory) / "answers.jsonl"
            write_jsonl(questions_path, tasks)
            write_jsonl(answers_path, [*answers.values(), answers["test-1"]])
            with self.assertRaisesRegex(ValueError, "duplicate answer"):
                load_dataset(questions_path, answers_path)

    def test_cli_defaults_are_bundle_relative_and_work_in_a_clean_copy(self):
        tasks, answers = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "standalone"
            (bundle / "eval").mkdir(parents=True)
            (bundle / "mcqa_388").mkdir()
            elsewhere = root / "unrelated-cwd"
            elsewhere.mkdir()
            for name in ("common.py", "mcqa.py"):
                shutil.copy2(BUNDLE_ROOT / "eval" / name, bundle / "eval" / name)
            write_jsonl(bundle / "mcqa_388/questions.jsonl.gz", tasks)
            write_jsonl(bundle / "mcqa_388/answers.jsonl", answers.values())
            write_jsonl(elsewhere / "predictions.jsonl", [{"task_id": "test-1", "selected_option_id": "B"}])
            completed = subprocess.run(
                [sys.executable, str(bundle / "eval/mcqa.py"), "--predictions", "predictions.jsonl", "--output", "output/nested/scores.json"],
                cwd=elsewhere, text=True, capture_output=True, check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads(completed.stdout)
            self.assertEqual(report["n"], 3)
            self.assertEqual(report["accuracy"], 1 / 3)
            self.assertEqual(json.loads((elsewhere / "output/nested/scores.json").read_text()), report)

    def test_cli_rejects_overwriting_any_input(self):
        tasks, answers = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            questions_path = root / "questions.jsonl"
            answers_path = root / "answers.jsonl"
            predictions_path = root / "predictions.jsonl"
            write_jsonl(questions_path, tasks)
            write_jsonl(answers_path, answers.values())
            write_jsonl(predictions_path, [{"task_id": "test-1", "selected_option_id": "B"}])
            for output in (questions_path, answers_path, predictions_path):
                original = output.read_bytes()
                with self.subTest(output=output.name):
                    completed = subprocess.run([
                        sys.executable, str(BUNDLE_ROOT / "eval/mcqa.py"),
                        "--predictions", str(predictions_path),
                        "--questions", str(questions_path),
                        "--answers", str(answers_path),
                        "--output", str(output.parent / "." / output.name),
                    ], text=True, capture_output=True, check=False)
                    self.assertEqual(completed.returncode, 2)
                    self.assertIn("must not overwrite", completed.stderr)
                    self.assertEqual(output.read_bytes(), original)

    def test_cli_fails_cleanly_for_unknown_prediction_id(self):
        tasks, answers = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_jsonl(root / "questions.jsonl", tasks)
            write_jsonl(root / "answers.jsonl", answers.values())
            write_jsonl(root / "predictions.jsonl", [{"task_id": "wrong-id", "selected_option_id": "A"}])
            completed = subprocess.run([
                sys.executable, str(BUNDLE_ROOT / "eval/mcqa.py"),
                "--predictions", str(root / "predictions.jsonl"),
                "--questions", str(root / "questions.jsonl"),
                "--answers", str(root / "answers.jsonl"),
            ], text=True, capture_output=True, check=False)
            self.assertEqual(completed.returncode, 2)
            self.assertIn("unknown prediction task_id", completed.stderr)
            self.assertEqual(completed.stdout, "")


if __name__ == "__main__":
    unittest.main()
