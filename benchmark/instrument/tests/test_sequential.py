"""Sequential scoring and public policy-boundary tests; run within Slurm."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

BUNDLE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUNDLE_ROOT))

from eval.sequential import MODES, OfferedChoicePolicy, score_episodes


def tasks():
    return [
        {"task_id": "one", "workflow_id": "alpha", "instrument_id": "furnace", "expansion": {"condition": "normal"}},
        {"task_id": "two", "workflow_id": "alpha", "instrument_id": "furnace", "expansion": {"condition": "fault"}},
        {"task_id": "three", "workflow_id": "beta", "instrument_id": "pump", "expansion": {"condition": "normal"}},
    ]


def episode(task_id="one", mode="teacher_forced", grades=(True, False, True),
            queries=(False, False, True), outcome=True):
    records = []
    for index, (correct, query) in enumerate(zip(grades, queries), 1):
        records.append({"decision_index": index, "proposal_correct": correct,
                        "auxiliary_query": query, "corrected": mode == "teacher_forced" and not correct,
                        "checkpoint_id": None if query else f"checkpoint-{index}",
                        "stage": None if query else "operate", "proposed_action": {"kind": "read"},
                        "executed_action": {"kind": "read"}, "response": {"status": "ok"}})
    return {"task_id": task_id, "mode": mode, "status": "ok", "decision_records": records,
            "trajectory_outcome_met": outcome, "menu_membership_verified": True,
            "autonomous_success": outcome if mode == "free_run" else None,
            "corrected_trajectory_outcome_met": outcome if mode == "teacher_forced" else None,
            "decisions": len(records), "interventions": sum(row["corrected"] for row in records)}


class SequentialScoreTests(unittest.TestCase):
    def test_missing_and_infrastructure_keep_full_cohort(self):
        rows = [episode(), {"task_id": "two", "mode": "teacher_forced",
                            "status": "infrastructure_error", "error": "backend unavailable"}]
        report = score_episodes(tasks(), rows)
        scored = report["by_mode"]["teacher_forced"]
        self.assertEqual((scored["n"], scored["completed"], scored["missing"], scored["infrastructure_failures"]), (3, 1, 1, 1))
        self.assertEqual(scored["coverage"], 1 / 3)
        self.assertEqual(scored["assisted_goal_completion_rate"], 1 / 3)
        self.assertEqual(scored["completed_episode_goal_rate"], 1)
        self.assertEqual(scored["decisions_observed"], 3)
        self.assertFalse(report["complete"])
        self.assertEqual(report["expected_episodes"], 6)
        self.assertEqual(report["by_mode"]["free_run"]["missing"], 3)

    def test_original_proposals_and_control_queries_are_separate(self):
        report = score_episodes(tasks(), [episode()], ["teacher_forced"])["by_mode"]["teacher_forced"]
        self.assertEqual(report["observed_proposal_accuracy_micro"], 2 / 3)
        self.assertEqual(report["observed_control_accuracy_micro"], 1 / 2)
        self.assertEqual(report["observed_query_admissibility"], 1)
        self.assertEqual(report["interventions"], 1)
        self.assertEqual(report["by_stage"]["operate"]["observed_accuracy"], 1 / 2)

    def test_assisted_completion_is_not_autonomous(self):
        report = score_episodes(tasks()[:1], [episode(), episode(mode="free_run", outcome=False)])
        self.assertTrue(report["complete"])
        self.assertEqual(report["by_mode"]["teacher_forced"]["assisted_goal_completion_rate"], 1)
        self.assertNotIn("autonomous_success_rate", report["by_mode"]["teacher_forced"])
        self.assertEqual(report["by_mode"]["free_run"]["autonomous_success_rate"], 0)
        self.assertEqual(report["by_mode"]["free_run"]["interventions"], 0)

    def test_workflow_macro_uses_equal_workflow_weight(self):
        rows = [episode("one", grades=(False,), queries=(False,), outcome=False),
                episode("two", grades=(False,), queries=(False,), outcome=False),
                episode("three", grades=(True,), queries=(False,))]
        report = score_episodes(tasks(), rows, ["teacher_forced"])["by_mode"]["teacher_forced"]
        self.assertEqual(report["observed_proposal_accuracy_micro"], 1 / 3)
        self.assertEqual(report["observed_workflow_macro_proposal_accuracy"], 1 / 2)
        self.assertEqual(report["workflow_macro_goal_completion_rate"], 1 / 2)
        self.assertEqual(report["by_instrument_id"]["furnace"]["completed"], 2)
        self.assertEqual(report["by_condition"]["fault"]["completed"], 1)

    def test_episode_macro_does_not_pool_longer_episodes(self):
        rows = [episode("one", grades=(False,), queries=(False,)),
                episode("two", grades=(True, True, True), queries=(False, False, False))]
        report = score_episodes(tasks()[:2], rows, ["teacher_forced"])["by_mode"]["teacher_forced"]
        self.assertEqual(report["observed_proposal_accuracy_micro"], 3 / 4)
        self.assertEqual(report["observed_proposal_accuracy_episode_macro"], 1 / 2)

    def test_repeated_checkpoint_requires_every_proposal_correct(self):
        row = episode(grades=(False, True), queries=(False, False))
        row["decision_records"][1]["checkpoint_id"] = "checkpoint-1"
        report = score_episodes(tasks()[:1], [row], ["teacher_forced"])["by_mode"]["teacher_forced"]
        self.assertEqual(report["observed_checkpoint_accuracy_episode_macro"], 0)

    def test_unknown_and_duplicate_records_rejected(self):
        for rows, message in (([episode("unknown")], "unknown episode"),
                              ([episode(), episode()], "duplicate episode")):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                score_episodes(tasks(), rows)

    def test_modes_validated_even_if_not_scored(self):
        row = episode(mode="free_run")
        row["mode"] = "typo"
        with self.assertRaisesRegex(ValueError, "unknown episode mode"):
            score_episodes(tasks(), [row], ["teacher_forced"])
        for modes in ([], ["typo"], ["free_run", "free_run"], [["free_run"]]):
            with self.subTest(modes=modes), self.assertRaises(ValueError):
                score_episodes(tasks(), [], modes)

    def test_other_known_mode_ignored_and_counted(self):
        report = score_episodes(tasks()[:1], [episode(), episode(mode="free_run")], ["teacher_forced"])
        self.assertTrue(report["complete"])
        self.assertEqual(report["ignored_other_mode_rows"], 1)

    def test_inconsistent_flags_rejected(self):
        mutators = (
            lambda row: row.update(menu_membership_verified=False),
            lambda row: row.update(trajectory_outcome_met=1),
            lambda row: row.update(interventions=9),
            lambda row: row.update(decisions=9),
            lambda row: row.update(corrected_trajectory_outcome_met=False),
            lambda row: row.update(autonomous_success=True),
            lambda row: row["decision_records"][0].update(proposal_correct=1),
            lambda row: row["decision_records"][0].update(corrected=True),
            lambda row: row["decision_records"][1].update(decision_index=1),
            lambda row: row["decision_records"][0].update(stage=None),
            lambda row: row.update(termination="invalid_curriculum_correction"),
        )
        for mutate in mutators:
            row = episode()
            mutate(row)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                score_episodes(tasks(), [row])

    def test_empty_coverage_has_null_step_accuracy(self):
        report = score_episodes(tasks(), [])["by_mode"]["teacher_forced"]
        self.assertEqual(report["cohort_goal_completion_rate"], 0)
        self.assertIsNone(report["observed_proposal_accuracy_micro"])
        self.assertIsNone(report["completed_episode_goal_rate"])
        self.assertEqual(report["workflows_with_observations"], 0)

    def test_duplicate_or_empty_tasks_rejected(self):
        for data in ([], [tasks()[0], tasks()[0]]):
            with self.subTest(data=data), self.assertRaises(ValueError):
                score_episodes(data, [])

    def test_summary_fields_do_not_override_original_record_scores(self):
        row = episode()
        row.update(checkpoint_accuracy=1, control_decision_accuracy=1, decision_accuracy_all_proposals=1)
        report = score_episodes(tasks()[:1], [row], ["teacher_forced"])["by_mode"]["teacher_forced"]
        self.assertEqual(report["observed_control_accuracy_micro"], 1 / 2)


class Menus:
    def action_options(self, view, history):
        return [{"text": "command", "kind": "command", "capability_id": "power.set"},
                {"text": "finish", "action": {"kind": "finish"}}]

    def argument_options(self, view, history, option):
        return [{"text": "off", "action": {"kind": "invoke", "arguments": {"on": False}}},
                {"text": "on", "action": {"kind": "invoke", "arguments": {"on": True}}}]


class PolicyBoundaryTests(unittest.TestCase):
    def test_command_and_argument_choices_form_one_action(self):
        class Plugin:
            last_action_usage = {"input_tokens": 123}
            seen = []
            def choose(self, view, history, options):
                self.seen.append((copy.deepcopy(view), copy.deepcopy(history), copy.deepcopy(options)))
                return 0 if options[0]["text"] == "command" else 1
        plugin = Plugin()
        policy = OfferedChoicePolicy(plugin, Menus())
        view, history = {"goal": "operate"}, [{"response": {"status": "ok"}}]
        result = policy.next_action(view, history)
        self.assertEqual(result, {"kind": "invoke", "arguments": {"on": True}})
        self.assertEqual(len(plugin.seen), 2)
        self.assertEqual(policy.last_action_usage["input_tokens"], 123)
        self.assertEqual(set(plugin.seen[0][0]), {"goal"})

    def test_direct_action_needs_no_parameter_choice(self):
        class Plugin:
            def choose(self, view, history, options):
                return 1
        self.assertEqual(OfferedChoicePolicy(Plugin(), Menus()).next_action({}, []), {"kind": "finish"})

    def test_invalid_indices_not_coerced(self):
        for value in (True, -1, 2, "A", None):
            class Plugin:
                def choose(self, view, history, options):
                    return value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "offered zero-based index"):
                OfferedChoicePolicy(Plugin(), Menus()).next_action({}, [])

    def test_native_action_plugin_receives_copies(self):
        class Plugin:
            def next_action(self, view, history):
                view.clear()
                history.clear()
                return {"kind": "finish"}
        view, history = {"goal": "operate"}, [{"response": "feedback"}]
        self.assertEqual(OfferedChoicePolicy(Plugin(), Menus()).next_action(view, history), {"kind": "finish"})
        self.assertEqual(view, {"goal": "operate"})
        self.assertEqual(len(history), 1)


class CLIAndSchemaTests(unittest.TestCase):
    def test_stdlib_score_cli_from_other_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            task_path, episodes_path = directory / "tasks.jsonl", directory / "episodes.jsonl"
            task_path.write_text(json.dumps(tasks()[0]) + "\n", encoding="utf-8")
            episodes_path.write_text("\n".join(json.dumps(episode(mode=mode)) for mode in MODES) + "\n", encoding="utf-8")
            output_path = directory / "scores.json"
            result = subprocess.run([sys.executable, str(BUNDLE_ROOT / "eval/sequential.py"), "score",
                                     "--tasks", str(task_path), "--episodes", str(episodes_path),
                                     "--output", str(output_path)], cwd=directory, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(json.loads(output_path.read_text())["complete"])

    def test_schemas_are_json_and_document_interactive_record(self):
        schema = json.loads((BUNDLE_ROOT / "eval/schemas/sequential-episode.schema.json").read_text())
        self.assertIn("mode", schema["required"])
        self.assertIn("decision_records", schema["properties"])
        schema = json.loads((BUNDLE_ROOT / "eval/schemas/sequential-policy-choice.schema.json").read_text())
        self.assertEqual(schema["type"], "integer")


if __name__ == "__main__":
    unittest.main()
