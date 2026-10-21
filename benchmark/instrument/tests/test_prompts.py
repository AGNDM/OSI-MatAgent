"""Public evidence boundary and serialization regression checks (stdlib only)."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from prompts import expand_text_record, export_prompts, solver_view, task_text


def question(task_id="case-a", split="test"):
    error = {"code": "LOCKED", "meaning_zh": "请先解锁", "side_effects": "none"}
    capability = {
        "input_schema": {"type": "object", "properties": {"on": {"type": "boolean"}}},
        "preconditions": [{"id": "unlocked", "state_ids": ["controls.locked"]}],
        "description_zh": "启动设备",
        "errors": [error],
        "observes": ["motor.running"],
        "output_schema": {"type": "object"},
    }
    return {
        "task_id": task_id,
        "group_id": "group-private-to-evaluation",
        "split": split,
        "difficulty": "L4",
        "tier": "T3",
        "seed": 123,
        "correct_option_ids": ["A"],
        "candidate_results": [{"option_id": "A", "valid": True}],
        "instrument_id": "test-instrument",
        "question": "Which action establishes the operating goal?",
        "options": [
            {"option_id": "A", "text": "Start", "steps": [
                {"kind": "invoke", "capability_id": "motor.start", "arguments": {"on": True}}]},
            {"option_id": "B", "text": "Stop", "steps": [
                {"kind": "invoke", "capability_id": "motor.stop", "arguments": {}}]},
        ],
        "goal": {"objective": "Run the motor", "proposed_steps": [
            {"kind": "invoke", "capability_id": "controls.unlock", "arguments": {}}]},
        "manual": ["An accepted command alone does not prove completion."],
        "history": [],
        "observation": {
            "instance_id": "runtime-instance",
            "request_id": "request-3",
            "state_revision": 7,
            "clock": {"time_s": 14.0, "domain": "simulation"},
            "observations": {
                "temperature": {"value": 303.15, "unit": "K", "quality": "estimated",
                                "sample_time_s": 12.0, "sample_window_s": 2.0},
                "motor.running": {"value": False, "unit": "1", "quality": "valid",
                                  "sample_time_s": 14.0, "sample_window_s": None},
            },
            "error": None,
        },
        "contract": {
            "capabilities": {
                "motor.start": copy.deepcopy(capability),
                "motor.stop": copy.deepcopy(capability),
                "controls.unlock": copy.deepcopy(capability),
                "unrelated.operation": copy.deepcopy(capability),
            },
            "state_slots": {"temperature": {"meaning_zh": "温度", "reference_conditions": {"unit": "K"},
                                               "internal_annotation": "not public prompt content"}},
        },
    }


class PromptTests(unittest.TestCase):
    def test_excludes_labels_and_evaluation_metadata(self):
        view = solver_view(question())
        for field in ("task_id", "group_id", "split", "difficulty", "tier", "seed",
                      "correct_option_ids", "candidate_results"):
            self.assertNotIn(field, view)
        self.assertNotIn("instance_id", view["observation"])
        self.assertNotIn("request_id", view["observation"])
        self.assertNotIn("state_revision", view["observation"])
        self.assertEqual(view["input_protocol"], "public-menu-union-v2")

    def test_preserves_observation_values_quality_units_and_time(self):
        task = question()
        view = solver_view(task)
        self.assertEqual(view["observation"]["observations"], task["observation"]["observations"])
        self.assertEqual(view["observation"]["clock"], task["observation"]["clock"])
        packed = json.loads(task_text(task))
        self.assertEqual(packed["observation"]["columns"],
                         ["value", "unit", "quality", "sample_time_s", "sample_window_s"])
        self.assertEqual(packed["observation"]["readings"]["temperature"],
                         [303.15, "K", "estimated", 12.0, 2.0])

    def test_preserves_docs_for_all_offered_candidates_and_proposed_steps(self):
        task = question()
        view = solver_view(task)
        self.assertEqual(set(view["contract"]["capabilities"]),
                         {"motor.start", "motor.stop", "controls.unlock"})
        for capability_id, docs in view["contract"]["capabilities"].items():
            original = task["contract"]["capabilities"][capability_id]
            for field in ("input_schema", "preconditions", "description_zh", "errors", "observes"):
                self.assertEqual(docs[field], original[field])
            self.assertNotIn("output_schema", docs)
        self.assertNotIn("internal_annotation", view["contract"]["state_slots"]["temperature"])

    def test_packing_roundtrip_retains_shared_error_order_and_readings(self):
        task = question()
        second_error = {"code": "INVALID_ARGUMENT", "meaning_zh": "参数无效", "side_effects": "none"}
        task["contract"]["capabilities"]["motor.start"]["errors"].append(second_error)
        task["contract"]["capabilities"]["motor.stop"]["errors"].insert(0, second_error)
        packed = json.loads(task_text(task))
        self.assertEqual(len(packed["contract"]["error_definitions"]), 2)
        self.assertEqual(expand_text_record(packed), solver_view(task))
        self.assertEqual(task_text(packed), task_text(task))

    def test_serializing_raw_and_packed_inputs_never_mutates_them(self):
        task = question()
        before = copy.deepcopy(task)
        full = task_text(task)
        without_options = json.loads(task_text(task, include_options=False))
        self.assertNotIn("options", without_options)
        self.assertEqual(task, before)
        packed = json.loads(full)
        packed_before = copy.deepcopy(packed)
        self.assertEqual(task_text(packed), full)
        self.assertEqual(packed, packed_before)

    def test_interactive_view_keeps_all_public_tools(self):
        task = question()
        view = solver_view(task, interactive=True)
        self.assertEqual(set(view["contract"]["capabilities"]), set(task["contract"]["capabilities"]))
        self.assertEqual(view["input_protocol"], "public-tools-v2")

    def test_export_defaults_to_all_heldout_splits_without_metadata_in_prompt(self):
        tasks = [question("train-case", "train"), question("test-case", "test"),
                 question("fault-case", "test_fault"), question("validation-case", "validation")]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "prompts.jsonl"
            self.assertEqual(export_prompts(tasks, output), 2)
            records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([record["task_id"] for record in records], ["test-case", "fault-case"])
            for record in records:
                self.assertEqual(set(record), {"task_id", "prompt"})
                self.assertNotIn("task_id", json.loads(record["prompt"]))
            self.assertEqual(export_prompts(tasks, output, "all"), 4)
            self.assertEqual(export_prompts(tasks, output, "validation"), 1)

    def test_export_rejects_duplicates_unknown_split_and_empty_cohort(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "prompts.jsonl"
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                export_prompts([question(), question()], output)
            with self.assertRaisesRegex(ValueError, "Unknown split"):
                export_prompts([question()], output, "nonexistent")
            with self.assertRaisesRegex(ValueError, "No questions selected"):
                export_prompts([question(split="train")], output)


if __name__ == "__main__":
    unittest.main()
