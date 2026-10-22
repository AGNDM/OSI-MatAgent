"""Public proposal/gate/choice contracts and actual native correction replay."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest

from comparison.agentic import ExpertDecisionBackend, expert_messages, parse_proposal
from comparison.policies import load_bundle
from comparison.run import run_sequential, validate_config


def proposal(ids):
    return {"diagnosis": "Use current public instrument evidence.",
            "evidence_checks": ["Check public command prerequisites and observation sample times."],
            "candidates": [{"option_id": identifier, "rationale": "This is an offered public option."} for identifier in ids]}


class FakeExpert:
    identity = {"kind": "test-local-hf-expert", "diagnostic": True}
    last_generation_receipt = {"max_new_tokens": 384, "diagnostic": True}
    def __init__(self, raw=None, error=None):
        self.raw, self.error, self.calls = raw, error, []
    def _generate(self, messages, **kwargs):
        self.calls.append(copy.deepcopy(messages))
        if self.error:
            raise RuntimeError(self.error)
        if self.raw is not None:
            return self.raw, 100, 25
        offered = json.loads(messages[-1]["content"].split("Offered public options:\n", 1)[1].split("\nReturn the required", 1)[0])
        ids = [option["option_id"] for option in offered[:4]]
        return json.dumps(proposal(ids)), 100, 25


class FakeChooser:
    identity = {"kind": "test-local-chooser", "diagnostic": True}
    def __init__(self, error=False):
        self.calls, self.error = [], error
    def decide(self, messages, option_ids, **kwargs):
        self.calls.append((copy.deepcopy(messages), list(option_ids), copy.deepcopy(kwargs)))
        if self.error:
            return {"status": "error", "error": "unsupported native context", "input_tokens": None,
                    "output_tokens": None, "metadata": {"error_kind": "infrastructure"}}
        return {"status": "ok", "selected_option_id": option_ids[0],
                "probabilities": {label: 1 / len(option_ids) for label in option_ids},
                "confidence": 1 / len(option_ids), "input_tokens": 40, "output_tokens": 0,
                "metadata": {"probability_semantics": "test conditional menu probabilities"}}


def experiment():
    return {"name": "agentic-test", "architecture": "expert_decision", "expert": {"kind": "hf"},
            "backend": {"kind": "native"}, "method": "native", "max_candidates": 4}


class ProposalGateTests(unittest.TestCase):
    def test_exact_public_candidate_set_gate(self):
        value = parse_proposal(json.dumps(proposal(["A", "B"])), ["A", "B"])
        self.assertEqual(len(value["candidates"]), 2)

    def test_unknown_and_duplicate_candidates_rejected(self):
        for ids in (["A", "A"], ["A", "Z"]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                parse_proposal(json.dumps(proposal(ids)), ["A", "B"])

    def test_single_candidate_collapse_requires_declared_variable_protocol(self):
        raw = json.dumps(proposal(["A"]))
        with self.assertRaises(ValueError):
            parse_proposal(raw, ["A", "B"])
        self.assertEqual(len(parse_proposal(raw, ["A", "B"], variable=True)["candidates"]), 1)

    def test_non_json_and_extra_fields_rejected(self):
        for raw in ("```json\n{}\n```", json.dumps({**proposal(["A"]), "hidden_reference": "A"})):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_proposal(raw, ["A"])

    def test_preserves_original_public_history_and_messages(self):
        original = [{"role": "system", "content": "original instruction"},
                    {"role": "user", "content": "all observations and full temporal history marker"}]
        result = expert_messages(original, ["A"], {"A": "public action"})
        self.assertEqual(result[1], original[1])
        self.assertEqual(original[0]["content"], "original instruction")
        self.assertIn("overrides earlier", result[0]["content"])

    def test_config_rejects_remote_expert_and_invalid_size(self):
        for changes in ({"expert": {"kind": "openai"}}, {"max_candidates": 5}, {"max_candidates": True}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_config({"schema_version": 1, "experiments": [{**experiment(), **changes}]})


class AgenticBackendTests(unittest.TestCase):
    def request(self, expert=None, chooser=None):
        expert, chooser = expert or FakeExpert(), chooser or FakeChooser()
        backend = ExpertDecisionBackend(experiment(), expert_backend=expert, chooser_backend=chooser)
        messages = [{"role": "system", "content": "public evidence system"},
                    {"role": "user", "content": "full observation and action history"}]
        result = backend.decide(messages, ["A", "B", "C", "D", "E"],
            option_descriptions={label: f"public action {label}" for label in "ABCDE"}, method="native")
        return result, expert, chooser, messages

    def test_two_stages_aggregate_real_usage_and_keep_conditional_vector_separate(self):
        result, expert, chooser, original = self.request()
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["input_tokens"], 140)
        self.assertEqual(result["output_tokens"], 25)
        self.assertIsNone(result["probabilities"])
        trace = result["metadata"]["agentic_trace"]
        self.assertEqual(trace["shortlist_ids"], list("ABCD"))
        self.assertEqual(set(trace["shortlist_probabilities"]), set("ABCD"))
        self.assertEqual(chooser.calls[0][0][:2], original)
        self.assertEqual(chooser.calls[0][1], list("ABCD"))
        self.assertFalse(trace["gold_information_used"])

    def test_malformed_expert_reply_counts_model_error_without_chooser_call(self):
        chooser = FakeChooser()
        result, _, _, _ = self.request(FakeExpert(raw="not JSON"), chooser)
        self.assertEqual(result["metadata"]["error_kind"], "model_output")
        self.assertEqual(result["input_tokens"], 100)
        self.assertEqual(result["output_tokens"], 25)
        self.assertEqual(chooser.calls, [])

    def test_expert_context_error_is_infrastructure(self):
        result, _, chooser, _ = self.request(FakeExpert(error="context overflow"))
        self.assertEqual(result["metadata"]["error_kind"], "infrastructure")
        self.assertEqual(chooser.calls, [])

    def test_native_chooser_failure_does_not_become_model_error(self):
        result, _, _, _ = self.request(chooser=FakeChooser(error=True))
        self.assertEqual(result["metadata"]["error_kind"], "infrastructure")
        self.assertIsNone(result["input_tokens"])


class AgenticNativeReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = Path(os.environ.get("COMPARISON_TEST_BUNDLE", Path(__file__).resolve().parents[2] / "instrument")).resolve()
        if not cls.bundle.is_dir():
            raise unittest.SkipTest("Set COMPARISON_TEST_BUNDLE to the frozen instrument release")
        cls.helpers = load_bundle(cls.bundle)

    def run_episode(self, expert):
        backend = ExpertDecisionBackend(experiment(), expert_backend=expert, chooser_backend=FakeChooser())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            run_sequential(self.bundle, self.helpers, backend, experiment(), {"mode": "teacher_forced"}, output, 1)
            return self.helpers["common"].load_jsonl(output / "episodes.jsonl")[0]

    def test_native_execution_corrections_and_posthoc_private_reference_audit(self):
        expert = FakeExpert()
        episode = self.run_episode(expert)
        self.assertEqual(episode["status"], "ok", episode.get("error"))
        self.assertTrue(episode["trajectory_outcome_met"])
        self.assertTrue(episode["posthoc_expert_reference_coverage"]["private_replay_verified"])
        self.assertGreater(episode["posthoc_expert_reference_coverage"]["reference_offered_events"], 0)
        for messages in expert.calls:
            self.assertNotIn("posthoc_reference_candidate_coverage", json.dumps(messages))
        events = [event for record in episode["decision_records"] for event in record["usage"]["selection_events"]]
        self.assertTrue(any(event["backend_result"]["metadata"]["agentic_trace"]["stages"][-1]["stage"] == "native_execute_and_verify" for event in events))

    def test_malformed_expert_json_is_corrected_then_native_goal_verified(self):
        episode = self.run_episode(FakeExpert(raw="invalid proposal"))
        self.assertEqual(episode["status"], "ok", episode.get("error"))
        self.assertTrue(episode["trajectory_outcome_met"])
        self.assertEqual(episode["control_decision_accuracy"], 0)
        self.assertEqual(episode["posthoc_expert_reference_coverage"]["reference_shortlist_coverage"], 0)


if __name__ == "__main__":
    unittest.main()
