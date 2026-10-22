"""Contract tests; run through Slurm, without downloading or loading weights."""
import os
from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from comparison.backends import (
    DecisionBackend, HuggingFaceBackend, OllamaBackend, _completion_decision,
    _loopback_url, _messages, _parse_verbalized, _sample_completions,
    _generation_kwargs, _official_logits, build_backend, require_slurm,
)


class RecordingBackend(DecisionBackend):
    def _decide(self, messages, ids, method, samples, seed, temperature):
        self.seen = messages
        return {"selected_option_id": ids[0], "probabilities": {ids[0]: .7, ids[1]: .3},
                "confidence": .7, "input_tokens": 123, "output_tokens": 0,
                "metadata": {"probability_semantics": "test"}}


class BackendContractTests(unittest.TestCase):
    def setUp(self):
        self.messages = [{"role": "system", "content": "public operating rules"},
                         {"role": "user", "content": "goal, observations, full prior history"}]

    def test_requires_compute_allocation(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "Slurm"):
                require_slurm()
            with self.assertRaisesRegex(RuntimeError, "Slurm"):
                HuggingFaceBackend({"model_path": "/does/not/exist"})

    def test_choice_contract_preserves_public_evidence(self):
        backend = RecordingBackend({"method": "choice"})
        result = backend.decide(self.messages, ["A", "B"])
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["input_tokens"], 123)
        self.assertTrue(backend.seen[0]["content"].startswith(self.messages[0]["content"]))
        self.assertEqual(self.messages[0]["content"], "public operating rules")
        self.assertTrue(backend.seen[1]["content"].startswith(self.messages[1]["content"]))
        self.assertEqual(self.messages[1]["content"], "goal, observations, full prior history")
        self.assertEqual(result["metadata"]["context_policy"], "error_without_truncation")

    def test_readout_instruction_overrides_format_at_system_priority(self):
        messages = [{"role": "system", "content": "Always reply with one letter."},
                    {"role": "user", "content": "all observations"}]
        prepared = _messages(messages, ["A", "B"], "verbalized")
        self.assertEqual(sum(message["role"] == "system" for message in prepared), 1)
        self.assertIn("overrides earlier response-format instructions only", prepared[0]["content"])
        self.assertIn("JSON object", prepared[0]["content"])
        self.assertTrue(prepared[0]["content"].startswith("Always reply with one letter."))

    def test_readout_adds_only_one_system_when_caller_has_none(self):
        prepared = _messages([{"role": "user", "content": "all public context"}], ["A", "B"], "verbalized")
        self.assertEqual([message["role"] for message in prepared], ["system", "user"])
        self.assertIn("JSON object", prepared[0]["content"])
        plain = _messages(self.messages, ["A", "B"], "choice")
        self.assertNotIn("JSON object", plain[0]["content"])

    def test_compatibility_choose_has_same_scores(self):
        result = RecordingBackend({}).choose(self.messages, ["A", "B"], request_id="x")
        self.assertEqual(result["index"], 0)
        self.assertEqual(result["probabilities"], [.7, .3])
        self.assertEqual(result["usage"]["input_tokens"], 123)

    def test_requires_full_option_description_coverage(self):
        with self.assertRaises(ValueError):
            RecordingBackend({}).decide(self.messages, ["A", "B"], option_descriptions={"A": "one"})

    def test_sampling_needs_ten_draws(self):
        backend = RecordingBackend({})
        for n in [0, 9, True, 10.0]:
            with self.assertRaisesRegex(ValueError, "ten"):
                backend.decide(self.messages, ["A", "B"], method="sampling", samples=n)

    def test_confidence_does_not_invent_residual_probability_mass(self):
        result = _completion_decision('{"selected_option_id":"B","confidence":0.8}',
                                      100, 15, ["A", "B", "C"], "verbalized", 0)
        self.assertEqual(result["selected_option_id"], "B")
        self.assertEqual(result["confidence"], .8)
        self.assertIsNone(result["probabilities"])

    def test_citations_are_a_separately_declared_verbalized_protocol(self):
        plain = _messages(self.messages, ["A", "B"], "verbalized")
        explicit_plain = _messages(self.messages, ["A", "B"], "verbalized", cite_causes=False)
        cited = _messages(self.messages, ["A", "B"], "verbalized", cite_causes=True)
        self.assertEqual(plain, explicit_plain)
        self.assertNotIn("causes_cited", plain[0]["content"])
        self.assertIn("causes_cited", cited[0]["content"])
        self.assertTrue(cited[1]["content"].startswith(self.messages[1]["content"]))

    def test_cited_confidence_keeps_explicit_identifiers_without_fabricating_vector(self):
        result = _completion_decision('{"selected_option_id":"A","confidence":0.8,"causes_cited":["power.on","POWER_OFF"]}',
            100, 20, ["A", "B"], "verbalized", 0, cite_causes=True)
        self.assertEqual(result["causes_cited"], ["power.on", "POWER_OFF"])
        self.assertIsNone(result["probabilities"])
        self.assertEqual(result["input_tokens"], 100)
        self.assertEqual(result["metadata"]["citation_protocol"], "explicit_public_identifier_array-v1")

    def test_cited_variant_requires_well_formed_citation_array(self):
        for suffix in ('', ',"causes_cited":null', ',"causes_cited":[true]', ',"causes_cited":["x","x"]'):
            result = _completion_decision('{"selected_option_id":"A","confidence":0.8' + suffix + '}',
                100, 20, ["A", "B"], "verbalized", 0, cite_causes=True)
            self.assertEqual(result["status"], "error")
            self.assertEqual(result["metadata"]["error_kind"], "model_output")
            self.assertEqual(result["output_tokens"], 20)

    def test_cited_variant_cannot_change_label_or_sampling_protocol(self):
        backend = RecordingBackend({"cite_causes": True})
        for method in ("choice", "logprobs", "sampling"):
            with self.subTest(method=method), self.assertRaisesRegex(ValueError, "verbalized-only"):
                backend.decide(self.messages, ["A", "B"], method=method)

    def test_cited_wrapper_rejects_a_backend_result_that_omits_citations(self):
        result = RecordingBackend({"cite_causes": True}).decide(self.messages, ["A", "B"], method="verbalized")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["metadata"]["error_kind"], "model_output")
        self.assertEqual(result["input_tokens"], 123)

    def test_malformed_json_keeps_real_token_usage(self):
        result = _completion_decision('answer is B', 100, 15, ["A", "B"], "verbalized", 0)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_type"], "model_output_invalid")
        self.assertEqual(result["input_tokens"], 100)
        self.assertEqual(result["output_tokens"], 15)
        self.assertIsNone(result["selected_option_id"])

    def test_confidence_values_strict(self):
        for value in ['true', 'null', 'NaN', '1.1', '-0.1', '"0.4"']:
            with self.assertRaises((ValueError, TypeError)):
                _parse_verbalized('{"selected_option_id":"A","confidence":' + value + '}', ["A", "B"])

    def test_sampling_records_all_draws_and_usage(self):
        calls = []
        def generate(messages, **kwargs):
            calls.append(kwargs)
            return ("A" if len(calls) <= 6 else "B"), 100, 2
        result = _sample_completions(generate, self.messages, ["A", "B"], 10, 5, .8)
        self.assertEqual(result["probabilities"], {"A": .6, "B": .4})
        self.assertEqual(result["input_tokens"], 1000)
        self.assertEqual(result["output_tokens"], 20)
        self.assertEqual(len({call["seed"] for call in calls}), 10)

    def test_sampling_never_renormalizes_away_invalid_draws(self):
        outputs = iter(["A"] * 9 + ["not an option"])
        result = _sample_completions(lambda *args, **kwargs: (next(outputs), 10, 1),
                                     self.messages, ["A", "B"], 10, 0, .8)
        self.assertEqual(result["status"], "error")
        self.assertIsNone(result["probabilities"])
        self.assertEqual(result["metadata"]["invalid_draws"], 1)
        self.assertEqual(result["metadata"]["error_kind"], "model_output")
        self.assertEqual(len(result["metadata"]["draws"]), 10)

    def test_sampling_preserves_all_draw_errors_and_unknown_usage(self):
        outputs = iter(["A"] * 8 + [None, "B"])
        def generate(*args, **kwargs):
            value = next(outputs)
            if value is None:
                raise RuntimeError("backend request failed")
            return value, 10, 1
        result = _sample_completions(generate, self.messages, ["A", "B"], 10, 0, .8)
        self.assertEqual(len(result["metadata"]["draws"]), 10)
        self.assertEqual(result["metadata"]["valid_draws"], 9)
        self.assertEqual(result["metadata"]["winning_vote_fraction"], .8)
        self.assertIsNone(result["input_tokens"])
        self.assertIsNone(result["probabilities"])
        self.assertEqual(result["error_type"], "infrastructure_error")

    def test_sampling_handles_zero_vote_choices(self):
        result = _sample_completions(lambda *args, **kwargs: ("A", 10, 1),
                                     self.messages, ["A", "B", "C"], 10, 0, .8)
        self.assertEqual(result["probabilities"], {"A": 1.0, "B": 0.0, "C": 0.0})
        self.assertEqual(result["metadata"]["winning_vote_fraction"], 1.0)

    def test_official_forward_keeps_model_specific_logit_processing(self):
        calls = []
        class SoftcappedModel:
            @property
            def model(self):
                raise AssertionError("Do not bypass the released causal-LM forward")
            def get_output_embeddings(self):
                raise AssertionError("The bare head omits model-specific postprocessing")
            def forward(self, input_ids, use_cache=True, return_dict=False, logits_to_keep=0):
                calls.append((input_ids, use_cache, return_dict, logits_to_keep))
                # Sentinel represents logits after a model-specific softcap.
                return SimpleNamespace(logits={"postprocessed": "softcapped"})
            def __call__(self, **kwargs):
                return self.forward(**kwargs)
        logits, limit = _official_logits(SoftcappedModel(), {"input_ids": "public tokens"})
        self.assertEqual(logits, {"postprocessed": "softcapped"})
        self.assertEqual(calls, [("public tokens", False, True, 1)])
        self.assertEqual(limit, "logits_to_keep")

    def test_official_forward_supports_older_logit_limit_argument(self):
        class OlderModel:
            def forward(self, input_ids, use_cache=True, return_dict=False, num_logits_to_keep=0):
                self.limit = num_logits_to_keep
                return SimpleNamespace(logits="official logits")
            def __call__(self, **kwargs):
                return self.forward(**kwargs)
        model = OlderModel()
        logits, limit = _official_logits(model, {"input_ids": [1, 2]})
        self.assertEqual(model.limit, 1)
        self.assertEqual(limit, "num_logits_to_keep")
        self.assertEqual(logits, "official logits")

    def test_official_forward_without_limit_keeps_released_computation(self):
        class BasicModel:
            def forward(self, input_ids, use_cache=True, return_dict=False):
                return SimpleNamespace(logits="full official logits")
            def __call__(self, **kwargs):
                return self.forward(**kwargs)
        logits, limit = _official_logits(BasicModel(), {"input_ids": [1, 2]})
        self.assertEqual(logits, "full official logits")
        self.assertEqual(limit, "full_forward_output")

    def test_greedy_generation_overrides_checkpoint_repetition_penalty(self):
        calls = []
        class Suffix:
            def numel(self):
                return 1
        class Generated:
            def __getitem__(self, key):
                return Suffix()
        class Model:
            generation_config = SimpleNamespace(to_dict=lambda: {"repetition_penalty": 1.05,
                                                                  "top_k": 20, "top_p": .8})
            def generate(self, **kwargs):
                calls.append(kwargs)
                return Generated()
        backend = HuggingFaceBackend.__new__(HuggingFaceBackend)
        backend.model, backend.device, backend.max_new_tokens = Model(), "cpu", 192
        backend.tokenizer = SimpleNamespace(eos_token_id=151645, decode=lambda *args, **kwargs: "A")
        backend.torch = SimpleNamespace(random=SimpleNamespace(fork_rng=lambda **kwargs: nullcontext()),
                                         inference_mode=lambda: nullcontext(), manual_seed=lambda seed: None)
        backend._inputs = lambda *args: ({"input_ids": "complete public input"}, 10)
        raw, input_tokens, output_tokens = backend._generate(self.messages)
        self.assertEqual((raw, input_tokens, output_tokens), ("A", 10, 1))
        self.assertEqual(calls[0]["repetition_penalty"], 1.0)
        self.assertEqual(calls[0]["top_k"], 0)
        self.assertEqual(calls[0]["top_p"], 1.0)
        self.assertIs(calls[0]["do_sample"], False)
        self.assertEqual(backend.last_generation_receipt["effective_generation_config"]["repetition_penalty"], 1.0)
        self.assertEqual(len(backend.last_generation_receipt["effective_generation_config_sha256"]), 64)

    def test_sampling_and_greedy_share_declared_generation_filters(self):
        for sample in (False, True):
            settings = _generation_kwargs(sample=sample, max_new_tokens=192,
                                           pad_token_id=1, temperature=.8)
            self.assertEqual(settings["repetition_penalty"], 1.0)
            self.assertEqual(settings["top_k"], 0)
            self.assertEqual(settings["top_p"], 1.0)
            self.assertEqual(settings["temperature"], .8 if sample else 1.0)

    def test_only_loopback_servers_are_accepted(self):
        self.assertEqual(_loopback_url("http://127.0.0.1:11434/"), "http://127.0.0.1:11434")
        self.assertEqual(_loopback_url("http://[::1]:11434"), "http://[::1]:11434")
        for address in ["https://example.com", "http://10.0.0.1", "http://user:pass@localhost", "file:///tmp/model"]:
            with self.assertRaises(ValueError):
                _loopback_url(address)

    def test_unrecognized_backend_never_substitutes(self):
        with self.assertRaises(ValueError):
            build_backend({"kind": "hosted"})

    def test_ollama_full_menu_logprobs_are_explicitly_unsupported(self):
        backend = OllamaBackend.__new__(OllamaBackend)
        with self.assertRaisesRegex(NotImplementedError, "complete conditional-menu"):
            backend._decide(self.messages, ["A", "B"], "logprobs", 10, 0, .8)

    def test_ollama_rejects_tokenizer_mismatch_or_truncation(self):
        class Tokenizer:
            def apply_chat_template(self, *args, **kwargs):
                return "complete public prompt"
            def encode(self, *args, **kwargs):
                return [1, 2, 3]
        backend = OllamaBackend.__new__(OllamaBackend)
        backend.tokenizer = Tokenizer()
        backend.max_context, backend.max_new_tokens = 100, 10
        backend.model_name = "test:local"
        backend._request = lambda *args: {"model": "test:local", "done": True,
                                         "prompt_eval_count": 2, "eval_count": 1, "response": "A"}
        with self.assertRaisesRegex(ValueError, "tokenizer mismatch or server truncation"):
            backend._generate(self.messages)

    def test_ollama_context_overflow_is_rejected_before_request(self):
        class Tokenizer:
            def apply_chat_template(self, *args, **kwargs):
                return "complete public prompt"
            def encode(self, *args, **kwargs):
                return list(range(95))
        backend = OllamaBackend.__new__(OllamaBackend)
        backend.tokenizer = Tokenizer()
        backend.max_context, backend.max_new_tokens = 100, 10
        backend._request = lambda *args: self.fail("overflow must not reach model server")
        with self.assertRaisesRegex(ValueError, "evidence was not truncated"):
            backend._generate(self.messages)


if __name__ == "__main__":
    unittest.main()
