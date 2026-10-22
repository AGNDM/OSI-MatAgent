"""Feature provenance, actual fitting scopes, and public-only T1 references.

All computation belongs inside Slurm. Neural encoders are explicit test doubles;
the scaler/logistic/kNN fits and frozen dataset splits used below are real.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from comparison.features import (
    ARTIFACT_SCHEMA, CachedEncoder, FrozenFeatureBackend, ReferenceBackend,
    _encoder_signature, _validate_cohort, fit_artifacts, public_task,
)
from comparison.feature_io import task_text
from comparison.policies import load_bundle
from comparison.run import MCQA_SYSTEM


class ExplicitTestEncoder:
    """Deterministic synthetic vectors for tests; never a production fallback."""
    def __init__(self):
        self.provenance = {"feature_source": "explicit_unit_test_double", "repository": "test/encoder",
            "revision": "test-fixed", "model_identity": {"kind": "embedding", "model_files_sha256": {"weights": "1" * 64}},
            "chunk_policy": "all_tokens_test", "max_chunk_tokens_with_specials": 256,
            "model_dtype": "float32", "device": "cpu"}
        self.calls, self.last_usage = [], {}
    def encode(self, texts):
        self.calls.append(list(texts))
        self.last_usage = {"input_tokens": sum(len(text) for text in texts), "truncated_tokens": 0}
        return np.array([[value + 1 for value in hashlib.sha256(text.encode()).digest()[:8]] for text in texts], dtype=float)


class FeatureProjectionTests(unittest.TestCase):
    def test_complete_messages_survive_feature_projection_without_metadata(self):
        messages = [{"role": "system", "content": "system"}, {"role": "user", "content": "full-history-marker " * 400}]
        record = public_task(messages, ["A", "B"], {"A": "exact native option A", "B": "exact native option B"})
        record.update(task_id="private-id", group_id="private-group", split="train")
        text = task_text(record, include_options=False)
        self.assertEqual(json.loads(record["question"]), messages)
        self.assertEqual(text.count("full-history-marker"), 400)
        self.assertNotIn("private-group", text)
        self.assertNotIn("private-id", text)

    def test_cache_counts_cold_work_and_reuses_identical_content(self):
        source = ExplicitTestEncoder()
        encoder = CachedEncoder(source)
        first = encoder.encode(["one", "one", "two"])
        self.assertEqual(source.calls, [["one", "two"]])
        self.assertEqual(encoder.last_usage["cold_texts"], 2)
        self.assertEqual(encoder.last_usage["cache_hits"], 1)
        second = encoder.encode(["two", "one"])
        np.testing.assert_array_equal(second, first[[2, 0]])
        self.assertEqual(encoder.last_usage["input_tokens"], 0)
        self.assertEqual(encoder.last_usage["cold_texts"], 0)

    def test_encoder_signature_includes_weights_dtype_and_chunk_policy(self):
        encoder = CachedEncoder(ExplicitTestEncoder())
        original = _encoder_signature(encoder)
        for key, value in (("model_dtype", "float16"), ("max_chunk_tokens_with_specials", 128), ("chunk_policy", "truncate")):
            changed = copy.deepcopy(encoder)
            changed.provenance[key] = value
            self.assertNotEqual(_encoder_signature(changed), original)
        changed = copy.deepcopy(encoder)
        changed.provenance["model_identity"]["model_files_sha256"]["weights"] = "2" * 64
        self.assertNotEqual(_encoder_signature(changed), original)

    def test_cohort_identity_or_group_overlap_is_rejected(self):
        cohort = {name: {"task_ids": [name], "groups": [name]} for name in ("train", "validation", "test")}
        _validate_cohort(cohort)
        cohort["test"]["groups"] = ["train"]
        with self.assertRaises(ValueError):
            _validate_cohort(cohort)

    def test_singleton_menu_has_deterministic_probability_without_encoding(self):
        backend = FrozenFeatureBackend.__new__(FrozenFeatureBackend)
        class NeverPredict:
            def predict(self, task):
                raise AssertionError("Singleton menus must not encode or invoke a classifier")
        backend.decision = NeverPredict()
        result = backend.decide([{"role": "user", "content": "public context"}], ["A"], option_descriptions={"A": "only action"})
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["probabilities"], {"A": 1.0})
        self.assertEqual(result["input_tokens"], 0)
        self.assertTrue(result["metadata"]["single_option_deterministic"])


class FrozenFitArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = Path(os.environ.get("COMPARISON_TEST_BUNDLE", Path(__file__).resolve().parents[2] / "instrument")).resolve()
        if not cls.bundle.is_dir():
            raise unittest.SkipTest("Set COMPARISON_TEST_BUNDLE to the frozen instrument release")
        cls.helpers = load_bundle(cls.bundle)
        cls.temp = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.temp.name) / "fitted"
        cls.encoder = CachedEncoder(ExplicitTestEncoder())
        with patch("comparison.features.build_encoder", return_value=cls.encoder):
            fit_artifacts(cls.bundle, cls.directory, {"encoder_kind": "test-double"}, methods=("cosine", "linear", "knn"))
        cls.tasks, cls.answers = cls.helpers["common"].load_dataset(cls.bundle / "mcqa_388/questions.jsonl.gz", cls.bundle / "mcqa_388/answers.jsonl")

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "temp"):
            cls.temp.cleanup()

    def test_train232_validation42_and_no_heldout_labels_in_fitted_artifact(self):
        manifest = json.loads((self.directory / "linear/manifest.json").read_text())
        self.assertEqual(manifest["schema_version"], ARTIFACT_SCHEMA)
        self.assertEqual(len(manifest["cohort"]["train"]["task_ids"]), 232)
        self.assertEqual(len(manifest["cohort"]["validation"]["task_ids"]), 42)
        self.assertFalse(set(manifest["cohort"]["train"]["task_ids"]) & set(manifest["cohort"]["test"]["task_ids"]))
        self.assertEqual(manifest["fit"]["fitting_scope"], "train_only")
        self.assertEqual(manifest["fit"]["scaler_scope"], "train_only")
        self.assertFalse(manifest["fitted_neural_weights"])

    def test_all_three_actual_fitted_methods_reload_and_predict(self):
        task = next(task for task in self.tasks if task["split"].startswith("test"))
        messages = [{"role": "system", "content": MCQA_SYSTEM},
                    {"role": "user", "content": self.helpers["prompts"].task_text(task)}]
        labels = [option["option_id"] for option in task["options"]]
        descriptions = {option["option_id"]: json.dumps(option, sort_keys=True, ensure_ascii=False, allow_nan=False) for option in task["options"]}
        for method in ("cosine", "linear", "knn"):
            with self.subTest(method=method), patch("comparison.features.build_encoder", return_value=CachedEncoder(ExplicitTestEncoder())):
                backend = FrozenFeatureBackend({"artifact_path": str(self.directory / method)})
                result = backend.decide(messages, labels, option_descriptions=descriptions)
            self.assertEqual(result["status"], "ok", result.get("error"))
            self.assertEqual(set(result["probabilities"]), set(labels))
            self.assertAlmostEqual(sum(result["probabilities"].values()), 1)
            self.assertIsNone(result["gpu_seconds"])

    def test_pickle_checksum_is_checked_before_loading_or_model_construction(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "copy"
            shutil.copytree(self.directory / "linear", destination)
            with (destination / "decision.pkl").open("ab") as handle:
                handle.write(b"tampered")
            with patch("comparison.features.build_encoder") as build, patch("comparison.features.pickle.load") as load:
                with self.assertRaisesRegex(ValueError, "checksum"):
                    FrozenFeatureBackend({"artifact_path": str(destination)})
                build.assert_not_called()
                load.assert_not_called()

    def test_weight_signature_mismatch_rejects_pickle_before_loading(self):
        encoder = CachedEncoder(ExplicitTestEncoder())
        encoder.provenance["model_identity"]["model_files_sha256"]["weights"] = "3" * 64
        with patch("comparison.features.build_encoder", return_value=encoder), patch("comparison.features.pickle.load") as load:
            with self.assertRaisesRegex(ValueError, "weights/tokenizer/protocol"):
                FrozenFeatureBackend({"artifact_path": str(self.directory / "linear")})
            load.assert_not_called()

    def test_majority_prior_has_mandatory_checksum_and_train_only_identity(self):
        backend = ReferenceBackend({"reference": "majority", "artifact_path": str(self.directory)})
        self.assertEqual(len(backend.fit["training_task_ids"]), 232)
        result = backend.decide([{"role": "user", "content": "public"}], ["A", "B", "C"])
        self.assertEqual(result["status"], "ok")
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            shutil.copy(self.directory / "majority.json", destination)
            with self.assertRaises(FileNotFoundError):
                ReferenceBackend({"reference": "majority", "artifact_path": str(destination)})
            shutil.copy(self.directory / "majority-manifest.json", destination)
            (destination / "majority.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "checksum"):
                ReferenceBackend({"reference": "majority", "artifact_path": str(destination)})


class T1ReferenceTests(unittest.TestCase):
    def task(self, *, power=False):
        return {"input_protocol": "public-menu-union-v2", "instrument_id": "nabertherm_l9", "question": "Can this call start?",
            "observation": {"observations": {"power.enabled": {"value": power}, "program.running": {"value": False},
                "door.closed": {"value": True}, "program.segments": {"value": ["public segment"]}}},
            "contract": {"capabilities": {"thermal.start_program": {"input_schema": {"type": "object"}}}},
            "goal": {"proposed_steps": [{"kind": "invoke", "capability_id": "thermal.start_program", "arguments": {}}]},
            "options": [{"option_id": "A", "steps": [{"kind": "answer", "answer": "succeeded"}]},
                        {"option_id": "B", "steps": [{"kind": "answer", "answer": "rejected"}]}]}

    def test_rules_use_public_observations_and_expose_citations_without_oracle(self):
        backend = ReferenceBackend({"reference": "t1"})
        for power, expected in ((False, "B"), (True, "A")):
            with self.subTest(power=power):
                result = backend.decide([{"role": "user", "content": json.dumps(self.task(power=power))}], ["A", "B"])
                self.assertEqual(result["status"], "ok")
                self.assertEqual(result["selected_option_id"], expected)
                self.assertIn("power.enabled", result["causes_cited"])
                self.assertFalse(result["metadata"]["oracle_access"])

    def test_missing_public_state_stays_undefined_not_guessed(self):
        public = self.task()
        public["observation"]["observations"].pop("power.enabled")
        result = ReferenceBackend({"reference": "t1"}).decide([{"role": "user", "content": json.dumps(public)}], ["A", "B"])
        self.assertEqual(result["status"], "undefined")
        self.assertIsNone(result["probabilities"])

    def test_native_sequential_payload_is_explicitly_undefined(self):
        result = ReferenceBackend({"reference": "t1"}).decide([{"role": "user", "content": json.dumps({"environment": {"goal": "public"}})}], ["A"])
        self.assertEqual(result["status"], "undefined")
        self.assertEqual(result["metadata"]["reporting_scope"], "T1_reference_only")


if __name__ == "__main__":
    unittest.main()
