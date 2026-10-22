"""Study merge must reject overlapping, missing and foreign partitions."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from comparison.study import authenticate_identities, authenticate_validation_receipt, fingerprint, identity_signature, runtime_group, validate_partition


class StudyTests(unittest.TestCase):
    def test_complete_disjoint_partition(self):
        result = validate_partition(["a", "b", "c"], [["a", "c"], ["b"]])
        self.assertEqual(result["observed"], 3)
        self.assertEqual(result["missing_task_ids"], [])

    def test_overlap_rejected_even_for_partial_merge(self):
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            validate_partition(["a", "b"], [["a"], ["a"]], complete=False)

    def test_missing_coverage_explicit(self):
        result = validate_partition(["a", "b"], [["a"]], complete=False)
        self.assertEqual(result["missing_task_ids"], ["b"])
        with self.assertRaisesRegex(ValueError, "exactly cover"):
            validate_partition(["a", "b"], [["a"]])

    def test_foreign_rows_rejected(self):
        with self.assertRaisesRegex(ValueError, "Foreign"):
            validate_partition(["a"], [["z"]], complete=False)

    def test_runtime_environment_selection(self):
        self.assertEqual(runtime_group({"backend": {"kind": "native", "family": "apus"}}), "apus")
        self.assertEqual(runtime_group({"backend": {"kind": "native", "family": "openjev"}}), "hf")
        self.assertEqual(runtime_group({"backend": {"kind": "ollama"}}), "ollama")

    def test_identity_keeps_weights_and_ignores_ephemeral_transport(self):
        value = identity_signature({"base_url": "http://127.0.0.1:1111", "load_wall_time_s": 3,
                                    "expert": {"weight_sha256": {"weights": "123"}, "hardware": "GPU"}})
        self.assertEqual(value, {"expert": {"weight_sha256": {"weights": "123"}}})

    def test_different_fitted_feature_weights_cannot_merge(self):
        for field in ("artifact_sha256", "manifest_sha256", "model_files_sha256", "model_dtype"):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "Incompatible"):
                authenticate_identities([{ "encoder": {field: "first"}}, {"encoder": {field: "different"}}])

    def test_failed_lazy_native_load_can_lack_hashes_but_cannot_hide_changed_weights(self):
        authenticate_identities([{"family": "openjev"}, {"family": "openjev", "local_files_sha256": {"weights": "a"}}])
        with self.assertRaisesRegex(ValueError, "Incompatible"):
            authenticate_identities([{"family": "openjev"}, {"local_files_sha256": {"weights": "a"}},
                                     {"local_files_sha256": {"weights": "b"}}])

    def test_validation_receipt_binds_semantic_checks_and_exact_source_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inventory = {"comparison/run.py": "expected", "configuration_input": "fixed"}
            receipt = {"status": "validated", "semantic_smoke_assertions_passed": True,
                       "snapshot_source_sha256": inventory,
                       "snapshot_source_inventory_sha256": fingerprint(inventory)}
            (root / "receipt.json").write_text(json.dumps(receipt))
            validator = Mock()
            with patch("comparison.study.__file__", str(root / "snapshot/comparison/study.py")), \
                 patch("comparison.study.source_inventory", return_value=inventory) as actual, \
                 patch("comparison.study.load_bundle", return_value={"validate": SimpleNamespace(validate_bundle=validator)}):
                self.assertEqual(authenticate_validation_receipt(root)["status"], "validated")
                validator.assert_called_once()
                actual.return_value = {"comparison/run.py": "changed", "configuration_input": "fixed"}
                with self.assertRaisesRegex(ValueError, "changed after validation"):
                    authenticate_validation_receipt(root)


if __name__ == "__main__":
    unittest.main()
