"""Corruption and public/evaluator boundary checks for the sequential release."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from eval.validate_sequential import (_fingerprint, _schema_accepts, _validate_bank,
                                      _validate_cases, validate_sequential_dataset)


BUNDLE = Path(__file__).resolve().parents[1]


class SequentialValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dataset = BUNDLE / "sequential_120"
        cls.public = [json.loads(line) for line in (dataset / "cases.jsonl").read_text().splitlines()]
        cls.cases = [json.loads(line) for line in (dataset / "evaluator/case-specs.jsonl").read_text().splitlines()]
        cls.bank = json.loads((dataset / "action-presets.json").read_text())
        cls.archive = json.loads((dataset / "public-contracts.json").read_text())
        cls.contracts = _validate_bank(cls.bank, cls.archive)
        cls.metadata = {"data_included": True}

    def check_cases(self, public=None, cases=None, bank=None):
        return _validate_cases(self.public if public is None else public,
                               self.cases if cases is None else cases,
                               self.bank if bank is None else bank, self.contracts)

    def copy_static_dataset(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        bundle = Path(temporary.name)
        source = BUNDLE / "sequential_120"
        destination = bundle / "sequential_120"
        # Preserve the vendored source so data corruption cannot accidentally
        # pass by depending on the original development checkout.
        shutil.copytree(source / "runtime", destination / "runtime")
        for name in ("cases.jsonl", "action-presets.json", "public-contracts.json", "manifest.json",
                     "evaluator/case-specs.jsonl", "provenance/source-configuration.json",
                     "provenance/source-manifest.json", "provenance/native-evidence.json",
                     "provenance/native-validation.json", "provenance/context-preflight.json",
                     "provenance/cross-machine-preflight.json", "provenance/public-contracts-receipt.json"):
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / name, target)
        return bundle

    def test_full_frozen_release_and_receipts_validate_without_replay(self):
        receipt = validate_sequential_dataset(BUNDLE, self.metadata)
        self.assertEqual(receipt["counts"], {"cases": 120, "instruments": 8, "workflow_templates": 14,
                                           "public_commands": 146, "schema_validated_argument_presets": 1954})
        self.assertTrue(receipt["public_evaluator_boundary_checked"])
        self.assertFalse(receipt["native_replay_performed"])
        self.assertFalse(receipt["model_performance_measured"])
        self.assertFalse(receipt["train_test_split_assigned"])

    def test_public_and_private_ids_must_be_unique_and_match(self):
        public = copy.deepcopy(self.public)
        public[0]["task_id"] = public[1]["task_id"]
        with self.assertRaisesRegex(ValueError, "Duplicate sequential task ID"):
            self.check_cases(public=public)
        public[0]["task_id"] = "unknown-evaluator-case"
        with self.assertRaisesRegex(ValueError, "IDs do not match"):
            self.check_cases(public=public)

    def test_public_projection_rejects_nested_evaluator_metadata(self):
        public = copy.deepcopy(self.public)
        public[0]["initial_context"]["budget"]["seed"] = 42
        with self.assertRaisesRegex(ValueError, "Private evaluator field"):
            self.check_cases(public=public)
        public = copy.deepcopy(self.public)
        public[0]["initial_context"]["oracle"] = "private"
        with self.assertRaisesRegex(ValueError, "initial_context fields"):
            self.check_cases(public=public)

    def test_public_goal_and_budget_must_match_the_evaluator(self):
        for field, value, message in (("goal", "Different goal", "goal mismatch"),
                                      ("budget", {"max_actions": 1000}, "budget mismatch")):
            public = copy.deepcopy(self.public)
            public[0]["initial_context"][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, message):
                self.check_cases(public=public)

    def test_case_count_is_not_inferred_from_an_incomplete_export(self):
        with self.assertRaisesRegex(ValueError, "expected 120"):
            self.check_cases(public=self.public[:-1], cases=self.cases[:-1])

    def test_dependency_cycles_and_malformed_branch_rules_fail(self):
        cases = copy.deepcopy(self.cases)
        cases[0]["checkpoints"][0]["depends_on"] = [cases[0]["checkpoints"][0]["checkpoint_id"]]
        with self.assertRaisesRegex(ValueError, "dependency graph"):
            self.check_cases(cases=cases)
        cases = copy.deepcopy(self.cases)
        cases[0]["checkpoints"][0]["when_conditions"] = None
        with self.assertRaisesRegex(ValueError, "condition list"):
            self.check_cases(cases=cases)

    def test_omitted_optional_conditions_follow_native_empty_list_defaults(self):
        cases = copy.deepcopy(self.cases)
        for field in ("fixture_conditions", "escalation_conditions"):
            cases[0].pop(field, None)
        cases[0]["checkpoints"][0].pop("when_conditions", None)
        cases[0]["checkpoints"][0]["accept"].pop("conditions", None)
        self.check_cases(cases=cases)
        cases[0]["fixture_conditions"] = None
        with self.assertRaisesRegex(ValueError, "condition list"):
            self.check_cases(cases=cases)

    def test_parameter_preset_corruption_fails_before_source_hash_check(self):
        bank = copy.deepcopy(self.bank)
        bank["nabertherm_l9"]["power.set"][0] = {"enabled": "true"}
        with self.assertRaisesRegex(ValueError, "violates public input schema"):
            _validate_bank(bank, self.archive)
        bank = copy.deepcopy(self.bank)
        bank["nabertherm_l9"]["power.set"].append(copy.deepcopy(bank["nabertherm_l9"]["power.set"][0]))
        with self.assertRaisesRegex(ValueError, "Duplicate parameter preset"):
            _validate_bank(bank, self.archive)

    def test_a_missing_public_command_is_not_silently_removed(self):
        bank = copy.deepcopy(self.bank)
        del bank["nabertherm_l9"]["power.set"]
        with self.assertRaisesRegex(ValueError, "command mismatch"):
            _validate_bank(bank, self.archive)

    def test_public_contract_hash_and_hidden_commands_are_checked(self):
        archive = copy.deepcopy(self.archive)
        digest = archive["instruments"]["nabertherm_l9"]
        archive["contracts_by_sha256"][digest]["capabilities"]["fault.inject"] = {}
        with self.assertRaisesRegex(ValueError, "contract hash mismatch"):
            _validate_bank(self.bank, archive)
        contract = archive["contracts_by_sha256"].pop(digest)
        new_digest = _fingerprint(contract)
        archive["instruments"]["nabertherm_l9"] = new_digest
        archive["contracts_by_sha256"][new_digest] = contract
        bank = copy.deepcopy(self.bank)
        bank["nabertherm_l9"]["fault.inject"] = [{}]
        bank["__meta__"]["support_report"]["nabertherm_l9"]["fault.inject"] = {
            "preset_count": 1, "schema_validated": True, "exhaustive_parameter_coverage": False}
        with self.assertRaisesRegex(ValueError, "Private or unavailable public command"):
            _validate_bank(bank, archive)

    def test_finite_parameters_remain_nonexhaustive_and_no_split_is_invented(self):
        cases = copy.deepcopy(self.cases)
        cases[0]["split"] = "test"
        with self.assertRaisesRegex(ValueError, "has not been assigned"):
            self.check_cases(cases=cases)
        bank = copy.deepcopy(self.bank)
        bank["__meta__"]["support_report"]["nabertherm_l9"]["power.set"]["exhaustive_parameter_coverage"] = True
        with self.assertRaisesRegex(ValueError, "support report"):
            _validate_bank(bank, self.archive)

    def test_reference_parameters_must_be_available_to_the_model(self):
        cases = copy.deepcopy(self.cases)
        node = next(node for node in cases[0]["checkpoints"] if node["reference_action"]["kind"] == "invoke")
        node["reference_action"]["arguments"] = {"not_in_shared_bank": True}
        with self.assertRaisesRegex(ValueError, "absent from shared public presets"):
            self.check_cases(cases=cases)

    def test_manifest_cannot_claim_wrong_numbers(self):
        bundle = self.copy_static_dataset()
        path = bundle / "sequential_120/manifest.json"
        manifest = json.loads(path.read_text())
        manifest["cases"] = 121
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "declared cases count mismatch"):
            validate_sequential_dataset(bundle, self.metadata)

    def test_evaluator_changes_cannot_borrow_the_native_validation_proof(self):
        bundle = self.copy_static_dataset()
        path = bundle / "sequential_120/evaluator/case-specs.jsonl"
        cases = copy.deepcopy(self.cases)
        cases[0]["seed"] += 1
        path.write_text("".join(json.dumps(case) + "\n" for case in cases))
        with self.assertRaisesRegex(ValueError, "Frozen evaluator case specifications hash mismatch"):
            validate_sequential_dataset(bundle, self.metadata)

    def test_corrupt_native_receipt_and_falsely_named_model_results_fail(self):
        for filename, field, value, message in (
                ("native-validation.json", "completed_diagnostic_episodes", 479, "receipt hash mismatch"),
                ("context-preflight.json", "model_performance_measured", True, "Invalid context preflight")):
            bundle = self.copy_static_dataset()
            path = bundle / "sequential_120/provenance" / filename
            receipt = json.loads(path.read_text())
            receipt[field] = value
            path.write_text(json.dumps(receipt))
            with self.subTest(filename=filename), self.assertRaisesRegex(ValueError, message):
                validate_sequential_dataset(bundle, self.metadata)

    def test_stdlib_schema_checks_cover_types_limits_and_branches(self):
        schema = {"type": "object", "additionalProperties": False, "required": ["enabled", "rate"],
                  "properties": {"enabled": {"type": "boolean"}, "rate": {"type": "number", "minimum": 0, "maximum": 1}}}
        self.assertTrue(_schema_accepts({"enabled": True, "rate": 0.5}, schema))
        for value in ({"enabled": "true", "rate": 0.5}, {"enabled": True, "rate": 2},
                      {"enabled": True}, {"enabled": True, "rate": 1, "extra": 3}):
            with self.subTest(value=value):
                self.assertFalse(_schema_accepts(value, schema))
        self.assertTrue(_schema_accepts(None, {"anyOf": [{"type": "number"}, {"type": "null"}]}))
        self.assertFalse(_schema_accepts(float("nan"), {"type": "number"}))


if __name__ == "__main__":
    unittest.main()
