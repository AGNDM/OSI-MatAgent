"""Maintainer refresh cannot silently change frozen benchmark assets."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import release


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.bundle = Path(self.temporary.name)
        self.put("README.md", "Original guide\n")
        self.put("mcqa_388/questions.jsonl.gz", "frozen questions")
        self.put("mcqa_388/answers.jsonl", "frozen answers")
        self.put("mcqa_388/splits/train.txt", "q1\n")
        self.put("eval/sequential.py", "frozen grader")
        self.put("sequential_120/runtime/vendor/native.py", "frozen native runtime")
        self.put("sequential_120/runtime/requirements.txt", "jsonschema==4.26.0\n")
        self.put("sequential_120/cases.jsonl", "frozen cases")
        self.put("sequential_120/README.md", "Original sequential guide\n")
        self.put("sequential_120/manifest.json", json.dumps({"cases": 120,
            "files": release._inventory(self.bundle / "sequential_120")}))
        self.put("manifest.json", json.dumps({"bundle_version": "1.1.0",
            "files": release._inventory(self.bundle)}))
        self.baseline = (self.bundle / "manifest.json").read_bytes()

    def put(self, name, text):
        path = self.bundle / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def refresh(self):
        with patch.object(release, "validate_bundle", return_value={"status": "passed"}):
            return release.refresh_bundle(self.bundle, version="1.1.1")

    def test_refresh_updates_packaging_and_nested_inventory(self):
        self.put("README.md", "New colleague guide\n")
        self.put("sequential_120/README.md", "New sequential guide\n")
        self.put("tools/new.py", "# new release tooling\n")
        self.put("sequential_120/runtime/requirements.txt", "jsonschema==4.26.0\nattrs==25.4.0\n")
        receipt = self.refresh()
        manifest = release._json(self.bundle / "manifest.json")
        self.assertEqual(manifest["bundle_version"], "1.1.1")
        self.assertIn("tools/new.py", manifest["files"])
        self.assertEqual(release._json(self.bundle / "sequential_120/manifest.json")["cases"], 120)
        self.assertEqual(receipt["frozen_assets_verified"], 6)
        self.assertFalse(receipt["native_replay_performed"])

    def test_changed_data_answers_splits_graders_and_native_are_rejected(self):
        names = ["mcqa_388/questions.jsonl.gz", "mcqa_388/answers.jsonl",
                 "mcqa_388/splits/train.txt", "eval/sequential.py",
                 "sequential_120/cases.jsonl", "sequential_120/runtime/vendor/native.py"]
        for name in names:
            with self.subTest(name=name):
                original = (self.bundle / name).read_bytes()
                self.put(name, "changed")
                with self.assertRaisesRegex(ValueError, "Frozen asset changed"):
                    self.refresh()
                (self.bundle / name).write_bytes(original)
                self.assertEqual((self.bundle / "manifest.json").read_bytes(), self.baseline)

    def test_added_or_missing_frozen_assets_are_rejected(self):
        self.put("sequential_120/runtime/vendor/extra.py", "new runtime")
        with self.assertRaisesRegex(ValueError, "Frozen asset inventory changed"):
            self.refresh()
        (self.bundle / "sequential_120/runtime/vendor/extra.py").unlink()
        (self.bundle / "mcqa_388/splits/train.txt").unlink()
        with self.assertRaisesRegex(ValueError, "Frozen asset inventory changed"):
            self.refresh()

    def test_failed_validation_rolls_back_both_manifests(self):
        nested = self.bundle / "sequential_120/manifest.json"
        original_nested = nested.read_bytes()
        self.put("README.md", "Updated guide\n")
        with patch.object(release, "validate_bundle", side_effect=ValueError("invalid metadata")):
            with self.assertRaisesRegex(ValueError, "invalid metadata"):
                release.refresh_bundle(self.bundle, version="1.1.1")
        self.assertEqual((self.bundle / "manifest.json").read_bytes(), self.baseline)
        self.assertEqual(nested.read_bytes(), original_nested)

    def test_edited_nested_metadata_is_not_silently_blessed(self):
        nested = self.bundle / "sequential_120/manifest.json"
        metadata = release._json(nested)
        metadata["cases"] = 121
        nested.write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Nested manifest was edited"):
            self.refresh()

    def test_only_explicit_historical_provenance_relocation_is_allowed(self):
        old, new = next(iter(release._RELOCATIONS.items()))
        self.put(old, '{"historical": true}\n')
        manifest = release._json(self.bundle / "manifest.json")
        manifest["files"][old] = release._record(self.bundle / old)
        self.put("manifest.json", json.dumps(manifest))
        (self.bundle / new).parent.mkdir(parents=True, exist_ok=True)
        (self.bundle / old).rename(self.bundle / new)
        self.assertEqual(self.refresh()["status"], "passed")
        self.assertNotIn(old, release._json(self.bundle / "manifest.json")["files"])

    def test_inventory_rejects_symlinks(self):
        (self.bundle / "README.md").unlink()
        (self.bundle / "README.md").symlink_to(self.bundle / "mcqa_388/answers.jsonl")
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.refresh()

    def test_invalid_version_and_non_slurm_cli_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "MAJOR.MINOR.PATCH"):
            release.refresh_bundle(self.bundle, version="latest")
        with patch.dict(release.os.environ, {}, clear=True):
            self.assertEqual(release.main(["check", "--bundle", str(self.bundle)]), 1)


if __name__ == "__main__":
    unittest.main()
