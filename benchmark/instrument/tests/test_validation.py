"""Static release integrity and split-boundary checks, using temporary bundles."""
from __future__ import annotations

import copy
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from eval.validate import validate_bundle


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.bundle = Path(self.temporary.name) / "bundle"
        (self.bundle / "mcqa_388/splits").mkdir(parents=True)
        self.tasks = [self.task("q1", "train", "g1"), self.task("q2", "train", "g1"),
                      self.task("q3", "test", "g2")]
        self.answers = [{"task_id": task["task_id"], "correct_option_ids": ["A"]}
                        for task in self.tasks]
        self.write_dataset()
        (self.bundle / "README.md").write_text("Portable release\n", encoding="utf-8")
        self.freeze()

    @staticmethod
    def task(task_id, split, group):
        return {"task_id": task_id, "split": split, "group_id": group,
                "instrument_id": "example", "question": "Which action?",
                "options": [{"option_id": "A", "text": "Allowed", "steps": []},
                            {"option_id": "B", "text": "Rejected", "steps": []}],
                "observation": {}, "history": [], "manual": [],
                "contract": {"capabilities": {}, "state_slots": {}}}

    def write_dataset(self):
        with gzip.open(self.bundle / "mcqa_388/questions.jsonl.gz", "wt", encoding="utf-8") as handle:
            for task in self.tasks:
                handle.write(json.dumps(task) + "\n")
        (self.bundle / "mcqa_388/answers.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in self.answers), encoding="utf-8")
        for path in (self.bundle / "mcqa_388/splits").glob("*.txt"):
            path.unlink()
        for split in {task["split"] for task in self.tasks}:
            ids = [task["task_id"] for task in self.tasks if task["split"] == split]
            (self.bundle / "mcqa_388/splits" / f"{split}.txt").write_text(
                "\n".join(ids) + "\n", encoding="utf-8")

    def freeze(self):
        files = {}
        for path in self.bundle.rglob("*"):
            if path.is_file() and path.name not in {"manifest.json", "validation.json"}:
                payload = path.read_bytes()
                files[path.relative_to(self.bundle).as_posix()] = {
                    "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
        self.manifest = {"bundle_version": "1.0.0", "files": files,
                         "datasets": {"mcqa_388": {"questions": 3, "base_cases": 2, "groups": 2},
                                      "sequential_120": {"status": "placeholder"}}}
        self.save_manifest()

    def save_manifest(self):
        (self.bundle / "manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")

    def test_counts_follow_manifest_and_static_validation_does_not_replay(self):
        receipt = validate_bundle(self.bundle)
        self.assertEqual(receipt["status"], "passed")
        self.assertEqual(receipt["counts"]["questions"], 3)
        self.assertEqual(receipt["counts"]["groups"], 2)
        self.assertEqual(receipt["counts"]["base_cases_declared"], 2)
        self.assertEqual(receipt["split_counts"], {"test": 1, "train": 2})
        self.assertFalse(receipt["native_replay_performed"])

    def test_file_tampering_is_detected_even_at_the_same_size(self):
        (self.bundle / "README.md").write_text("Portable RELEASE\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            validate_bundle(self.bundle)

    def test_byte_count_tampering_is_detected(self):
        self.manifest["files"]["README.md"]["bytes"] += 1
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "Byte count mismatch"):
            validate_bundle(self.bundle)

    def test_unlisted_and_missing_files_fail(self):
        extra = self.bundle / "unexpected-answer-key.json"
        extra.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Unlisted publishable"):
            validate_bundle(self.bundle)
        extra.unlink()
        (self.bundle / "README.md").unlink()
        with self.assertRaisesRegex(ValueError, "Missing manifest file"):
            validate_bundle(self.bundle)

    def test_manifest_paths_must_stay_portable_and_inside_the_bundle(self):
        original = copy.deepcopy(self.manifest)
        for name in ("../outside", "/tmp/outside", "C:/outside", "mcqa_388/../outside",
                     "mcqa_388//answers.jsonl", "mcqa_388\\answers.jsonl"):
            with self.subTest(name=name):
                self.manifest = copy.deepcopy(original)
                self.manifest["files"][name] = {"sha256": "0" * 64, "bytes": 0}
                self.save_manifest()
                with self.assertRaisesRegex(ValueError, "Unsafe bundle path"):
                    validate_bundle(self.bundle)

    def test_symlinks_cannot_introduce_external_dependencies(self):
        external = Path(self.temporary.name) / "outside.txt"
        external.write_text("Portable release\n", encoding="utf-8")
        (self.bundle / "README.md").unlink()
        (self.bundle / "README.md").symlink_to(external)
        with self.assertRaisesRegex(ValueError, "escapes its root|symlink"):
            validate_bundle(self.bundle)

    def test_runtime_outputs_and_validation_receipt_are_ignored(self):
        for name in ("output/results.json", "eval/__pycache__/validate.cpython-312.pyc",
                     ".git/config", "stray.pyc", "stray.pyo", "validation.json",
                     ".venv/lib/package.py", ".venv-models/bin/python",
                     ".pytest_cache/README.md", ".cache/pip/wheel.whl"):
            path = self.bundle / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("runtime", encoding="utf-8")
        self.assertEqual(validate_bundle(self.bundle)["status"], "passed")

    def test_declared_question_and_group_counts_are_checked(self):
        for field in ("questions", "groups"):
            with self.subTest(field=field):
                original = self.manifest["datasets"]["mcqa_388"][field]
                self.manifest["datasets"]["mcqa_388"][field] = original + 1
                self.save_manifest()
                with self.assertRaisesRegex(ValueError, "count mismatch"):
                    validate_bundle(self.bundle)
                self.manifest["datasets"]["mcqa_388"][field] = original

    def test_multiple_correct_answers_are_rejected(self):
        self.answers[0]["correct_option_ids"] = ["A", "B"]
        self.write_dataset()
        self.freeze()
        with self.assertRaisesRegex(ValueError, "one correct option"):
            validate_bundle(self.bundle)

    def test_group_cannot_cross_split_boundaries(self):
        self.tasks[2]["group_id"] = "g1"
        self.write_dataset()
        self.freeze()
        self.manifest["datasets"]["mcqa_388"]["groups"] = 1
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "Group crosses split"):
            validate_bundle(self.bundle)

    def test_missing_or_incorrect_split_files_are_rejected(self):
        (self.bundle / "mcqa_388/splits/test.txt").unlink()
        self.freeze()
        with self.assertRaisesRegex(ValueError, "Split files do not match"):
            validate_bundle(self.bundle)
        self.write_dataset()
        (self.bundle / "mcqa_388/splits/train.txt").write_text("q1\nq3\n", encoding="utf-8")
        self.freeze()
        with self.assertRaisesRegex(ValueError, "Split files overlap|Split IDs do not match"):
            validate_bundle(self.bundle)

    def test_duplicate_split_members_are_rejected(self):
        (self.bundle / "mcqa_388/splits/train.txt").write_text("q1\nq2\nq1\n", encoding="utf-8")
        self.freeze()
        with self.assertRaisesRegex(ValueError, "Duplicate IDs in split"):
            validate_bundle(self.bundle)


if __name__ == "__main__":
    unittest.main()
