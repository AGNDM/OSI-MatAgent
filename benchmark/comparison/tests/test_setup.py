"""Setup lock parsing must preserve exact versions across inherited locks."""
from pathlib import Path
import tempfile
import unittest

from comparison.setup import read_pins


class SetupLockTests(unittest.TestCase):
    def test_recursive_includes_and_matching_duplicates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "base.txt").write_text("# locked\nexample-package==1.2\n")
            (root / "model.txt").write_text("-r base.txt\nexample_package==1.2\nother==3.4\n")
            seen = set()
            self.assertEqual(read_pins(root / "model.txt", seen),
                             {"example-package": "1.2", "other": "3.4"})
            self.assertEqual(seen, {(root / "base.txt").resolve(), (root / "model.txt").resolve()})

    def test_conflicting_lock_pins_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "base.txt").write_text("package==1\n")
            (root / "model.txt").write_text("-r base.txt\npackage==2\n")
            with self.assertRaisesRegex(ValueError, "Conflicting"):
                read_pins(root / "model.txt")

    def test_unpinned_dependency_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "requirements.txt"
            path.write_text("package>=1\n")
            with self.assertRaisesRegex(ValueError, "exact package pin"):
                read_pins(path)


if __name__ == "__main__":
    unittest.main()
