"""Publication boundary regression cases; no real files are published."""
import importlib.util
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
spec = importlib.util.spec_from_file_location("export_source", Path(__file__).resolve().parents[1] / "export-source.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ExportBoundary(unittest.TestCase):
    def test_private_and_generated_files_are_rejected(self):
        for path in [".git/config", "docs/open-source/todolist.md", "pybind/python/arctn/_lib/arctn_engine.dll",
                     "src/engine.dll", "src/internal/engine.rs", "target/release/tnpath.exe", "private/engine.rs"]:
            with self.subTest(path=path):
                self.assertFalse(module.allowed(path, b"example"))

    def test_binary_disguised_as_source_is_rejected(self):
        for data in [b"MZbinary", b"\x7fELFbinary", b"\xcf\xfa\xed\xfebinary"]:
            self.assertFalse(module.allowed("src/example.rs", data))

    def test_traversal_is_rejected(self):
        for path in ["src/../../private.rs", "/src/lib.rs", "src\\lib.rs"]:
            self.assertFalse(module.allowed(path, b"text"))

    def test_required_source_is_allowed(self):
        for path in ["src/lib.rs", "Cargo.lock", "LICENSE", "pybind/licenses/LICENSE", "tests/fixtures/demo_tiny6.net.json"]:
            self.assertTrue(module.allowed(path, b"text"))


if __name__ == "__main__":
    unittest.main()
