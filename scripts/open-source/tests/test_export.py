"""Publication boundary regression cases; no real files are published."""
import importlib.util
from pathlib import Path
import sys
import tomllib
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
        for path in ["src/example.rs", "CHANGELOG.md"]:
            for data in [b"MZbinary", b"\x7fELFbinary", b"\xcf\xfa\xed\xfebinary"]:
                with self.subTest(path=path, data=data):
                    self.assertFalse(module.allowed(path, data))

    def test_traversal_is_rejected(self):
        for path in ["src/../../private.rs", "/src/lib.rs", "src\\lib.rs"]:
            self.assertFalse(module.allowed(path, b"text"))

    def test_required_source_is_allowed(self):
        for path in ["src/lib.rs", "Cargo.lock", "README.md", "README.en.md", "README.zh-CN.md", "CHANGELOG.md", "LICENSE", "pybind/licenses/LICENSE", "tests/fixtures/demo_tiny6.net.json"]:
            with self.subTest(path=path):
                self.assertTrue(module.allowed(path, b"text"))

    def test_changelog_does_not_allow_unreviewed_documents(self):
        self.assertTrue(module.allowed("CHANGELOG.md", (module.ROOT / "CHANGELOG.md").read_bytes()))
        for path in ["NOTES.md", "CHANGELOG-private.md", "private/CHANGELOG.md"]:
            with self.subTest(path=path):
                self.assertFalse(module.allowed(path, b"text"))


class PackageMetadata(unittest.TestCase):
    def test_repository_links_point_to_public_repository(self):
        repository = "https://github.com/Quill-ArcLight/ArcTN"
        for manifest, table, fields in [
            ("Cargo.toml", "package", {"homepage": repository, "repository": repository}),
            ("pybind/Cargo.toml", "package", {"homepage": repository}),
            ("pybind/pyproject.toml", "project", {
                "Source": repository, "Issues": repository + "/issues",
            }),
        ]:
            metadata = tomllib.loads((module.ROOT / manifest).read_text(encoding="utf-8"))[table]
            if table == "project":
                metadata = metadata["urls"]
            for key, expected in fields.items():
                with self.subTest(manifest=manifest, key=key):
                    self.assertEqual(metadata[key], expected)


if __name__ == "__main__":
    unittest.main()
