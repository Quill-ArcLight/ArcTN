"""Export a reviewable source-only snapshot, inventory, SBOM and checksums.

This creates a local review bundle, not publication approval. Private engines,
Git history, internal reports, untracked build outputs and wheel files are never
included. Audit the exported source directory before publishing it.
"""
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import tomllib
import zipfile

from audit import ROOT, REPORTS, write_json
from advisory_policy import evaluate

ROOT_FILES = {".gitignore", ".gitattributes", "Cargo.toml", "Cargo.lock", "build.rs", "README.md", "README.en.md",
              "LICENSE", "CONTRIBUTING.md", "SECURITY.md", "THIRD_PARTY_NOTICES.md", "deny.toml"}
PREFIXES = ("src/", "tests/", "examples/", "THIRD_PARTY_LICENSES/", "scripts/open-source/", ".github/workflows/",
            "pybind/src/", "pybind/python/", "pybind/tests/", "pybind/licenses/")
EXTRA_FILES = {"docs/mpi.md", "docs/engine-interface.md", "pybind/Cargo.toml", "pybind/Cargo.lock",
               "pybind/build.rs", "pybind/README.md", "pybind/pyproject.toml"}
BINARY_EXTENSIONS = {".dll", ".so", ".dylib", ".exe", ".pyd", ".pdb", ".a", ".lib", ".rlib", ".whl", ".pyc", ".zip"}


def allowed(relative, data):
    p = PurePosixPath(relative)
    if p.is_absolute() or ".." in p.parts or "\\" in relative:
        return False
    if any(part in {".git", "target", "__pycache__", "_lib", "internal"} for part in p.parts):
        return False
    if p.suffix.lower() in BINARY_EXTENSIONS or data[:4] in {b"\x7fELF", b"\xcf\xfa\xed\xfe", b"\xfe\xed\xfa\xcf", b"\xca\xfe\xba\xbe"} or data.startswith(b"MZ"):
        return False
    return relative in ROOT_FILES or relative in EXTRA_FILES or relative.startswith(PREFIXES)


def export(destination):
    destination = Path(destination).resolve()
    if not destination.is_relative_to((ROOT / "target/source-release").resolve()):
        raise SystemExit("Bundle destination must stay under target/source-release")
    if destination.exists():
        raise SystemExit("Use a new output directory; an existing bundle is never overwritten")
    policy = json.loads((ROOT / "scripts/open-source/advisory-policy.json").read_text(encoding="utf-8"))
    for name, lockfile in [("root", "Cargo.lock"), ("pybind", "pybind/Cargo.lock")]:
        report = json.loads((REPORTS / f"audit-final-{name}.stdout.log").read_text(encoding="utf-8"))
        result = evaluate(report, policy, tomllib.loads((ROOT / lockfile).read_text(encoding="utf-8")), lockfile)
        if not result["passed"]:
            raise SystemExit(f"Advisory review failed for {lockfile}: {result['errors']}")
    files = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT).decode().split("\0")
    records, contents = [], {}
    for rel in sorted(set(files) - {""}):
        p = ROOT / rel
        if not p.exists():
            continue
        if p.is_symlink() or not p.is_file():
            raise SystemExit(f"Unexpected source type: {rel}")
        data = p.read_bytes()
        if not allowed(rel, data):
            raise SystemExit(f"Candidate needs explicit review: {rel}")
        contents[rel] = data
        records.append(dict(path=rel, size=len(data), sha256=hashlib.sha256(data).hexdigest()))
    source = destination / "source"
    distribution = destination / "distribution"
    distribution.mkdir(parents=True)
    for rel, data in contents.items():
        p = source / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    with zipfile.ZipFile(distribution / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for rel, data in contents.items():
            info = zipfile.ZipInfo(rel, (2026, 9, 14, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
    with (distribution / "source-files.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "size", "sha256"])
        writer.writeheader()
        writer.writerows(records)
    bom = json.loads((REPORTS / "workspace-source.cdx.json").read_text(encoding="utf-8"))
    coverage = json.loads((REPORTS / "cargo-coverage.json").read_text(encoding="utf-8"))
    for path, expected in coverage["lock_hashes"].items():
        if hashlib.sha256(contents[path]).hexdigest() != expected:
            raise SystemExit(f"SBOM lockfile mismatch: {path}")
    if coverage["missing_license_files"] or not coverage["lockfiles_fully_covered"]:
        raise SystemExit("Incomplete dependency evidence")
    bom["metadata"]["properties"] += [dict(name="arctn:source-zip-sha256", value=hashlib.sha256((distribution / "source.zip").read_bytes()).hexdigest()),
        dict(name="arctn:base-commit", value=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip())]
    write_json(distribution / "workspace-source.cdx.json", bom)
    python_bom_file = REPORTS / "python-validation.cdx.json"
    if python_bom_file.exists():
        python_bom = json.loads(python_bom_file.read_text(encoding="utf-8"))
        # Environment SBOMs can contain the local wheel installation URL.
        for component in python_bom.get("components", []):
            component["externalReferences"] = [r for r in component.get("externalReferences", []) if not r.get("url", "").startswith("file:")]
        python_bom.setdefault("metadata", {}).setdefault("properties", []).append(dict(name="arctn:scope", value="Windows x64 CPython 3.13 validation environment: source-only arctn plus runtime, optional integrations, build and test dependencies. Not the proprietary engine or a binary-complete SBOM."))
        write_json(distribution / "python-validation.cdx.json", python_bom)
        lock = (REPORTS / "python-requirements.lock").read_text(encoding="utf-8")
        # Keep exact versions/hashes, omit the autogenerated local command.
        lock = "# Validation environment: Windows x64, CPython 3.13.\n" + "\n".join(line for line in lock.splitlines() if not line.lstrip().startswith("#")) + "\n"
        (distribution / "python-validation-requirements.lock").write_text(lock, encoding="utf-8")
    files = sorted(distribution.iterdir())
    (distribution / "SHA256SUMS").write_text("".join(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.name}\n" for f in files), encoding="utf-8")
    write_json(destination / "review-status.json", dict(status="candidate-requires-review", files=len(records),
        excluded=["Git history", "internal reports", "private engine", "binaries"],
        note="Check current audit results, provenance confirmation and license authority before publication. This file is internal and not in distribution."))
    print(distribution)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination")
    export(parser.parse_args().destination)
