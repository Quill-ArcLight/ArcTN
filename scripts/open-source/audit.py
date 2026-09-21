"""Run traceable local release checks. Reports stay in ignored target/.

Usage: python scripts/open-source/audit.py prepare|rust|metadata|python|security
Independent stages can run concurrently after prepare. No network publication.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import tomllib
from datetime import datetime, timezone
from advisory_policy import evaluate

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "target/open-source-audit/current"
SOURCE = OUT / "source"
REPORTS = OUT / "reports"
TOOLS = ROOT / "target/open-source-tools"
RESULTS = []


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(name, command, cwd=SOURCE, env=None, timeout=1800):
    REPORTS.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    before = time.monotonic()
    child_env = dict(os.environ)
    child_env.pop("ARCTN_ENGINE_LIBRARY", None)
    child_env.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8", CARGO_TERM_COLOR="never",
                     UV_CACHE_DIR=str(TOOLS / "uv-cache"))
    if env:
        child_env.update(env)
    command = [str(x) for x in command]
    with (REPORTS / f"{name}.stdout.log").open("wb") as stdout, (REPORTS / f"{name}.stderr.log").open("wb") as stderr:
        try:
            result = subprocess.run(command, cwd=cwd, env=child_env, stdout=stdout, stderr=stderr, timeout=timeout)
            code = result.returncode
        except (OSError, subprocess.TimeoutExpired) as error:
            stderr.write(str(error).encode("utf-8"))
            code = -1
    record = dict(name=name, command=command, cwd=str(cwd), started=started,
                  seconds=round(time.monotonic() - before, 2), exit_code=code)
    write_json(REPORTS / f"{name}.result.json", record)
    RESULTS.append(record)
    print(f"{name}: exit={code} ({record['seconds']}s)", flush=True)
    return code


def prepare():
    if SOURCE.exists():
        raise SystemExit("Existing audit snapshot: archive target/open-source-audit/current before preparing another run.")
    REPORTS.mkdir(parents=True, exist_ok=True)
    files = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT).decode().split("\0")
    records = []
    for relative in sorted(set(files) - {""}):
        path = ROOT / relative
        if not path.exists():
            continue
        if path.is_symlink() or not path.is_file() or relative.startswith(("docs/open-source/", "target/", "pybind/target/", ".git/")):
            raise SystemExit(f"Unexpected candidate: {relative}")
        data = path.read_bytes()
        destination = SOURCE / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        records.append(dict(path=relative, size=len(data), sha256=hashlib.sha256(data).hexdigest()))
    write_json(REPORTS / "source-files.json", records)
    with (REPORTS / "source-files.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "size", "sha256"])
        writer.writeheader()
        writer.writerows(records)
    write_json(REPORTS / "scope.json", dict(timestamp=datetime.now(timezone.utc).isoformat(),
        base_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
        git_status=subprocess.check_output(["git", "status", "--porcelain=v1"], cwd=ROOT).decode(),
        platform=sys.platform, file_count=len(records), source_snapshot_sha256=hashlib.sha256(
            json.dumps(records, sort_keys=True).encode()).hexdigest(),
        scope="Current source including two Cargo locks, optional MPI dependencies and Python extras. Engine binaries require separate input and audit."))
    run("git-authors", ["git", "log", "--all", "--format=%an <%ae>"], cwd=ROOT)
    run("git-refs", ["git", "show-ref"], cwd=ROOT)
    run("rust-version", ["rustc", "-Vv"])
    print(f"Prepared {len(records)} files", flush=True)


def rust():
    env = {"CARGO_TARGET_DIR": str(OUT / "rust-build"), "RAYON_NUM_THREADS": "2", "MATMUL_NUM_THREADS": "2"}
    checks = [
        ("fmt", ["cargo", "fmt", "--all", "--check"]),
        ("pybind-fmt", ["cargo", "fmt", "--manifest-path", "pybind/Cargo.toml", "--check"]),
        ("rust-tests", ["cargo", "test", "--locked", "--all-targets"]),
        ("rust-mt-tests", ["cargo", "test", "--locked", "--all-targets", "--features", "mt"]),
        ("doctests", ["cargo", "test", "--locked", "--doc"]),
        ("clippy", ["cargo", "clippy", "--locked", "--all-targets", "--features", "mt", "--", "-D", "warnings"]),
        ("rustdoc", ["cargo", "doc", "--locked", "--no-deps"]),
        ("example", ["cargo", "run", "--locked", "--example", "contraction"]),
        ("cargo-package", ["cargo", "package", "--locked", "--allow-dirty"]),
    ]
    for name, command in checks:
        run(name, command, env={**env, **({"RUSTDOCFLAGS": "-D warnings"} if name == "rustdoc" else {})})


def metadata():
    for name, manifest in [("root", "Cargo.toml"), ("pybind", "pybind/Cargo.toml")]:
        run(f"metadata-final-{name}", ["cargo", "metadata", "--locked", "--all-features", "--format-version", "1", "--manifest-path", manifest])


def python_checks():
    venv = TOOLS / "python-validation"
    py = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    run("python-venv", ["uv", "venv", "--allow-existing", "--python", "3.13", venv])
    requirements = REPORTS / "python-requirements.in"
    requirements.write_text("numpy>=1.23\nopt_einsum>=3.3\ncotengra>=0.6\nquimb>=1.7\nmaturin>=1.9.3,<2\npytest\n", encoding="utf-8")
    lock = REPORTS / "python-requirements.lock"
    if run("python-resolve", ["uv", "pip", "compile", "--python", py, "--generate-hashes", requirements, "-o", lock]):
        return
    wheelhouse = TOOLS / "wheelhouse"
    offline = ["--no-index", "--find-links", wheelhouse] if wheelhouse.exists() else []
    if run("python-install", ["uv", "pip", "sync", "--python", py, *offline, "--require-hashes", lock]):
        return
    run("python-freeze", ["uv", "pip", "freeze", "--python", py])
    env = {"PYO3_PYTHON": str(py), "CARGO_TARGET_DIR": str(OUT / "python-build"), "RAYON_NUM_THREADS": "2"}
    run("pybind-clippy", ["cargo", "clippy", "--locked", "--manifest-path", "pybind/Cargo.toml", "--", "-D", "warnings"], env=env)
    wheels = OUT / "python-dist"
    if run("wheel-build", [py, "-m", "maturin", "build", "--release", "--locked", "--manifest-path", "pybind/Cargo.toml", "--out", wheels], env=env) == 0:
        wheel = sorted(wheels.glob("*.whl"))[-1]
        if run("wheel-install", ["uv", "pip", "install", "--python", py, "--no-deps", wheel]) == 0:
            run("python-tests", [py, "-m", "pytest", "-q", "pybind/tests", "--junitxml", REPORTS / "python-tests.xml"], env=env)
            run("python-inspect", [py, "-c", "import importlib.metadata,json; print(json.dumps([dict(name=d.metadata['Name'],version=d.version,license=d.metadata.get('License-Expression') or d.metadata.get('License'),requires=d.requires,files=[str(x) for x in d.files or [] if 'license' in str(x).lower() or 'copying' in str(x).lower()]) for d in importlib.metadata.distributions()]))"])
    run("sdist-build", [py, "-m", "maturin", "sdist", "--manifest-path", "pybind/Cargo.toml", "--out", wheels], env=env)


def security():
    gitleaks = next((TOOLS / "gitleaks").rglob("gitleaks.exe"))
    audit = next((TOOLS / "cargo-audit").rglob("cargo-audit.exe"))
    run("gitleaks-history", [gitleaks, "git", ROOT, "--log-opts=--all", "--redact=100", "--report-format", "json", "--report-path", REPORTS / "gitleaks-history.json"], cwd=ROOT)
    run("gitleaks-current", [gitleaks, "dir", SOURCE, "--redact=100", "--report-format", "json", "--report-path", REPORTS / "gitleaks-current.json"])
    for name, lock in [("root", "Cargo.lock"), ("pybind", "pybind/Cargo.lock")]:
        run(f"audit-final-{name}", [audit, "audit", "--file", lock, "--json", "--db", TOOLS / "rustsec-db"])
        report = REPORTS / f"audit-final-{name}.stdout.log"
        if report.exists():
            try:
                data = json.loads(report.read_text(encoding="utf-8"))
                policy = json.loads((SOURCE / "scripts/open-source/advisory-policy.json").read_text(encoding="utf-8"))
                locked = tomllib.loads((SOURCE / lock).read_text(encoding="utf-8"))
                result = evaluate(data, policy, locked, lock)
                RESULTS.append(dict(name=f"advisory-review-{name}", exit_code=0 if result["passed"] else 1))
                write_json(REPORTS / f"advisory-review-{name}.json", result)
            except (ValueError, KeyError, TypeError):
                RESULTS.append(dict(name=f"audit-json-{name}", exit_code=1))
    run("rustsec-db-revision", ["git", "-C", TOOLS / "rustsec-db", "rev-parse", "HEAD"])


def licenses():
    deny = next((TOOLS / "cargo-deny").rglob("cargo-deny.exe"))
    for name, manifest in [("root", "Cargo.toml"), ("pybind", "pybind/Cargo.toml")]:
        run(f"deny-{name}", [deny, "--manifest-path", manifest, "--all-features", "--config", SOURCE / "deny.toml", "--format", "json", "check", "licenses", "sources", "bans"])


def mpi():
    """Check the optional MPI feature separately from the default build."""
    run("mpi-check", ["cargo", "check", "--locked", "--all-features", "--all-targets"],
        env={"CARGO_TARGET_DIR": str(OUT / "rust-build")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "rust", "metadata", "python", "security", "licenses", "mpi"])
    args = parser.parse_args()
    {"prepare": prepare, "rust": rust, "metadata": metadata, "python": python_checks, "security": security, "licenses": licenses, "mpi": mpi}[args.stage]()
    raise SystemExit(1 if any(r["exit_code"] for r in RESULTS) else 0)
