"""Accept only a time-limited, exact maintenance advisory; preserve raw scans.

CLI runs cargo-audit without ignores against both complete locks, then applies
the same policy as the local audit runner. Errors and new vulnerabilities fail.
"""
from datetime import date, datetime, timezone
import argparse
import json
from pathlib import Path
import subprocess
import tomllib


def evaluate(report, policy, lock, lockfile, today=None):
    today = today or datetime.now(timezone.utc).date()
    errors, accepted = [], []
    if not policy.get("owner") or not date.fromisoformat(policy["accepted_on"]) <= today < date.fromisoformat(policy["review_due"]):
        errors.append("Maintenance acceptance is not effective or has expired; review required")
    rules = [e for e in policy["exceptions"] if e["lockfile"] == lockfile]
    for rule in rules:
        matches = [p for p in lock["package"] if p["name"] == rule["package"]]
        if len(matches) != 1 or any(matches[0].get(k) != rule[k] for k in ["version", "source", "checksum"]):
            errors.append("Accepted package version/source/checksum changed: " + rule["package"])
    if report.get("error") or report.get("settings", {}).get("ignore"):
        errors.append("Audit failed or was run with ignored advisories")
    if report.get("lockfile", {}).get("dependency-count") != len(lock["package"]):
        errors.append("Audit does not cover the supplied complete lockfile")
    vulns = report.get("vulnerabilities")
    if not isinstance(vulns, dict) or vulns.get("found") is not False or vulns.get("count") != 0 or vulns.get("list") != []:
        errors.append("Known vulnerability or incomplete vulnerability report")
    warnings = report.get("warnings")
    if not isinstance(warnings, dict):
        errors.append("Missing warnings report")
        warnings = {}
    encountered = set()
    for kind, entries in warnings.items():
        for item in entries:
            package = item.get("package", {})
            advisory = item.get("advisory", {})
            matching = [rule for rule in rules if
                kind == rule["kind"] == item.get("kind") == advisory.get("informational") == "unmaintained" and
                advisory.get("id") == rule["advisory"] and advisory.get("package") == rule["package"] and
                package.get("name") == rule["package"] and
                all(package.get(k) == rule[k] for k in ["version", "source", "checksum"])]
            if len(matching) == 1:
                accepted.append(advisory["id"])
                encountered.add(advisory["id"])
            else:
                errors.append("Unaccepted advisory: " + str(advisory.get("id", kind)))
    for rule in rules:
        if rule["advisory"] not in encountered:
            errors.append("Accepted advisory no longer matches; review/remove stale exception: " + rule["advisory"])
    return {"passed": not errors, "errors": errors, "accepted": accepted,
            "review_due": policy["review_due"], "owner": policy["owner"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--audit-executable", default="cargo-audit")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output or root / "target/advisory-review"
    output.mkdir(parents=True, exist_ok=True)
    policy = json.loads((root / "scripts/open-source/advisory-policy.json").read_text(encoding="utf-8"))
    results = []
    for name, lockfile in [("root", "Cargo.lock"), ("pybind", "pybind/Cargo.lock")]:
        with (output / f"{name}.json").open("wb") as stdout, (output / f"{name}.stderr.log").open("wb") as stderr:
            proc = subprocess.run([args.audit_executable, "audit", "--file", lockfile, "--json"], cwd=root, stdout=stdout, stderr=stderr)
        try:
            report = json.loads((output / f"{name}.json").read_text(encoding="utf-8"))
            lock = tomllib.loads((root / lockfile).read_text(encoding="utf-8"))
            result = evaluate(report, policy, lock, lockfile)
        except (ValueError, KeyError, TypeError) as error:
            result = {"passed": False, "errors": [str(error)]}
        if proc.returncode:
            result["passed"] = False
            result["errors"].append(f"cargo-audit exited {proc.returncode}")
        result["lockfile"] = lockfile
        results.append(result)
    (output / "policy-result.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))
    return 0 if all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
