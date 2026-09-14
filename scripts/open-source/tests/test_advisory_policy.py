"""The maintenance exception must never suppress a new security finding."""
from copy import deepcopy
from datetime import date
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from advisory_policy import evaluate


class AdvisoryPolicyTests(unittest.TestCase):
    def setUp(self):
        self.policy = json.loads((Path(__file__).resolve().parents[1] / "advisory-policy.json").read_text())
        self.rule = self.policy["exceptions"][0]
        self.package = {k: self.rule[k] for k in ["version", "source", "checksum"]}
        self.package["name"] = self.rule["package"]
        self.lock = {"package": [self.package]}
        self.warning = {"kind": "unmaintained", "package": self.package,
                        "advisory": {"id": self.rule["advisory"], "package": self.rule["package"], "informational": "unmaintained"}}
        self.report = {"lockfile": {"dependency-count": 1}, "settings": {"ignore": []},
                       "vulnerabilities": {"found": False, "count": 0, "list": []},
                       "warnings": {"unmaintained": [self.warning]}}

    def check(self, today=date(2026, 9, 14)):
        return evaluate(self.report, self.policy, self.lock, "Cargo.lock", today)

    def test_exact_maintenance_notice_is_accepted(self):
        self.assertTrue(self.check()["passed"])

    def test_expiry_and_before_effective_date_fail(self):
        self.assertFalse(self.check(date(2026, 10, 15))["passed"])
        self.assertFalse(self.check(date(2026, 9, 13))["passed"])

    def test_vulnerability_with_even_the_same_id_fails(self):
        self.report["vulnerabilities"] = {"found": True, "count": 1, "list": [self.warning]}
        self.assertFalse(self.check()["passed"])

    def test_additional_warning_fails(self):
        item = deepcopy(self.warning)
        item["advisory"]["id"] = "RUSTSEC-2099-0001"
        self.report["warnings"]["unmaintained"].append(item)
        self.assertFalse(self.check()["passed"])

    def test_changed_version_checksum_or_source_fails(self):
        for key, value in [("version", "0.1.8"), ("checksum", "0" * 64), ("source", "git+https://example.org/repo")]:
            with self.subTest(key=key):
                before = self.package[key]
                self.package[key] = value
                self.assertFalse(self.check()["passed"])
                self.package[key] = before

    def test_reclassified_advisory_fails(self):
        self.warning["advisory"]["informational"] = "unsound"
        self.assertFalse(self.check()["passed"])

    def test_removed_package_or_advisory_requires_review(self):
        self.report["warnings"] = {}
        self.assertFalse(self.check()["passed"])
        self.lock["package"] = []
        self.assertFalse(self.check()["passed"])

    def test_incomplete_or_suppressed_audit_fails(self):
        self.report["settings"]["ignore"] = [self.rule["advisory"]]
        self.assertFalse(self.check()["passed"])
        self.report["settings"]["ignore"] = []
        self.report["lockfile"]["dependency-count"] = 0
        self.assertFalse(self.check()["passed"])

    def test_unaffected_python_lock_is_allowed(self):
        report = deepcopy(self.report)
        report["warnings"] = {}
        self.assertTrue(evaluate(report, self.policy, self.lock, "pybind/Cargo.lock", date(2026, 9, 14))["passed"])

    def test_missing_vulnerability_results_fail(self):
        del self.report["vulnerabilities"]
        self.assertFalse(self.check()["passed"])


if __name__ == "__main__":
    unittest.main()
