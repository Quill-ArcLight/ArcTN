"""Create a complete source dependency SBOM and retain Cargo license texts.

Requires metadata-final-{root,pybind}.stdout.log from locked, all-feature Cargo
metadata. Does not infer contents of proprietary engines or installed MPI.
"""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import tomllib
import uuid

from audit import ROOT, REPORTS, write_json


def ref(package):
    return f"pkg:cargo/{package['name']}@{package['version']}"


def selected(expression):
    if not expression:
        return None
    expression = expression.replace("/", " OR ")
    if expression == "(MIT OR Apache-2.0) AND Unicode-3.0":
        return "MIT AND Unicode-3.0"
    if "AND" not in expression and "MIT" in [s.strip() for s in expression.split(" OR ")]:
        return "MIT"
    return expression


def main():
    packages, nodes, scopes = {}, {}, {}
    for scope in ["root", "pybind"]:
        metadata = json.loads((REPORTS / f"metadata-final-{scope}.stdout.log").read_text(encoding="utf-8"))
        by_id = {p["id"]: ref(p) for p in metadata["packages"]}
        for p in metadata["packages"]:
            key = ref(p)
            packages[key] = p
            scopes.setdefault(key, []).append(scope)
        for n in metadata["resolve"]["nodes"]:
            nodes.setdefault(by_id[n["id"]], set()).update(by_id[x] for x in n["dependencies"])
    checksums, missing = {}, []
    for path in [ROOT / "Cargo.lock", ROOT / "pybind/Cargo.lock"]:
        for p in tomllib.loads(path.read_text(encoding="utf-8"))["package"]:
            key = ref(p)
            if key not in packages:
                missing.append(dict(lock=path.relative_to(ROOT).as_posix(), package=key))
            if p.get("checksum"):
                checksums[key] = p["checksum"]
    if missing:
        write_json(REPORTS / "sbom-missing.json", missing)
        raise SystemExit("Cargo metadata omitted lockfile packages; inspect sbom-missing.json")
    components, inventory, missing_license_files = [], [], []
    for key, p in sorted(packages.items()):
        third_party = p["source"] is not None
        expression = selected(p["license"]) if third_party else p["license"]
        c = dict(type="library", name=p["name"], version=p["version"], purl=key,
                 **{"bom-ref": key}, properties=[dict(name="arctn:manifests", value=",".join(scopes[key]))])
        if expression:
            c["licenses"] = [{"expression": expression}]
        if key in checksums:
            c["hashes"] = [{"alg": "SHA-256", "content": checksums[key]}]
        if third_party:
            c["externalReferences"] = [dict(type="distribution", url=f"https://crates.io/api/v1/crates/{p['name']}/{p['version']}/download")]
        else:
            c["properties"].append(dict(name="arctn:component-kind", value="local source"))
        components.append(c)
        texts = []
        if third_party:
            package_root = Path(p["manifest_path"]).parent
            candidates = [f for f in package_root.rglob("*") if f.is_file() and
                          (f.name.lower().startswith(("license", "licence", "copying", "notice", "copyright"))) and
                          not f.name.endswith((".rs", ".py", ".json"))]
            for f in candidates:
                relative = f.relative_to(package_root)
                target = ROOT / "THIRD_PARTY_LICENSES/cargo" / f"{p['name']}-{p['version']}" / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(f, target)
                texts.append(dict(path=target.relative_to(ROOT).as_posix(), sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
            if not texts:
                supplement_file = ROOT / "THIRD_PARTY_LICENSES/license-supplements.json"
                supplements = json.loads(supplement_file.read_text(encoding="utf-8")) if supplement_file.exists() else []
                for row in supplements:
                    if row.get("package") == f"{p['name']}@{p['version']}":
                        f = ROOT / row["path"]
                        if hashlib.sha256(f.read_bytes()).hexdigest() != row["sha256"]:
                            raise SystemExit(f"Supplement hash changed: {f}")
                        texts.append({k: row[k] for k in ["path", "sha256", "url"]})
                if not texts:
                    missing_license_files.append(key)
        inventory.append(dict(name=p["name"], version=p["version"], purl=key,
            source=p["source"], declared_license=p["license"], selected_license=expression,
            scope=scopes[key], crate_archive_sha256=checksums.get(key), license_files=texts))
    aggregate = "arctn-source-release"
    bom = dict(bomFormat="CycloneDX", specVersion="1.5", serialNumber=f"urn:uuid:{uuid.uuid4()}", version=1,
        metadata=dict(timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            component={"type": "application", "name": "ArcTN source release", "version": "1.0.0", "bom-ref": aggregate},
            properties=[dict(name="arctn:scope", value="Union of both Cargo lockfiles, all features/platform dependencies including dev/build. Excludes Python distributions, proprietary engine, MPI runtime, OS and arbitrary external backends.")]),
        components=components, dependencies=[{"ref": aggregate, "dependsOn": ["pkg:cargo/arctn@1.0.0", "pkg:cargo/arctn-py@1.0.0"]}] +
            [{"ref": key, "dependsOn": sorted(nodes.get(key, []))} for key in sorted(packages)])
    write_json(REPORTS / "workspace-source.cdx.json", bom)
    write_json(REPORTS / "cargo-license-inventory.json", inventory)
    write_json(REPORTS / "cargo-coverage.json", dict(components=len(packages), third_party=sum(p["source"] is not None for p in packages.values()),
        lockfiles_fully_covered=not missing, missing_license_files=missing_license_files,
        lock_hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ["Cargo.lock", "pybind/Cargo.lock"]}))
    with (ROOT / "THIRD_PARTY_LICENSES/cargo-components.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["name", "version", "declared_license", "selected_license", "crate_sha256", "source_url", "license_directory"])
        for row in inventory:
            if row["source"]:
                writer.writerow([row["name"], row["version"], row["declared_license"], row["selected_license"], row["crate_archive_sha256"],
                    f"https://crates.io/crates/{row['name']}/{row['version']}", f"cargo/{row['name']}-{row['version']}"])
    print(json.dumps(dict(components=len(packages), missing_license_files=missing_license_files)))


if __name__ == "__main__":
    main()
