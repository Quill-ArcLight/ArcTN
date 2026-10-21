"""Select MQT Bench circuits by the rule in protocol-networks.json.

The rule depends only on file names (family, level, compiler, qubit count),
never on planner results. Input is either the downloaded archive or a text
listing of its member names (one name per line, optional tab-separated
columns after the name), so the planned manifest can be produced before the
archive is downloaded and checked against it afterwards.

Usage:
    python select_circuits.py --protocol ../protocol-networks.json --archive MQTBench_all.zip \
        --out work/manifest.json --extract-to work/circuits
"""
import argparse
import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path

NAME = re.compile(r"^(?P<family>.+?)_(?P<level>alg|indep|nativegates|mapped)_(?P<rest>.+)_(?P<n>\d+)\.qasm$")


def parse_name(name):
    """Return (family, level, compiler, n) for a benchmark file name, or None."""
    base = name.rsplit("/", 1)[-1]
    match = NAME.match(base)
    if not match:
        return None
    return match["family"], match["level"], match["rest"], int(match["n"])


def select(names, rule):
    """Apply the selection rule; return selected entries and per-family counts."""
    level, compiler = rule["level"], rule["compiler"]
    grid = set(rule["qubit_grid"])
    groups = {family: group for group in ("every_qubit_count", "fixed_sizes") for family in rule[group]}
    selected, unknown = [], set()
    for name in sorted(names):
        parsed = parse_name(name)
        if parsed is None:
            continue
        family, file_level, file_compiler, n = parsed
        if file_level != level or file_compiler != compiler:
            continue
        group = groups.get(family)
        if group is None:
            unknown.add(family)
            continue
        if group == "every_qubit_count" and n not in grid:
            continue
        selected.append({"member": name, "family": family, "group": group, "num_qubits": n})
    return selected, sorted(unknown)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--listing")
    source.add_argument("--archive")
    parser.add_argument("--out", required=True)
    parser.add_argument("--extract-to", help="write the selected members (archive only) into this directory")
    args = parser.parse_args()

    rule = json.loads(Path(args.protocol).read_text(encoding="utf-8"))["circuit_selection"]
    archive = None
    if args.archive:
        archive = zipfile.ZipFile(args.archive)
        names = archive.namelist()
    else:
        names = [line.split("\t", 1)[0] for line in Path(args.listing).read_text(encoding="utf-8").splitlines() if line]

    selected, unknown = select(names, rule)
    if unknown:
        # Every family at the chosen level must be assigned to a group, so the
        # rule cannot silently drop families that the protocol did not list.
        sys.exit(f"families at {rule['level']}/{rule['compiler']} not assigned to a group: {unknown}")
    if archive is not None:
        target = Path(args.extract_to) if args.extract_to else None
        if target is not None:
            target.mkdir(parents=True, exist_ok=True)
        for entry in selected:
            data = archive.read(entry["member"])
            entry["sha256"] = hashlib.sha256(data).hexdigest()
            if target is not None:
                # Write by base name only, so member paths cannot leave the target.
                (target / Path(entry["member"]).name).write_bytes(data)
    elif args.extract_to:
        sys.exit("--extract-to needs --archive")

    counts = {}
    for entry in selected:
        counts.setdefault(entry["group"], {}).setdefault(entry["family"], 0)
        counts[entry["group"]][entry["family"]] += 1
    manifest = {"schema_version": 1, "rule": rule, "source": args.archive or args.listing,
                "hashed": archive is not None, "total": len(selected), "counts": counts,
                "circuits": selected}
    Path(args.out).write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"selected {len(selected)} circuits")
    for group, families in counts.items():
        print(f"  {group}: {sum(families.values())} ({len(families)} families)")


if __name__ == "__main__":
    main()
