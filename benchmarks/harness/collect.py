#!/usr/bin/env python3
"""Collect worker results into runs.csv (and paths.jsonl.xz for planning runs).

The columns are those of the recorded results in this directory, so the
analysis scripts can read either.

Example:
    python harness/collect.py work/qc32-planning --out work/qc32-planning-collected
    python harness/collect.py work/qc32-execution --out work/qc32-execution-collected
"""
import argparse
import csv
import json
import lzma
from pathlib import Path

PLANNING = ["network", "configuration", "seed", "status", "flops", "total_write", "max_size", "peak_size",
            "planning_seconds", "process_seconds"]
EXECUTION = ["network", "configuration", "seed", "status", "flops", "dtype", "compile_seconds",
             "contraction_seconds", "process_seconds"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path, help="the --results directory of run_benchmark.py")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows, paths, columns = {}, [], PLANNING
    for path in sorted(args.results.glob("seed-*/failed/*.json")):
        value = json.loads(path.read_text())
        key = (value["network"], value["configuration"], value["seed"])
        rows[key] = {"network": key[0], "configuration": key[1], "seed": key[2], "status": value["status"]}
    for path in sorted(args.results.glob("seed-*/*.json")):
        value = json.loads(path.read_text())
        key = (value["network"], value["configuration"], value["seed"])
        row = {"network": key[0], "configuration": key[1], "seed": key[2], "status": value["status"],
               "process_seconds": repr(value["process_seconds"]) if "process_seconds" in value else ""}
        # A finished run wins over a failure record left by a duplicate worker.
        if value["schema"] == "arctn-benchmarks-execution-v1":
            columns = EXECUTION
            row.update({k: value.get(k, "") for k in ("flops", "dtype")})
            for k in ("compile_seconds", "contraction_seconds"):
                row[k] = repr(value[k]) if value.get(k) is not None else ""
        else:
            m = value["metrics"]
            row.update(flops=m["flops"], total_write=m["total_write"], max_size=m["max_size"],
                       peak_size=m["peak_size"], planning_seconds=repr(value["planning_seconds"]))
            paths.append({"network": key[0], "configuration": key[1], "seed": key[2], "ssa_path": value["ssa_path"]})
        rows[key] = row
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "runs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, restval="", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows[key] for key in sorted(rows, key=lambda k: (k[1], k[0], k[2])))
    if columns is PLANNING:
        with lzma.open(args.out / "paths.jsonl.xz", "wt") as handle:
            for value in sorted(paths, key=lambda v: (v["configuration"], v["network"], v["seed"])):
                handle.write(json.dumps(value, separators=(",", ":")) + "\n")
    print(f"{len(rows)} runs -> {args.out}")


if __name__ == "__main__":
    main()
