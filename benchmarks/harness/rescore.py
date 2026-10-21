#!/usr/bin/env python3
"""Recompute the costs of recorded orders and compare them with runs.csv.

Every order in paths.jsonl.xz is replayed on its network with scoring.py; for
the slicing results, every plan in plans.jsonl.xz (order and sliced indices)
is replayed and checked against its size target. The script stops at the
first order that is invalid or whose costs differ from the recorded ones.

Example:
    python harness/rescore.py --networks qc32/networks --results qc32/results/planning
    python harness/rescore.py --networks qc32/networks --results qc32/results/slicing
    python harness/rescore.py --networks mqtbench/networks/networks.jsonl.xz --results mqtbench/results/planning
"""
import argparse
import csv
import json
import lzma
from pathlib import Path

from scoring import evaluate_sliced_plan, evaluate_ssa_path, load_network, require


def networks_by_name(source):
    if source.is_dir():
        return {p.stem: load_network(p) for p in source.glob("*.json")}
    found = {}
    with lzma.open(source, "rt") as handle:
        for line in handle:
            value = json.loads(line)
            value["size_dict"] = {int(k): int(v) for k, v in value["size_dict"].items()}
            found[value["name"]] = value
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--networks", type=Path, required=True, help="directory of network files or a .jsonl.xz bundle")
    parser.add_argument("--results", type=Path, required=True, help="directory with runs.csv and paths.jsonl.xz")
    args = parser.parse_args()

    networks = networks_by_name(args.networks)
    with (args.results / "runs.csv").open(newline="") as handle:
        recorded = {(r["network"], r["configuration"], int(r["seed"])): r for r in csv.DictReader(handle)}
    checked = 0
    sliced = (args.results / "plans.jsonl.xz").exists()
    fields = ("flops", "n_slices", "per_slice_max_size", "output_indices_sliced") if sliced else \
        ("flops", "total_write", "max_size", "peak_size")
    with lzma.open(args.results / ("plans.jsonl.xz" if sliced else "paths.jsonl.xz"), "rt") as handle:
        for line in handle:
            value = json.loads(line)
            key = (value["network"], value["configuration"], value["seed"])
            spec = networks[value["network"]]
            if sliced:
                metrics = evaluate_sliced_plan(spec["inputs"], spec["output"], spec["size_dict"], value["ssa_path"],
                                               value["sliced_indices"], value["target_size"],
                                               value["output_slicing_allowed"])
            else:
                metrics = evaluate_ssa_path(spec["inputs"], spec["output"], spec["size_dict"], value["ssa_path"])
            row = recorded[key]
            for field in fields:
                require(metrics[field] == int(row[field]), f"{key}: {field} {metrics[field]} != recorded {row[field]}")
            checked += 1
    successes = sum(1 for r in recorded.values() if r["status"] == "success")
    require(checked == successes, f"{checked} orders for {successes} successful runs")
    print(f"all {checked} recorded orders are valid and match the recorded costs")


if __name__ == "__main__":
    main()
