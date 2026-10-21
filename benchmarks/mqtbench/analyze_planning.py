#!/usr/bin/env python3
"""MQT Bench results of the paper (Section 6.6 and Appendix B).

First the networks of each task are counted by their number of tensors after
quimb simplification. Then, for every planned network, each configuration's
FLOPs, writes, largest result, peak live elements and planning time are
divided by those of ArcTN Heavy, and the geometric mean of these ratios is
taken over all networks, over small (3-999 tensors) and large (1000-4999)
networks, and over each task. Values above one favor Heavy.

Example:
    python mqtbench/analyze_planning.py
    python mqtbench/analyze_planning.py --runs work/mqtbench-collected/runs.csv
"""
import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = "arctn-heavy-rate-off"
CONFIGURATIONS = [("arctn-light-rate-off", "ArcTN Light"), ("cotengra-hyper128-optuna", "Cotengra Hyper128"),
                  ("cotengra-autohq-optuna-rate-off", "Cotengra AutoHQ"), ("cotengra-auto-cmaes-rate-off", "Cotengra Auto"),
                  ("omeco-treesa8", "OMECO TreeSA-8 (50 iter.)"), ("omeco-treesa8-niters200", "OMECO TreeSA-8 (200 iter.)"),
                  ("google-tnco-sa8-flops", "TNCO"), ("google-tnco-sa8-fuse4-flops", "TNCO (fuse=4)")]
METRICS = [("flops", "FLOPs"), ("total_write", "Writes"), ("max_size", "Largest"), ("peak_size", "Peak"),
           ("planning_seconds", "Time")]
TASKS = ["amplitude", "expectation_z", "expectation_zz", "state"]


def network_counts(path):
    counts = defaultdict(Counter)
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["status"] == "excluded":
                size = "timeout"
            elif row["status"] == "trivial":
                size = "trivial"
            else:
                tensors = int(row["tensors"])
                size = "small" if tensors < 1000 else "large" if tensors < 5000 else "very large"
            counts[row["task"]][size] += 1
    columns = ["trivial", "small", "large", "very large", "timeout"]
    print(f"{'task':16s} {'networks':>8s} " + " ".join(f"{c:>10s}" for c in columns))
    for task in TASKS + ["total"]:
        c = sum(counts.values(), Counter()) if task == "total" else counts[task]
        print(f"{task:16s} {sum(c.values()):8d} " + " ".join(f"{c[k]:10d}" for k in columns))


def ratio(row, base, metric):
    return math.log(float(row[metric])) - math.log(float(base[metric]))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=HERE / "results/planning/runs.csv")
    args = parser.parse_args()

    network_counts(HERE / "networks/conversion-summary.csv")
    with (HERE / "networks/networks.csv").open(newline="") as handle:
        info = {r["network"]: (r["task"], int(r["tensors"])) for r in csv.DictReader(handle)}
    runs = defaultdict(dict)
    with args.runs.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["status"] == "success":
                runs[row["network"]][row["configuration"]] = row
    subsets = {
        "All networks": lambda task, n: True,
        "Small networks (3-999 tensors)": lambda task, n: n < 1000,
        "Large networks (1000-4999 tensors)": lambda task, n: n >= 1000,
        "Amplitude": lambda task, n: task == "amplitude",
        "Expectation (<Z> and <ZZ>)": lambda task, n: task.startswith("expectation"),
        "State": lambda task, n: task == "state",
    }
    for title, keep in subsets.items():
        networks = [k for k, (task, n) in info.items() if keep(task, n) and BASE in runs[k]]
        print(f"\n{title} ({len(networks)} networks)")
        print(f"  {'configuration':28s} " + " ".join(f"{label:>9s}" for _, label in METRICS))
        for config, label in CONFIGURATIONS:
            pairs = [(runs[k][config], runs[k][BASE]) for k in networks if config in runs[k]]
            values = [math.exp(sum(ratio(a, b, m) for a, b in pairs) / len(pairs)) for m, _ in METRICS]
            missing = f"  ({len(networks) - len(pairs)} networks without a result)" if len(pairs) < len(networks) else ""
            print(f"  {label:28s} " + " ".join(f"{v:9.3g}" for v in values) + missing)


if __name__ == "__main__":
    main()
