#!/usr/bin/env python3
"""QC32 execution results of the paper (Section 6.3).

Each configuration has 160 scheduled runs (32 instances, five orders each).
FLOPs and times are geometric means over the runs that all five
configurations completed. Total time adds the wall time of the planning
process and that of the execution process for the same order.

Example:
    python qc32/analyze_execution.py
    # rerun results: totals use the process times of both collected runs
    python qc32/analyze_execution.py --runs work/qc32-execution-collected/runs.csv \
        --planning-runs work/qc32-planning-collected/runs.csv
"""
import argparse
import csv
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIGURATIONS = ["arctn-heavy-rate-on", "arctn-light-rate-on", "omeco-treesa8", "cotengra-autohq-optuna",
                  "cotengra-hyper128-optuna"]


def geometric_mean(values):
    return math.exp(math.fsum(math.log(v) for v in values) / len(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=HERE / "results/execution/runs.csv")
    parser.add_argument("--planning-runs", type=Path, help="needed when runs.csv has no total_seconds column")
    args = parser.parse_args()

    planning = {}
    if args.planning_runs:
        with args.planning_runs.open(newline="") as handle:
            planning = {(r["network"], r["configuration"], int(r["seed"])): float(r["process_seconds"])
                        for r in csv.DictReader(handle) if r["status"] == "success"}
    runs, scheduled = {}, {c: 0 for c in CONFIGURATIONS}
    with args.runs.open(newline="") as handle:
        for r in csv.DictReader(handle):
            if r["configuration"] not in scheduled:
                continue
            scheduled[r["configuration"]] += 1
            if r["status"] != "success":
                continue
            key = (r["network"], r["configuration"], int(r["seed"]))
            total = float(r["total_seconds"]) if r.get("total_seconds") else planning[key] + float(r["process_seconds"])
            runs[key] = {"flops": int(r["flops"]), "contraction": float(r["contraction_seconds"]), "total": total}
    positions = {(n, s) for n, c, s in runs}
    common = sorted(p for p in positions if all((p[0], c, p[1]) in runs for c in CONFIGURATIONS))
    means = {c: {m: geometric_mean([runs[n, c, s][m] for n, s in common]) for m in ("flops", "contraction", "total")}
             for c in CONFIGURATIONS}
    print(f"{len(common)} runs on {len({n for n, _ in common})} instances completed by all five configurations")
    print(f"{'configuration':26s} {'completed':>10s} {'FLOPs (1e9)':>12s} {'contraction (s)':>16s} {'total (s)':>10s} {'Heavy reduction (%)':>20s}")
    heavy = means[CONFIGURATIONS[0]]["total"]
    for c in CONFIGURATIONS:
        done = sum(1 for n, cc, s in runs if cc == c)
        m = means[c]
        reduction = "---" if c == CONFIGURATIONS[0] else f"{100 * (1 - heavy / m['total']):.2f}"
        print(f"{c:26s} {done:>4d}/{scheduled[c]:<5d} {m['flops'] / 1e9:12.3f} {m['contraction']:16.2f} {m['total']:10.2f} {reduction:>20s}")


if __name__ == "__main__":
    main()
