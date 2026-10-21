#!/usr/bin/env python3
"""QC32 planning results of the paper (Section 6.2).

For each instance and configuration, every metric is the median of the five
runs. The script then takes geometric means over the instances where all eight
configurations finished all five runs (30 of the 32 instances; OMECO failed on
two). It prints each ArcTN preset against its baselines and the geometric
means shown in the planning overview figure.

Example:
    python qc32/analyze_planning.py
    python qc32/analyze_planning.py --runs work/qc32-collected/runs.csv
"""
import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIGURATIONS = ["arctn-light-rate-on", "arctn-heavy-rate-on", "cotengra-auto-cmaes", "cotengra-autohq-optuna",
                  "cotengra-hyper128-optuna", "omeco-treesa8", "google-tnco-sa8-flops", "google-tnco-sa8-fuse4-flops"]
PAIRS = [("arctn-light-rate-on", "cotengra-auto-cmaes"), ("arctn-heavy-rate-on", "cotengra-autohq-optuna"),
         ("arctn-heavy-rate-on", "cotengra-hyper128-optuna"), ("arctn-heavy-rate-on", "omeco-treesa8"),
         ("arctn-heavy-rate-on", "google-tnco-sa8-flops"), ("arctn-heavy-rate-on", "google-tnco-sa8-fuse4-flops")]
METRICS = ["flops", "total_write", "max_size", "peak_size"]


def geometric_mean(values):
    return math.exp(math.fsum(math.log(v) for v in values) / len(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=HERE / "results/planning/runs.csv")
    args = parser.parse_args()

    groups = defaultdict(list)
    with args.runs.open(newline="") as handle:
        for row in csv.DictReader(handle):
            groups[row["network"], row["configuration"]].append(row)
    instances = sorted({network for network, _ in groups})
    common = [i for i in instances
              if all(len(groups[i, c]) == 5 and all(r["status"] == "success" for r in groups[i, c])
                     for c in CONFIGURATIONS)]
    medians = {}
    for i in common:
        for c in CONFIGURATIONS:
            rows = groups[i, c]
            medians[i, c] = {m: statistics.median(int(r[m]) for r in rows) for m in METRICS}
            medians[i, c]["time"] = statistics.median(float(r["planning_seconds"]) for r in rows)
    means = {c: {m: geometric_mean([medians[i, c][m] for i in common]) for m in METRICS + ["time"]}
             for c in CONFIGURATIONS}

    print(f"{len(common)} of {len(instances)} instances used; left out: {', '.join(sorted(set(instances) - set(common)))}")
    print()
    print(f"{'ArcTN':22s} {'baseline':30s} {'speedup':>8s} {'FLOPs':>8s} {'writes':>8s} {'largest':>8s} {'peak':>8s}")
    print(f"{'':53s} {'':>8s} {'(reduction, %)':>35s}")
    for arctn, baseline in PAIRS:
        a, b = means[arctn], means[baseline]
        reductions = [100 * (1 - a[m] / b[m]) for m in METRICS]
        print(f"{arctn:22s} {baseline:30s} {b['time'] / a['time']:8.2f} " + " ".join(f"{v:8.2f}" for v in reductions))
    print()
    print(f"{'configuration':30s} {'planning time (s)':>17s} {'FLOPs':>12s}")
    for c in CONFIGURATIONS:
        print(f"{c:30s} {means[c]['time']:17.3f} {means[c]['flops']:12.4e}")


if __name__ == "__main__":
    main()
