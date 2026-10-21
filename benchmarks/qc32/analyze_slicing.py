#!/usr/bin/env python3
"""QC32 slicing results of the paper (Section 6.4).

For every instance and configuration, total sliced FLOPs and full planning
time (order search plus slicing) are geometric means over the five runs; the
script then takes geometric means over instances. The scalar group uses the
scalar-output instances where all four configurations finished all five runs
(18 of 20); the tensor group uses all 12 tensor-output instances. Changes are
for Dynamic relative to each other configuration; negative values mean lower
FLOPs or shorter planning time.

Example:
    python qc32/analyze_slicing.py
"""
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIXED, DYNAMIC = "arctn-heavy-fixed-tree-post-slicing", "arctn-heavy-dynamic-slicing"
OTHERS = [(FIXED, "ArcTN Fixed"), ("omeco-treesa8-treesaslicer8", "OMECO"),
          ("cotengra-autohq-post-search-slicing", "Cotengra AutoHQ")]
CONFIGURATIONS = [FIXED, DYNAMIC] + [c for c, _ in OTHERS[1:]]


def geometric_mean(values):
    return math.exp(math.fsum(math.log(v) for v in values) / len(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=HERE / "results/slicing/runs.csv")
    args = parser.parse_args()

    groups = defaultdict(list)
    with args.runs.open(newline="") as handle:
        for row in csv.DictReader(handle):
            groups[row["group"], row["network"], row["configuration"]].append(row)
    for group in ("scalar", "tensor"):
        instances = sorted({n for g, n, _ in groups if g == group})
        used = [i for i in instances if all(len(groups[group, i, c]) == 5 and
                                            all(r["status"] == "success" for r in groups[group, i, c])
                                            for c in CONFIGURATIONS)]
        means = {}
        for c in CONFIGURATIONS:
            per_instance = [{m: geometric_mean([float(r[f]) for r in groups[group, i, c]])
                             for m, f in (("flops", "flops"), ("time", "planning_seconds"))} for i in used]
            means[c] = {m: geometric_mean([v[m] for v in per_instance]) for m in ("flops", "time")}
        print(f"{len(used)} {group}-output instances")
        print(f"  {'Dynamic relative to':20s} {'FLOPs change (%)':>17s} {'time change (%)':>16s}")
        for c, label in OTHERS:
            changes = [100 * (means[DYNAMIC][m] / means[c][m] - 1) for m in ("flops", "time")]
            print(f"  {label:20s} {changes[0]:+17.2f} {changes[1]:+16.2f}")


if __name__ == "__main__":
    main()
