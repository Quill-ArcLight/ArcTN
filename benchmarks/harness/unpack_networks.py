#!/usr/bin/env python3
"""Write each network of a .jsonl.xz bundle to <name>.json, plus index.csv for run_benchmark.py.

Example:
    python harness/unpack_networks.py mqtbench/networks/networks.jsonl.xz work/mqtbench-networks
"""
import argparse
import csv
import json
import lzma
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    with lzma.open(args.bundle, "rt") as handle:
        for line in handle:
            value = json.loads(line)
            (args.out / f"{value['name']}.json").write_text(json.dumps(value, separators=(",", ":")) + "\n")
            rows.append({"name": value["name"], "tensors": len(value["inputs"])})
    with (args.out / "index.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["name", "tensors"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} networks -> {args.out}")


if __name__ == "__main__":
    main()
