#!/usr/bin/env python3
"""Write each order of a paths.jsonl.xz file to <network>--s<seed>--<configuration>.json for execute_worker.py.

Example:
    python harness/unpack_orders.py qc32/results/planning/paths.jsonl.xz work/qc32-orders
"""
import argparse
import json
import lzma
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    count = 0
    with lzma.open(args.paths, "rt") as handle:
        for line in handle:
            value = json.loads(line)
            cell = f"{value['network']}--s{int(value['seed']):03d}--{value['configuration']}"
            (args.out / f"{cell}.json").write_text(json.dumps(value, separators=(",", ":")) + "\n")
            count += 1
    print(f"{count} orders -> {args.out}")


if __name__ == "__main__":
    main()
