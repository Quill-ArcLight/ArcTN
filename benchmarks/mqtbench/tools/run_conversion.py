"""Convert every selected circuit and task, one subprocess per pair.

Each (circuit, task) pair runs in its own process with the conversion time
limit from protocol.json, so one slow quimb simplification cannot block the
others. Results are merged into <out>/conversion-summary.jsonl in manifest
order; pairs that hit the limit are recorded as excluded.

Usage:
    python run_conversion.py --protocol ../protocol-networks.json --manifest work/manifest.json \
        --circuits work/circuits --out work/networks --workers 4
"""
import argparse
import concurrent.futures
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run_pair(qasm, task, args, limit, parts):
    part = parts / f"{qasm.stem}--{task}.jsonl"
    if part.exists():
        return part
    tmp = part.with_suffix(".running")
    tmp.unlink(missing_ok=True)
    command = [sys.executable, "-W", "ignore", str(HERE / "convert_tasks.py"), "--protocol", args.protocol,
               "--out", args.out, "--summary", str(tmp), "--tasks", task, "--", str(qasm)]
    started = time.perf_counter()
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=limit)
        if done.returncode != 0 or not tmp.exists():
            record = {"circuit": qasm.stem, "task": task, "status": "excluded",
                      "reason": f"converter exited {done.returncode}: {done.stderr.strip()[-300:]}"}
            tmp.write_text(json.dumps(record) + "\n", encoding="utf-8")
    except subprocess.TimeoutExpired:
        record = {"circuit": qasm.stem, "task": task, "status": "excluded",
                  "reason": f"conversion exceeded {limit:.0f} s", "elapsed_seconds": time.perf_counter() - started}
        tmp.write_text(json.dumps(record) + "\n", encoding="utf-8")
    tmp.rename(part)
    return part


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--circuits", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    protocol = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    limit = float(protocol["preprocessing"]["conversion_time_limit_seconds"])
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    out = Path(args.out)
    parts = out / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    pairs = [(Path(args.circuits) / Path(c["member"]).name, task)
             for c in manifest["circuits"] for task in protocol["tasks"]]
    # Small circuits first, so early results are usable while large ones run.
    pairs.sort(key=lambda pair: next(c["num_qubits"] for c in manifest["circuits"]
                                     if Path(c["member"]).name == pair[0].name))
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(run_pair, qasm, task, args, limit, parts) for qasm, task in pairs]
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            future.result()
            if index % 50 == 0 or index == len(futures):
                print(f"{index}/{len(futures)} pairs done", flush=True)
    order = {(Path(c["member"]).stem, t): i for i, (c, t) in
             enumerate((c, t) for c in manifest["circuits"] for t in protocol["tasks"])}
    records = []
    for part in parts.glob("*.jsonl"):
        records.extend(json.loads(line) for line in part.read_text(encoding="utf-8").splitlines() if line)
    records.sort(key=lambda r: order.get((r["circuit"], r.get("task")), len(order)))
    (out / "conversion-summary.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    print(f"wrote {len(records)} records")


if __name__ == "__main__":
    main()
