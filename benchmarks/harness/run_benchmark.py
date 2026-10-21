#!/usr/bin/env python3
"""Run every (seed, network, configuration) of a protocol on pinned 8-core slots.

All configurations share one queue: seed by seed, networks from fewest to most
tensors, and every configuration for a network before the next network. Each
run is a separate worker process pinned to eight physical cores of one CPU
package (Linux only): worker.py for planning protocols, execute_worker.py for
the execution protocol. The protocol caps the number of concurrent runs, the
number of concurrent TNCO runs, and the wall time of a run. Finished runs are
skipped, so the scheduler can be stopped and started again; stop running
workers before starting it again.

Example:
    python harness/run_benchmark.py --protocol qc32/protocol-planning.json \
        --networks qc32/networks --results work/qc32-planning
    # leave Cotengra Hyper128 for later
    python harness/run_benchmark.py ... --skip cotengra-hyper128-optuna
    python harness/run_benchmark.py ... --only cotengra-hyper128-optuna
"""
import argparse
import collections
import csv
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def physical_core_slots(cores_per_slot):
    """Group one logical CPU per physical core into slots, never across CPU packages."""
    by_package = collections.defaultdict(dict)
    for cpu_dir in sorted(Path("/sys/devices/system/cpu").glob("cpu[0-9]*"), key=lambda p: int(p.name[3:])):
        topology = cpu_dir / "topology"
        if not topology.exists():
            continue
        package = int((topology / "physical_package_id").read_text())
        core = int((topology / "core_id").read_text())
        by_package[package].setdefault(core, int(cpu_dir.name[3:]))
    slots = []
    for package in sorted(by_package):
        cpus = [by_package[package][core] for core in sorted(by_package[package])]
        for start in range(0, len(cpus) - cores_per_slot + 1, cores_per_slot):
            slots.append(cpus[start:start + cores_per_slot])
    return slots


def stamp_exit(proc):
    """Record when the worker exits, so its wall time does not depend on the polling interval."""
    proc.wait()
    proc.exit_time = time.time()


def mem_available_bytes():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return 0


def add_process_seconds(result_path, seconds):
    """Store the wall time of the whole worker process, as seen from here, in its result."""
    value = json.loads(result_path.read_text(encoding="utf-8"))
    value["process_seconds"] = seconds
    tmp = result_path.with_name(f".{result_path.name}.tmp")
    tmp.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(result_path)


def load_index(networks_dir, only, max_tensors):
    with (networks_dir / "index.csv").open(newline="") as handle:
        rows = [(r["name"], int(r["tensors"])) for r in csv.DictReader(handle)]
    if max_tensors is not None:
        rows = [r for r in rows if r[1] <= max_tensors]
    if only:
        rows = [r for r in rows if r[0] in only]
    rows.sort(key=lambda r: (r[1], r[0]))
    return [name for name, _ in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--networks", type=Path, required=True, help="directory with <name>.json files and index.csv")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="*", help="default: the protocol's seeds")
    parser.add_argument("--only", nargs="*", default=[], help="run only these configurations")
    parser.add_argument("--skip", nargs="*", default=[], help="leave out these configurations")
    parser.add_argument("--networks-subset", default="", help="comma-separated network names")
    parser.add_argument("--max-tensors", type=int)
    parser.add_argument("--max-concurrent", type=int, help="default: the protocol's limit")
    parser.add_argument("--limit", type=int, help="stop after starting this many runs")
    parser.add_argument("--strict-versions", action="store_true")
    parser.add_argument("--orders", type=Path, help="execution only: directory written by unpack_orders.py")
    parser.add_argument("--arrays", type=Path, help="execution only: directory with <name>.npz input arrays")
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    resources = protocol["resources"]
    configurations = [m["id"] for m in protocol["configurations"]]
    unknown = (set(args.only) | set(args.skip)) - set(configurations)
    if unknown:
        sys.exit(f"unknown configurations: {sorted(unknown)}")
    if args.only:
        configurations = [m for m in configurations if m in args.only]
    configurations = [m for m in configurations if m not in args.skip]
    tnco = {m["id"] for m in protocol["configurations"] if m["library"] == "TNCO"}
    seeds = args.seeds if args.seeds is not None else protocol["seeds"]
    only = {name for name in args.networks_subset.split(",") if name}
    networks = load_index(args.networks, only, args.max_tensors)
    cores = resources["physical_cores_per_run"]
    wanted = args.max_concurrent or resources["max_concurrent_runs"]
    slots = physical_core_slots(cores)[:wanted]
    if not slots:
        sys.exit(f"no slot of {cores} physical cores found (Linux CPU topology is required)")
    if len(slots) < wanted:
        print(f"note: {len(slots)} slots available, the protocol allows {wanted}", file=sys.stderr)

    queue = collections.deque()
    for seed in seeds:
        for network in networks:
            for config in configurations:
                cell = f"{network}--s{seed:03d}--{config}"
                out = args.results / f"seed-{seed}" / f"{cell}.json"
                fail = args.results / f"seed-{seed}" / "failed" / f"{cell}.json"
                if not out.exists() and not fail.exists():
                    queue.append((seed, network, config, cell, out, fail))
    total, launched = len(queue), 0
    numeric = str(resources.get("numeric_threads", 1))
    env = {**os.environ, "RAYON_NUM_THREADS": str(cores), "COTENGRA_NUM_WORKERS": str(cores),
           "OMP_NUM_THREADS": numeric, "OPENBLAS_NUM_THREADS": numeric, "MKL_NUM_THREADS": numeric,
           "NUMBA_NUM_THREADS": "1"}
    worker = HERE / protocol.get("worker", "worker.py")
    running = {}  # slot index -> (process, start time, cell, log path)
    finished = collections.Counter()
    args.results.mkdir(parents=True, exist_ok=True)
    status_path, last_status = args.results / "status.json", 0.0

    def write_status(final=False):
        value = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "queued_at_start": total, "launched": launched,
                 "running": len(running), "remaining": len(queue), "finished": dict(finished), "complete": final}
        tmp = status_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(value, indent=1) + "\n", encoding="utf-8")
        tmp.replace(status_path)

    def record_failure(cell, reason, elapsed, log_path):
        seed, network, config, cell_id, out, fail = cell
        fail.parent.mkdir(parents=True, exist_ok=True)
        tail = log_path.read_text(errors="replace")[-2000:] if log_path.exists() else ""
        fail.write_text(json.dumps({"status": reason, "cell_id": cell_id, "network": network, "configuration": config,
                                    "seed": seed, "elapsed_seconds": elapsed, "log_tail": tail}) + "\n")

    while queue or running:
        now = time.time()
        for index, (proc, started, cell, log_path) in list(running.items()):
            done = hasattr(proc, "exit_time")
            elapsed, code = (proc.exit_time if done else now) - started, proc.returncode if done else None
            if code is None and elapsed > resources["run_time_limit_seconds"]:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                # A planner that has already written its order but has not exited keeps the order.
                if cell[4].exists():
                    finished["success_after_time_limit"] += 1
                else:
                    record_failure(cell, "time-limit", elapsed, log_path)
                    finished["time-limit"] += 1
                del running[index]
            elif code is not None:
                if code != 0 or not cell[4].exists():
                    record_failure(cell, f"worker-exit-{code}", elapsed, log_path)
                    finished["failed"] += 1
                else:
                    add_process_seconds(cell[4], elapsed)
                    finished["success"] += 1
                del running[index]
        free = [i for i in range(len(slots)) if i not in running]
        tnco_running = sum(1 for _, _, c, _ in running.values() if c[2] in tnco)
        while free and queue and (args.limit is None or launched < args.limit):
            if mem_available_bytes() < resources.get("min_available_memory_bytes", 0):
                break
            pick = None
            for position, cell in enumerate(queue):
                if cell[0] != queue[0][0]:
                    break  # finish starting one seed before the next
                if cell[2] in tnco and tnco_running >= resources["max_concurrent_tnco_runs"]:
                    continue
                pick = position
                break
            if pick is None:
                break
            cell = queue[pick]
            del queue[pick]
            seed, network, config, cell_id, out, fail = cell
            log_path = out.parent / "logs" / f"{cell_id}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            index = free.pop(0)
            command = ["taskset", "-c", ",".join(map(str, slots[index])), sys.executable, str(worker),
                       "--protocol", str(args.protocol.resolve()), "--network", str((args.networks / f"{network}.json").resolve()),
                       "--configuration", config, "--seed", str(seed), "--cell-id", cell_id, "--output", str(out.resolve())]
            if args.strict_versions:
                command.append("--strict-versions")
            for flag, value in (("--orders", args.orders), ("--arrays", args.arrays)):
                if value is not None:
                    command += [flag, str(value.resolve())]
            started = time.time()
            with open(log_path, "w") as log:
                proc = subprocess.Popen(command, cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT,
                                        start_new_session=True)
            running[index] = (proc, started, cell, log_path)
            threading.Thread(target=stamp_exit, args=(proc,), daemon=True).start()
            launched += 1
            if config in tnco:
                tnco_running += 1
        if args.limit is not None and launched >= args.limit and not running:
            break
        if time.time() - last_status > 30:
            write_status()
            last_status = time.time()
        time.sleep(0.5)
    write_status(final=not queue)
    print(json.dumps({"launched": launched, "finished": dict(finished), "remaining": len(queue)}))


if __name__ == "__main__":
    main()
