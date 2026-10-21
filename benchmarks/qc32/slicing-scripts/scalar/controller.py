#!/usr/bin/env python3
"""Work-conserving controller for the frozen QC32 slicing benchmark."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Any, Mapping

import psutil


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"not a JSON object: {path}")
    return value


@dataclass(frozen=True)
class Cell:
    cell_id: str
    method: str
    repeat: int
    instance: str
    instance_path: Path


@dataclass
class LiveCell:
    cell: Cell
    slot: tuple[int, ...]
    process: subprocess.Popen[bytes]
    root: psutil.Process
    started_monotonic: float
    started_utc: str
    stdout_path: Path
    stderr_path: Path
    stdout_handle: Any
    stderr_handle: Any
    peak_rss_bytes: int = 0
    peak_threads: int = 0
    peak_processes: int = 0
    observed_affinities: set[tuple[int, ...]] = field(default_factory=set)
    observed_cpu_seconds: dict[tuple[int, float], float] = field(default_factory=dict)


def load_protocol(path: Path) -> dict[str, Any]:
    value = load_json(path)
    require(
        value.get("schema") == "qc32-final-heavy-slicing-protocol",
        "protocol schema drifted",
    )
    require(value.get("expected_cell_count") == 800, "expected cell count drifted")
    resources = value["resources"]
    require(resources["physical_cores_per_cell"] == 8, "cell width drifted")
    require(resources["maximum_concurrent_cells"] == 12, "slot count drifted")
    require(resources["hard_wall_seconds_per_cell"] == 3600, "hard wall drifted")
    require(value["queue"]["barrier_between_methods"] is False, "method barrier was enabled")
    return value


def load_slots(
    path: Path, protocol: Mapping[str, Any]
) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]:
    value = load_json(path)
    require(value.get("schema") == "qc32-live-physical-slots-v1", "slot schema drifted")
    slots = tuple(tuple(int(cpu) for cpu in row) for row in value["slots"])
    require(len(slots) == int(protocol["resources"]["maximum_concurrent_cells"]), "slot count drifted")
    require(all(len(slot) == 8 and len(set(slot)) == 8 for slot in slots), "slot width drifted")
    flat = [cpu for slot in slots for cpu in slot]
    require(len(flat) == len(set(flat)), "CPU slots overlap")
    require(value.get("live_gate_status") == "passed", "live CPU gate did not pass")
    reserved = tuple(int(cpu) for cpu in value["reserved_logical_cpus"])
    require(len(reserved) == len(set(reserved)), "reserved logical CPUs repeat")
    require(set(flat).issubset(reserved), "worker CPUs are not reserved")
    require(len(reserved) == 194, "expected 96 worker cores, 96 SMT siblings and one control core pair")
    return slots, reserved


def acquire_cpu_locks(lock_dir: Path, logical_cpus: tuple[int, ...]) -> list[int]:
    require(lock_dir.is_dir(), f"CPU lock directory is missing: {lock_dir}")
    descriptors: list[int] = []
    try:
        for cpu in sorted(logical_cpus):
            path = lock_dir / f"cpu-{cpu}.lock"
            descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o660)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BaseException:
                os.close(descriptor)
                raise
            descriptors.append(descriptor)
    except BaseException:
        for descriptor in descriptors:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)
        raise RuntimeError("one or more worker/control CPUs became reserved before launch")
    return descriptors


def load_cells(protocol: Mapping[str, Any], dataset: Path) -> list[Cell]:
    manifest = load_json(dataset / "manifest.json")
    require(
        manifest.get("schema") == protocol["corpus"]["schema"],
        "corpus manifest schema drifted",
    )
    instances = sorted(manifest["instances"], key=lambda row: int(row["ordinal"]))
    require(len(instances) == 32, "corpus instance count drifted")
    methods = [row["id"] for row in protocol["methods"]]
    repeats = [int(value) for value in protocol["repeat_ordinals"]]
    cells: list[Cell] = []
    for method in methods:
        for repeat in repeats:
            for row in instances:
                name = str(row["name"])
                instance_path = dataset / str(row["spec"])
                require(instance_path.is_file(), f"missing corpus instance: {instance_path}")
                cell_id = f"{name}--r{repeat:03d}--{method}"
                cells.append(Cell(cell_id, method, repeat, name, instance_path))
    require(len(cells) == int(protocol["expected_cell_count"]), "queue size drifted")
    require(len({cell.cell_id for cell in cells}) == len(cells), "duplicate cell id")
    return cells


def result_path(output: Path, cell: Cell) -> Path:
    return output / "results" / f"{cell.cell_id}.json"


def terminal_path(output: Path, cell: Cell) -> Path:
    return output / "terminal" / f"{cell.cell_id}.json"


def live_marker_path(output: Path, cell: Cell) -> Path:
    return output / "live" / f"{cell.cell_id}.json"


def interruption_path(output: Path, cell: Cell) -> Path:
    base = output / "interruptions" / f"{cell.cell_id}.json"
    attempt = 2
    path = base
    while path.exists():
        path = output / "interruptions" / f"{cell.cell_id}--attempt-{attempt:03d}.json"
        attempt += 1
    return path


def valid_success_result(path: Path, cell: Cell) -> bool:
    if not path.is_file():
        return False
    try:
        value = load_json(path)
    except Exception:
        return False
    return (
        value.get("schema") == "qc32-final-slicing-cell"
        and value.get("status") == "success"
        and value.get("cell_id") == cell.cell_id
        and value.get("method") == cell.method
        and value.get("instance") == cell.instance
        and value.get("repeat_ordinal") == cell.repeat
        and isinstance(value.get("ssa_path"), list)
        and isinstance(value.get("linear_path"), list)
        and isinstance(value.get("target"), dict)
        and isinstance(value.get("sliced_indices"), list)
        and isinstance(value.get("slicing_metrics"), dict)
        and isinstance(value.get("slicing_metrics", {}).get("exact_total_sliced_flops"), int)
        and value.get("slicing_metrics", {}).get("target_satisfied") is True
    )


def terminal_records(output: Path, cells: list[Cell]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for cell in cells:
        path = terminal_path(output, cell)
        if not path.is_file():
            result = result_path(output, cell)
            if result.exists():
                require(
                    valid_success_result(result, cell),
                    f"invalid orphan result requires manual recovery: {cell.cell_id}",
                )
                recovered = {
                    "schema": "qc32-final-slicing-terminal",
                    "status": "success",
                    "cell_id": cell.cell_id,
                    "instance": cell.instance,
                    "method": cell.method,
                    "repeat_ordinal": cell.repeat,
                    "started_utc": None,
                    "finished_utc": utc_now(),
                    "external_wall_seconds": None,
                    "returncode": None,
                    "physical_cpu_slot": None,
                    "resource_monitor": None,
                    "result_path": str(result),
                    "stdout_path": None,
                    "stderr_path": None,
                    "stderr_tail": None,
                    "recovered_after_controller_loss": True,
                    "recovery_limit": (
                        "The complete worker result is valid, but controller-only external timing "
                        "and process-tree resource fields cannot be reconstructed."
                    ),
                }
                atomic_write_json(path, recovered)
                records[cell.cell_id] = recovered
            continue
        value = load_json(path)
        require(value.get("schema") == "qc32-final-slicing-terminal", "terminal schema drifted")
        require(value.get("cell_id") == cell.cell_id, "terminal identity drifted")
        if value.get("status") == "controller-interrupted":
            result = result_path(output, cell)
            if result.exists():
                require(
                    valid_success_result(result, cell),
                    f"interrupted terminal has an invalid result: {cell.cell_id}",
                )
                value = {
                    **value,
                    "status": "success",
                    "finished_utc": utc_now(),
                    "result_path": str(result),
                    "recovered_after_controller_loss": True,
                    "recovery_limit": (
                        "The complete worker result is valid, but any missing controller-only "
                        "external timing or process-tree resource fields cannot be reconstructed."
                    ),
                }
                atomic_write_json(path, value)
                records[cell.cell_id] = value
            continue
        if value.get("status") == "success":
            require(valid_success_result(result_path(output, cell), cell), "successful result drifted")
        records[cell.cell_id] = value
    return records


def find_surviving_workers(cells: list[Cell], output: Path) -> list[dict[str, Any]]:
    """Find workers left alive after an unclean controller exit."""

    known = {cell.cell_id: cell for cell in cells}
    survivors: list[dict[str, Any]] = []
    for process in psutil.process_iter(("pid", "create_time", "cmdline")):
        try:
            command = list(process.info.get("cmdline") or ())
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if "--cell-id" not in command:
            continue
        position = command.index("--cell-id")
        if position + 1 >= len(command) or command[position + 1] not in known:
            continue
        cell_id = command[position + 1]
        if "--output" not in command:
            continue
        output_position = command.index("--output")
        if output_position + 1 >= len(command):
            continue
        expected_result = result_path(output, known[cell_id]).resolve()
        if Path(command[output_position + 1]).resolve() != expected_result:
            continue
        survivors.append(
            {
                "pid": int(process.info["pid"]),
                "create_time": float(process.info["create_time"]),
                "cell_id": cell_id,
            }
        )
    return survivors


def worker_environment(base: Mapping[str, str], python_packages: Path) -> dict[str, str]:
    environment = dict(base)
    for name in ("PYTHONHOME", "PYTHONSTARTUP", "PYTHONUSERBASE"):
        environment.pop(name, None)
    prior_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{python_packages}:{prior_path}" if prior_path else str(python_packages)
    )
    environment.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "RAYON_NUM_THREADS": "8",
            "COTENGRA_NUM_WORKERS": "8",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "BLIS_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
        }
    )
    return environment


def launch_cell(
    cell: Cell,
    slot: tuple[int, ...],
    args: argparse.Namespace,
) -> LiveCell:
    stdout_path = args.output / "logs" / f"{cell.cell_id}.stdout"
    stderr_path = args.output / "logs" / f"{cell.cell_id}.stderr"
    attempt = 2
    while stdout_path.exists() or stderr_path.exists():
        stdout_path = args.output / "logs" / f"{cell.cell_id}--attempt-{attempt:03d}.stdout"
        stderr_path = args.output / "logs" / f"{cell.cell_id}--attempt-{attempt:03d}.stderr"
        attempt += 1
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_handle = stdout_path.open("xb")
    stderr_handle = stderr_path.open("xb")
    marker = live_marker_path(args.output, cell)
    atomic_write_json(
        marker,
        {
            "schema": "qc32-final-slicing-live-marker",
            "status": "launching",
            "cell_id": cell.cell_id,
            "physical_cpu_slot": list(slot),
            "started_utc": utc_now(),
        },
    )
    command = [
        "taskset",
        "--cpu-list",
        ",".join(str(cpu) for cpu in slot),
        str(args.python),
        "-B",
        str(args.worker),
        "--protocol",
        str(args.protocol),
        "--instance",
        str(cell.instance_path),
        "--method",
        cell.method,
        "--repeat",
        str(cell.repeat),
        "--cell-id",
        cell.cell_id,
        "--output",
        str(result_path(args.output, cell)),
    ]
    try:
        process = subprocess.Popen(
            command,
            cwd=args.workspace,
            env=worker_environment(os.environ, args.python_packages),
            stdout=stdout_handle,
            stderr=stderr_handle,
            start_new_session=True,
        )
        root = psutil.Process(process.pid)
        atomic_write_json(
            marker,
            {
                "schema": "qc32-final-slicing-live-marker",
                "status": "running",
                "cell_id": cell.cell_id,
                "physical_cpu_slot": list(slot),
                "pid": int(process.pid),
                "process_create_time": float(root.create_time()),
                "started_utc": utc_now(),
            },
        )
    except BaseException:
        stdout_handle.close()
        stderr_handle.close()
        if marker.exists():
            marker.unlink()
        raise
    return LiveCell(
        cell=cell,
        slot=slot,
        process=process,
        root=root,
        started_monotonic=time.monotonic(),
        started_utc=utc_now(),
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        stdout_handle=stdout_handle,
        stderr_handle=stderr_handle,
    )


def sample_live(live: LiveCell) -> None:
    processes: list[psutil.Process] = [live.root]
    try:
        processes.extend(live.root.children(recursive=True))
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    rss = threads = count = 0
    for process in processes:
        try:
            identity = (process.pid, process.create_time())
            cpu = process.cpu_times()
            live.observed_cpu_seconds[identity] = max(
                live.observed_cpu_seconds.get(identity, 0.0),
                float(cpu.user + cpu.system),
            )
            rss += int(process.memory_info().rss)
            threads += int(process.num_threads())
            count += 1
            affinity = tuple(sorted(int(cpu_id) for cpu_id in process.cpu_affinity()))
            live.observed_affinities.add(affinity)
        except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
            continue
    live.peak_rss_bytes = max(live.peak_rss_bytes, rss)
    live.peak_threads = max(live.peak_threads, threads)
    live.peak_processes = max(live.peak_processes, count)


def stderr_tail(path: Path, limit: int = 4000) -> str:
    try:
        data = path.read_bytes()
    except OSError:
        return ""
    return data[-limit:].decode("utf-8", errors="replace")


def kill_process_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def finalize_live(
    live: LiveCell,
    args: argparse.Namespace,
    *,
    forced_status: str | None = None,
) -> dict[str, Any]:
    sample_live(live)
    returncode = live.process.poll()
    if returncode is None:
        kill_process_group(live.process)
        try:
            returncode = live.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            returncode = -signal.SIGKILL
    live.stdout_handle.close()
    live.stderr_handle.close()
    external_wall = time.monotonic() - live.started_monotonic
    result = result_path(args.output, live.cell)
    valid_result = valid_success_result(result, live.cell)
    if valid_result:
        status = "success"
    elif forced_status is not None:
        status = forced_status
    else:
        status = "error"
    if status != "success" and result.exists():
        status = "invalid-result"
    record = {
        "schema": "qc32-final-slicing-terminal",
        "status": status,
        "cell_id": live.cell.cell_id,
        "instance": live.cell.instance,
        "method": live.cell.method,
        "repeat_ordinal": live.cell.repeat,
        "started_utc": live.started_utc,
        "finished_utc": utc_now(),
        "external_wall_seconds": external_wall,
        "returncode": returncode,
        "physical_cpu_slot": list(live.slot),
        "resource_monitor": {
            "peak_recursive_rss_bytes": live.peak_rss_bytes,
            "peak_recursive_threads": live.peak_threads,
            "peak_recursive_processes": live.peak_processes,
            "recursive_cpu_observed_lower_bound_seconds": sum(live.observed_cpu_seconds.values()),
            "average_effective_cores_lower_bound": (
                sum(live.observed_cpu_seconds.values()) / external_wall if external_wall > 0 else None
            ),
            "observed_process_affinities": [list(value) for value in sorted(live.observed_affinities)],
            "sample_interval_seconds": float(args.sample_interval),
        },
        "result_path": str(result) if status == "success" else None,
        "stdout_path": str(live.stdout_path),
        "stderr_path": str(live.stderr_path),
        "stderr_tail": None if status == "success" else stderr_tail(live.stderr_path),
    }
    atomic_write_json(terminal_path(args.output, live.cell), record)
    marker = live_marker_path(args.output, live.cell)
    if marker.exists():
        marker.unlink()
    return record


def interrupt_live_for_resume(live: LiveCell, args: argparse.Namespace) -> dict[str, Any] | None:
    """Stop one worker without turning an unfinished cell into a terminal result."""

    sample_live(live)
    if live.process.poll() is None:
        kill_process_group(live.process)
        try:
            live.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
    live.stdout_handle.close()
    live.stderr_handle.close()
    elapsed = time.monotonic() - live.started_monotonic
    result = result_path(args.output, live.cell)
    if valid_success_result(result, live.cell):
        record = {
            "schema": "qc32-final-slicing-terminal",
            "status": "success",
            "cell_id": live.cell.cell_id,
            "instance": live.cell.instance,
            "method": live.cell.method,
            "repeat_ordinal": live.cell.repeat,
            "started_utc": live.started_utc,
            "finished_utc": utc_now(),
            "external_wall_seconds": elapsed,
            "returncode": live.process.poll(),
            "physical_cpu_slot": list(live.slot),
            "resource_monitor": {
                "peak_recursive_rss_bytes": live.peak_rss_bytes,
                "peak_recursive_threads": live.peak_threads,
                "peak_recursive_processes": live.peak_processes,
                "recursive_cpu_observed_lower_bound_seconds": sum(
                    live.observed_cpu_seconds.values()
                ),
                "observed_process_affinities": [
                    list(value) for value in sorted(live.observed_affinities)
                ],
                "sample_interval_seconds": float(args.sample_interval),
            },
            "result_path": str(result),
            "stdout_path": str(live.stdout_path),
            "stderr_path": str(live.stderr_path),
            "stderr_tail": None,
            "completed_during_controller_stop": True,
        }
        atomic_write_json(terminal_path(args.output, live.cell), record)
    else:
        require(not result.exists(), f"interrupted worker left an invalid result: {live.cell.cell_id}")
        record = None
        atomic_write_json(
            interruption_path(args.output, live.cell),
            {
                "schema": "qc32-final-slicing-interruption",
                "status": "retryable-controller-interruption",
                "cell_id": live.cell.cell_id,
                "instance": live.cell.instance,
                "method": live.cell.method,
                "repeat_ordinal": live.cell.repeat,
                "started_utc": live.started_utc,
                "interrupted_utc": utc_now(),
                "elapsed_seconds": elapsed,
                "returncode": live.process.poll(),
                "physical_cpu_slot": list(live.slot),
                "stdout_path": str(live.stdout_path),
                "stderr_path": str(live.stderr_path),
                "stderr_tail": stderr_tail(live.stderr_path),
            },
        )
    marker = live_marker_path(args.output, live.cell)
    if marker.exists():
        marker.unlink()
    return record


def write_summary(
    args: argparse.Namespace,
    protocol: Mapping[str, Any],
    cells: list[Cell],
    terminal: Mapping[str, Mapping[str, Any]],
    live: Mapping[tuple[int, ...], LiveCell],
    pending_count: int,
    *,
    controller_status: str,
) -> None:
    counts = Counter(value["status"] for value in terminal.values())
    per_method: dict[str, Counter[str]] = defaultdict(Counter)
    for value in terminal.values():
        per_method[str(value["method"])][str(value["status"])] += 1
    summary = {
        "schema": "qc32-final-slicing-summary",
        "campaign_name": protocol["campaign_name"],
        "controller_status": controller_status,
        "updated_utc": utc_now(),
        "expected_cells": len(cells),
        "terminal_cells": len(terminal),
        "success_cells": counts.get("success", 0),
        "wall_limit_cells": counts.get("wall-limit", 0),
        "error_cells": sum(value for key, value in counts.items() if key not in {"success", "wall-limit"}),
        "pending_cells": pending_count,
        "running_cells": len(live),
        "status_counts": dict(counts),
        "per_method_status_counts": {key: dict(value) for key, value in per_method.items()},
        "active": [
            {
                "cell_id": item.cell.cell_id,
                "method": item.cell.method,
                "instance": item.cell.instance,
                "repeat_ordinal": item.cell.repeat,
                "physical_cpu_slot": list(slot),
                "elapsed_seconds": time.monotonic() - item.started_monotonic,
                "peak_recursive_rss_bytes": item.peak_rss_bytes,
            }
            for slot, item in sorted(live.items())
        ],
    }
    atomic_write_json(args.output / "summary.json", summary)


def run(args: argparse.Namespace) -> int:
    protocol = load_protocol(args.protocol)
    slots, reserved_logical_cpus = load_slots(args.slots, protocol)
    cells = load_cells(protocol, args.dataset)
    args.output.mkdir(parents=True, exist_ok=True)
    for name in (
        "results",
        "terminal",
        "logs",
        "live",
        "interruptions",
        "shared-heavy-paths",
        "harness-temporary",
    ):
        (args.output / name).mkdir(exist_ok=True)
    lock_fd = os.open(args.output / "controller.lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(lock_fd)
        raise RuntimeError("another controller already holds this output directory") from exc

    survivors = find_surviving_workers(cells, args.output)
    require(
        not survivors,
        "one or more workers from this output directory are still alive; wait for or stop "
        f"them before resuming: {survivors}",
    )

    cpu_lock_fds = acquire_cpu_locks(args.lock_dir, reserved_logical_cpus)

    terminal = terminal_records(args.output, cells)
    pending = deque(cell for cell in cells if cell.cell_id not in terminal)
    live: dict[tuple[int, ...], LiveCell] = {}
    stopping = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stopping
        stopping = True

    previous_term = signal.signal(signal.SIGTERM, request_stop)
    previous_hup = signal.signal(signal.SIGHUP, request_stop)
    atomic_write_json(
        args.output / "controller.json",
        {
            "schema": "qc32-final-slicing-controller",
            "status": "running",
            "pid": os.getpid(),
            "started_utc": utc_now(),
            "protocol": str(args.protocol),
            "dataset": str(args.dataset),
            "slots": [list(slot) for slot in slots],
            "reserved_logical_cpus": list(reserved_logical_cpus),
            "cpu_lock_directory": str(args.lock_dir),
            "queue_policy": protocol["queue"],
        },
    )
    try:
        while pending or live:
            while pending and len(live) < len(slots) and not stopping:
                free_slot = next(slot for slot in slots if slot not in live)
                cell = pending.popleft()
                live[free_slot] = launch_cell(cell, free_slot, args)
                write_summary(
                    args,
                    protocol,
                    cells,
                    terminal,
                    live,
                    len(pending),
                    controller_status="running",
                )

            if stopping:
                interrupted_count = 0
                for slot, item in list(live.items()):
                    record = interrupt_live_for_resume(item, args)
                    if record is not None:
                        terminal[item.cell.cell_id] = record
                    else:
                        interrupted_count += 1
                    del live[slot]
                write_summary(
                    args,
                    protocol,
                    cells,
                    terminal,
                    live,
                    len(pending) + interrupted_count,
                    controller_status="interrupted",
                )
                return 130

            completed: list[tuple[tuple[int, ...], LiveCell, str | None]] = []
            for slot, item in live.items():
                sample_live(item)
                elapsed = time.monotonic() - item.started_monotonic
                if elapsed >= float(protocol["resources"]["hard_wall_seconds_per_cell"]):
                    completed.append((slot, item, "wall-limit"))
                elif item.process.poll() is not None:
                    completed.append((slot, item, None))

            if completed:
                for slot, item, forced in completed:
                    record = finalize_live(item, args, forced_status=forced)
                    terminal[item.cell.cell_id] = record
                    del live[slot]
                write_summary(
                    args,
                    protocol,
                    cells,
                    terminal,
                    live,
                    len(pending),
                    controller_status="running",
                )
                continue

            write_summary(
                args,
                protocol,
                cells,
                terminal,
                live,
                len(pending),
                controller_status="running",
            )
            time.sleep(float(args.sample_interval))

        status = "complete" if all(value["status"] == "success" for value in terminal.values()) else "complete-with-failures"
        write_summary(
            args,
            protocol,
            cells,
            terminal,
            live,
            0,
            controller_status=status,
        )
        atomic_write_json(
            args.output / "completion.json",
            {
                "schema": "qc32-final-slicing-completion",
                "status": status,
                "finished_utc": utc_now(),
                "expected_cells": len(cells),
                "terminal_cells": len(terminal),
                "success_cells": sum(value["status"] == "success" for value in terminal.values()),
            },
        )
        return 0
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        signal.signal(signal.SIGHUP, previous_hup)
        for descriptor in cpu_lock_fds:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--protocol", type=Path, required=True)
    value.add_argument("--slots", type=Path, required=True)
    value.add_argument("--lock-dir", type=Path, required=True)
    value.add_argument("--dataset", type=Path, required=True)
    value.add_argument("--workspace", type=Path, required=True)
    value.add_argument("--python", type=Path, required=True)
    value.add_argument("--python-packages", type=Path, required=True)
    value.add_argument("--worker", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--sample-interval", type=float, default=0.5)
    return value


def main() -> int:
    args = parser().parse_args()
    for name in (
        "protocol",
        "slots",
        "lock_dir",
        "dataset",
        "workspace",
        "python_packages",
        "worker",
        "output",
    ):
        setattr(args, name, getattr(args, name).resolve())
    # Keep the virtual-environment launcher path intact. Path.resolve() follows
    # env/bin/python to the system interpreter and loses that environment's
    # installed Cotengra and OMECO packages.
    args.python = args.python.absolute()
    try:
        return run(args)
    except Exception as exc:
        print(f"controller failed: {type(exc).__name__}: {exc}", file=__import__("sys").stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
