#!/usr/bin/env python3
"""Run one frozen path-search and output-index slicing cell."""

from __future__ import annotations

import argparse
from collections import Counter
import importlib.metadata
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence


METHODS = (
    "arctn-heavy-fixed-tree-post-slicing",
    "arctn-heavy-dynamic-slicing",
    "cotengra-autohq-post-search-slicing",
    "omeco-treesa8-treesaslicer8",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if value == math.inf:
            return "+inf"
        if value == -math.inf:
            return "-inf"
        return value
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        return json_safe(value.item())
    return repr(value)


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False, sort_keys=True)
        handle.write("\n")
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()


def load_protocol(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(
        value.get("schema") == "qc32-open-output-slicing-protocol",
        "protocol schema drifted",
    )
    method_ids = [row["id"] for row in value["methods"]]
    require(method_ids and len(method_ids) == len(set(method_ids))
            and set(method_ids).issubset(METHODS),
            "protocol methods must be a nonempty subset of the supported methods")
    repeats = value["repeat_ordinals"]
    require(repeats and len(repeats) == len(set(repeats)), "invalid repeat ordinals")
    expected_cells = int(value["corpus"]["instance_count"]) * len(repeats) * len(method_ids)
    require(value.get("expected_cell_count") == expected_cells, "protocol cell count drifted")
    require(
        value.get("objective", {}).get("primary_quality_metric") == "total_sliced_flops",
        "protocol objective drifted",
    )
    require(all(row.get("allow_outer") is True for row in value["methods"]),
            "all methods must permit output-index slicing")
    return value


def load_spec(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(
        value.get("schema") == "einsum-quantum32-language-neutral-instance-v1",
        "instance schema drifted",
    )
    require(value.get("name") == path.stem, "instance name drifted")
    inputs = tuple(tuple(int(label) for label in term) for term in value["inputs"])
    output = tuple(int(label) for label in value["output"])
    size_dict = {int(label): int(size) for label, size in value["size_dict"]}
    require(inputs and all(len(term) == len(set(term)) for term in inputs), "invalid inputs")
    used = set(output)
    for term in inputs:
        used.update(term)
    require(used == set(size_dict), "size dictionary does not match used labels")
    require(all(size > 0 for size in size_dict.values()), "non-positive dimension")
    return {
        "name": value["name"],
        "ordinal": int(value["ordinal"]),
        "inputs": inputs,
        "output": output,
        "size_dict": size_dict,
    }


def load_target(
    protocol_path: Path,
    protocol: Mapping[str, Any],
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    target_path = protocol_path.parent / str(protocol["target"]["table"])
    value = json.loads(target_path.read_text(encoding="utf-8"))
    require(value.get("schema") == "qc32-open-output-slicing-targets", "target schema drifted")
    base_target = int(protocol["target"]["base_target_elements"])
    minimum_blocks = int(protocol["target"]["minimum_output_blocks"])
    require(base_target > 0 and minimum_blocks >= 2, "invalid output-block target")
    require(int(value["base_target_elements"]) == base_target, "base target drifted")
    rows = {str(row["instance"]): row for row in value["instances"]}
    require(len(rows) == len(value["instances"]) == int(protocol["corpus"]["instance_count"])
            and spec["name"] in rows, "target table is incomplete")
    row = dict(rows[spec["name"]])
    output_elements = product_dimensions(spec["output"], spec["size_dict"])
    expected = min(base_target, output_elements // minimum_blocks)
    require(spec["output"] and 0 < expected < output_elements,
            "supplement requires a non-scalar output larger than the target")
    require(int(row["ordinal"]) == int(spec["ordinal"]), "target ordinal drifted")
    require(int(row["output_elements"]) == output_elements, "target output size drifted")
    require(int(row["effective_target_elements"]) == expected, "effective target drifted")
    require(row.get("output_slicing_required") is True, "output slicing flag drifted")
    return {
        "base_target_elements": base_target,
        "output_elements": output_elements,
        "effective_target_elements": expected,
        "target_relaxed_for_output": False,
        "minimum_output_blocks": minimum_blocks,
        "output_slicing_required": True,
        "scalar_output": False,
    }


def ssa_to_linear(ssa_path: Sequence[Sequence[int]], n_inputs: int) -> list[list[int]]:
    live = list(range(n_inputs))
    linear: list[list[int]] = []
    for step, pair in enumerate(ssa_path):
        require(len(pair) == 2, f"SSA step {step} is not binary")
        left_id, right_id = int(pair[0]), int(pair[1])
        require(left_id != right_id, f"SSA step {step} repeats a node")
        try:
            left = live.index(left_id)
            right = live.index(right_id)
        except ValueError as exc:
            raise RuntimeError(f"SSA step {step} references a non-live node") from exc
        linear.append([left, right])
        for position in sorted((left, right), reverse=True):
            live.pop(position)
        live.append(n_inputs + step)
    require(len(ssa_path) == n_inputs - 1 and len(live) == 1, "SSA path is incomplete")
    return linear


def product_dimensions(legs: Iterable[int], size_dict: Mapping[int, int]) -> int:
    answer = 1
    for leg in legs:
        answer *= size_dict[leg]
    return answer


def evaluate_ssa_path(
    inputs: Sequence[Sequence[int]],
    output: Sequence[int],
    size_dict: Mapping[int, int],
    ssa_path: Sequence[Sequence[int]],
) -> dict[str, Any]:
    """Validate and score one complete SSA path using exact Python integers."""

    n = len(inputs)
    require(n >= 1 and len(ssa_path) == n - 1, "SSA path length is not n_tensors - 1")
    output_set = set(output)
    appearances: Counter[int] = Counter()
    for term in inputs:
        appearances.update(term)
    appearances.update(output)

    slots: list[tuple[int, ...] | None] = []
    contraction_sizes: list[int] = []
    raw_sizes: list[int] = []
    refcount: Counter[int] = Counter()
    for term in inputs:
        counts = Counter(term)
        require(max(counts.values(), default=0) == 1, "a tensor repeats an index")
        unique = tuple(sorted(counts))
        slots.append(unique)
        refcount.update(unique)
        logical = [leg for leg, count in counts.items() if count != appearances[leg]]
        contraction_sizes.append(product_dimensions(logical, size_dict))
        raw_sizes.append(product_dimensions(term, size_dict))

    alive_size = sum(raw_sizes)
    peak_size = alive_size
    flops = 0
    writes = 0
    max_size = product_dimensions(output, size_dict) if n == 1 else 1
    max_contraction_size = 1

    for step, pair in enumerate(ssa_path):
        require(len(pair) == 2, f"SSA step {step} is not binary")
        left_id, right_id = int(pair[0]), int(pair[1])
        require(left_id != right_id, f"SSA step {step} self-contracts")
        require(0 <= left_id < len(slots) and 0 <= right_id < len(slots), f"SSA step {step} is invalid")
        left, right = slots[left_id], slots[right_id]
        require(left is not None and right is not None, f"SSA step {step} reuses a tensor")
        slots[left_id] = None
        slots[right_id] = None
        union = tuple(sorted(set(left).union(right)))
        flops += product_dimensions(union, size_dict)
        for leg in left:
            refcount[leg] -= 1
        for leg in right:
            refcount[leg] -= 1
        result = tuple(leg for leg in union if refcount[leg] > 0 or leg in output_set)
        refcount.update(result)
        result_size = product_dimensions(result, size_dict)
        writes += result_size
        max_size = max(max_size, result_size)
        max_contraction_size = max(
            max_contraction_size,
            contraction_sizes[left_id] + contraction_sizes[right_id] + result_size,
        )
        peak_size = max(peak_size, alive_size + result_size)
        alive_size += result_size - raw_sizes[left_id] - raw_sizes[right_id]
        slots.append(result)
        contraction_sizes.append(result_size)
        raw_sizes.append(result_size)

    alive = [value for value in slots if value is not None]
    require(len(alive) == 1, "SSA path does not leave one tensor")
    require(tuple(sorted(alive[0])) == tuple(sorted(output)), "SSA final indices differ from output")
    return {
        "exact_flops": flops,
        "exact_total_write": writes,
        "exact_combo64": flops + 64 * writes,
        "exact_max_size": max_size,
        "exact_max_contraction_size": max_contraction_size,
        "exact_peak_size": peak_size,
        "log10_flops": "-inf" if flops == 0 else math.log10(flops),
        "log2_max_size": math.log2(max_size),
        "log2_max_contraction_size": math.log2(max_contraction_size),
        "log2_peak_size": math.log2(peak_size),
    }


def exact_sliced_metrics(
    spec: Mapping[str, Any],
    path: list[list[int]],
    sliced_indices: Sequence[int],
    target_size: int,
    *,
    outer_slicing_allowed: bool,
) -> dict[str, Any]:
    """Replay one sliced plan using the common exact integer scorer."""

    legs = sorted(int(leg) for leg in sliced_indices)
    require(len(legs) == len(set(legs)), "duplicate sliced index")
    output_set = set(spec["output"])
    inner = sorted(set(legs).difference(output_set))
    outer = sorted(set(legs).intersection(output_set))
    if outer:
        require(outer_slicing_allowed, "method returned a forbidden output slice")
    sliced_sizes = dict(spec["size_dict"])
    sliced_dimensions: dict[str, int] = {}
    n_slices = 1
    for leg in legs:
        require(leg in sliced_sizes, "unknown sliced index")
        dimension = int(sliced_sizes[leg])
        sliced_dimensions[str(leg)] = dimension
        n_slices *= dimension
        sliced_sizes[leg] = 1
    unsliced_same_path = evaluate_ssa_path(
        spec["inputs"], spec["output"], spec["size_dict"], path
    )
    per_slice = evaluate_ssa_path(spec["inputs"], spec["output"], sliced_sizes, path)
    full_output_elements = product_dimensions(spec["output"], spec["size_dict"])
    output_block_elements = product_dimensions(spec["output"], sliced_sizes)
    output_blocks = product_dimensions(outer, spec["size_dict"])
    inner_slices = product_dimensions(inner, spec["size_dict"])
    require(output_blocks * output_block_elements == full_output_elements,
            "output blocks do not cover the full output")
    require(output_blocks * inner_slices == n_slices, "slice task count drifted")
    require(int(per_slice["exact_max_size"]) <= target_size, "sliced path violates target")
    total_flops = int(per_slice["exact_flops"]) * n_slices
    total_write = int(per_slice["exact_total_write"]) * n_slices
    base_flops = int(unsliced_same_path["exact_flops"])
    return {
        "sliced_indices": legs,
        "sliced_index_dimensions": sliced_dimensions,
        "sliced_inner_indices": inner,
        "sliced_outer_indices": outer,
        "sliced_index_count": len(legs),
        "sliced_inner_index_count": len(inner),
        "sliced_outer_index_count": len(outer),
        "output_block_count": output_blocks,
        "inner_slices_per_output_block": inner_slices,
        "full_output_elements": full_output_elements,
        "output_block_elements": output_block_elements,
        "n_slices": n_slices,
        "n_slices_decimal": str(n_slices),
        "log2_n_slices": math.log2(n_slices),
        "exact_per_slice_flops": int(per_slice["exact_flops"]),
        "exact_per_slice_flops_decimal": str(int(per_slice["exact_flops"])),
        "exact_total_sliced_flops": total_flops,
        "exact_total_sliced_flops_decimal": str(total_flops),
        "exact_per_slice_total_write": int(per_slice["exact_total_write"]),
        "exact_total_sliced_write": total_write,
        "exact_sliced_combo64": total_flops + 64 * total_write,
        "exact_per_slice_max_size": int(per_slice["exact_max_size"]),
        "exact_per_slice_max_contraction_size": int(per_slice["exact_max_contraction_size"]),
        "exact_per_slice_peak_size": int(per_slice["exact_peak_size"]),
        "same_path_unsliced_exact_flops": base_flops,
        "same_path_slicing_flops_ratio": (total_flops / base_flops if base_flops else None),
        "target_size": target_size,
        "target_satisfied": True,
        "outer_slicing_allowed_for_method": outer_slicing_allowed,
        "output_block_assembly_included": False,
        "gather_additions_included": False,
    }


def symbolic_network(
    spec: Mapping[str, Any],
) -> tuple[tuple[str, ...], str, dict[str, int], dict[str, int]]:
    import opt_einsum as oe

    labels = sorted(spec["size_dict"])
    symbols = {label: oe.get_symbol(position) for position, label in enumerate(labels)}
    reverse = {symbol: label for label, symbol in symbols.items()}
    inputs = tuple("".join(symbols[label] for label in term) for term in spec["inputs"])
    output = "".join(symbols[label] for label in spec["output"])
    sizes = {symbols[label]: int(spec["size_dict"][label]) for label in labels}
    return inputs, output, sizes, reverse


def shared_heavy_path_path(output: Path, spec: Mapping[str, Any], repeat: int) -> Path:
    run_directory = output.parent.parent
    return run_directory / "shared-heavy-paths" / f"{spec['name']}--r{repeat:03d}.json"


def load_shared_heavy_path(
    path: Path, spec: Mapping[str, Any], repeat: int
) -> dict[str, Any]:
    require(path.is_file(), f"paired Heavy path is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(value.get("schema") == "qc32-open-output-shared-heavy-path", "shared path schema drifted")
    require(value.get("status") == "success", "shared Heavy path is not successful")
    require(value.get("instance") == spec["name"], "shared Heavy instance drifted")
    require(value.get("repeat_ordinal") == repeat, "shared Heavy repeat drifted")
    require(value.get("method") == "arctn-heavy-shared-path", "shared Heavy method drifted")
    planner = dict(value["planner"])
    require(planner.get("preset") == "heavy", "shared path is not Heavy")
    require(planner.get("objective") == "total_flops", "shared Heavy objective drifted")
    path_value = [[int(left), int(right)] for left, right in value["ssa_path"]]
    require(
        value.get("linear_path") == ssa_to_linear(path_value, len(spec["inputs"])),
        "shared Heavy linear path drifted",
    )
    evaluate_ssa_path(spec["inputs"], spec["output"], spec["size_dict"], path_value)
    return value


def wait_for_shared_heavy_path(
    path: Path,
    spec: Mapping[str, Any],
    repeat: int,
    output: Path,
    maximum_wait_seconds: float,
) -> dict[str, Any]:
    run_directory = output.parent.parent
    producer_cell = (
        f"{spec['name']}--r{repeat:03d}--arctn-heavy-fixed-tree-post-slicing"
    )
    producer_terminal = run_directory / "terminal" / f"{producer_cell}.json"
    deadline = time.monotonic() + maximum_wait_seconds
    while not path.is_file():
        if producer_terminal.is_file():
            terminal = json.loads(producer_terminal.read_text(encoding="utf-8"))
            if terminal.get("status") != "success":
                raise RuntimeError(
                    "paired fixed-tree producer ended without a usable Heavy path: "
                    f"{terminal.get('status')}"
                )
        require(
            time.monotonic() < deadline,
            "timed out waiting for the paired Heavy path producer",
        )
        time.sleep(0.25)
    return load_shared_heavy_path(path, spec, repeat)


def create_shared_heavy_path(
    path: Path,
    spec: Mapping[str, Any],
    repeat: int,
    cooperative_seconds: float,
) -> dict[str, Any]:
    import arctn

    started_cpu = time.process_time()
    started = time.perf_counter()
    report = dict(
        arctn.arctn_schedule(
            [list(term) for term in spec["inputs"]],
            list(spec["output"]),
            spec["size_dict"],
            preset="heavy",
            seed=repeat,
            max_time=cooperative_seconds,
            use_ssa=True,
            flops_weight=1.0,
            read_write_weight=0.0,
            rate_enabled=True,
        )
    )
    api_wall = time.perf_counter() - started
    api_cpu = time.process_time() - started_cpu
    require(report.get("planner_objective") == "total_flops", "ArcTN objective drifted")
    require(float(report.get("flops_weight")) == 1.0, "ArcTN FLOPs weight drifted")
    require(float(report.get("read_write_weight")) == 0.0, "ArcTN read/write weight drifted")
    require(report.get("sliced") is False, "shared Heavy search unexpectedly sliced the path")
    require(report.get("target_size") is None, "shared Heavy search received a target")
    resolved = dict(report["resolved_budget"])
    require(int(resolved["trials"]) == 192, "Heavy trial cap drifted")
    require(float(resolved["secs"]) == cooperative_seconds, "Heavy cooperative max-time drifted")
    require(float(resolved["rate"]) == 1.0e8, "Heavy rate drifted")
    schedule = dict(report["schedule"])
    config = dict(schedule["config"])
    require(config.get("record_trial_paths") is False, "Heavy trial paths were unexpectedly enabled")
    path_value = [[int(left), int(right)] for left, right in report["path"]]
    exact = evaluate_ssa_path(spec["inputs"], spec["output"], spec["size_dict"], path_value)
    artifact = {
        "schema": "qc32-open-output-shared-heavy-path",
        "status": "success",
        "instance": spec["name"],
        "instance_ordinal": spec["ordinal"],
        "repeat_ordinal": repeat,
        "method": "arctn-heavy-shared-path",
        "planner": {
            "entrypoint": "arctn.arctn_schedule",
            "preset": "heavy",
            "seed": repeat,
            "rate_enabled": True,
            "objective": "total_flops",
            "max_time_argument_seconds": cooperative_seconds,
            "path_search_wall_seconds": api_wall,
            "path_search_process_cpu_seconds": api_cpu,
            "resolved_budget": json_safe(resolved),
            "schedule_summary": {
                "total_trial_cap": schedule.get("total_trial_cap"),
                "trial_count": len(schedule.get("trials", ())),
                "rate_reached": schedule.get("rate_reached"),
                "rate_caused_stop": schedule.get("rate_caused_stop"),
                "deadline_reached": schedule.get("deadline_reached"),
                "deadline_caused_stop": schedule.get("deadline_caused_stop"),
                "stopped_reason": schedule.get("stopped_reason"),
                "timing": json_safe(schedule.get("timing")),
                "record_trial_paths": config.get("record_trial_paths"),
            },
            "build_info": json_safe(dict(arctn.arctn_py.build_info())),
            "package_version": importlib.metadata.version("arctn"),
        },
        "ssa_path": path_value,
        "linear_path": ssa_to_linear(path_value, len(spec["inputs"])),
        "unsliced_exact_metrics": exact,
    }
    atomic_write_json(path, artifact)
    return load_shared_heavy_path(path, spec, repeat)


def run_arctn_harness(
    args: argparse.Namespace,
    method: Mapping[str, Any],
    shared_path_file: Path,
    shared_path: Mapping[str, Any],
    target_size: int,
) -> tuple[list[list[int]], list[int], dict[str, Any]]:
    mode = str(method["slicing_mode"])
    harness = (
        args.protocol.parent
        / "arctn-slicing-harness"
        / "target"
        / "release"
        / "arctn-slicing-harness"
    )
    require(harness.is_file(), f"ArcTN slicing harness is missing: {harness}")
    temporary_root = args.output.parent.parent / "harness-temporary"
    temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temporary_root) as directory:
        harness_output = Path(directory) / "result.json"
        completed = subprocess.run(
            [
                str(harness),
                "--mode",
                mode,
                "--network",
                str(args.instance),
                "--heavy-path",
                str(shared_path_file),
                "--target-size",
                str(target_size),
                "--output",
                str(harness_output),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        require(
            completed.returncode == 0,
            f"ArcTN slicing harness failed ({completed.returncode}): {completed.stderr[-2000:]}",
        )
        native = json.loads(harness_output.read_text(encoding="utf-8"))
    require(native.get("schema") == "arctn-open-output-slicing-harness-v1", "harness schema drifted")
    require(native.get("status") == "success", "harness result is not successful")
    require(native.get("mode") == mode, "harness mode drifted")
    require(native.get("method") == method["id"], "harness method drifted")
    require(native.get("target_size") == target_size, "harness target drifted")
    shared_ssa = [[int(a), int(b)] for a, b in shared_path["ssa_path"]]
    input_ssa = [[int(a), int(b)] for a, b in native["input_heavy_ssa_path"]]
    require(input_ssa == shared_ssa, "harness did not consume the exact shared Heavy path")
    path_value = [[int(a), int(b)] for a, b in native["final_ssa_path"]]
    if mode == "fixed":
        require(path_value == shared_ssa, "fixed-tree slicing changed the Heavy SSA path")
    objective = dict(native["objective"])
    require(objective.get("name") == "total_flops", "harness objective drifted")
    require(float(objective.get("flops_weight")) == 1.0, "harness FLOPs weight drifted")
    require(float(objective.get("read_write_weight")) == 0.0, "harness RW weight drifted")
    validation = dict(native["validation"])
    require(validation.get("target_satisfied") is True, "harness target validation failed")
    require(native["parameters"].get("allow_outer") is True, "harness disallowed output slicing")
    sliced = [int(leg) for leg in native["sliced_legs"]]
    heavy_planner = dict(shared_path["planner"])
    slice_wall = float(native["stage_timing"]["wall_seconds"])
    path_wall = float(heavy_planner["path_search_wall_seconds"])
    return path_value, sliced, {
        "entrypoint": method["entrypoint"],
        "mechanism": method["mechanism"],
        "preset": "heavy",
        "seed": int(shared_path["repeat_ordinal"]),
        "objective": "total_flops",
        "shared_heavy_path_file": str(shared_path_file),
        "shared_heavy_path_producer_method": "arctn-heavy-fixed-tree-post-slicing",
        "shared_heavy_path_exactly_consumed": True,
        "heavy_path_search": json_safe(heavy_planner),
        "slicing_stage": json_safe(native),
        "logical_planning_wall_seconds": path_wall + slice_wall,
        "timing_accounting": (
            "The Heavy path search was physically executed once and is shared by both ArcTN "
            "rows. Logical planning wall adds that same saved path-search wall to this row's "
            "independently measured slicing-stage wall."
        ),
    }


def run_arctn(
    spec: Mapping[str, Any],
    method: Mapping[str, Any],
    args: argparse.Namespace,
    target_size: int,
    cooperative_seconds: float,
) -> tuple[list[list[int]], list[int], dict[str, Any]]:
    shared_path_file = shared_heavy_path_path(args.output, spec, args.repeat)
    generated_now = False
    if method["shared_heavy_path_role"] == "producer":
        if shared_path_file.exists():
            shared = load_shared_heavy_path(shared_path_file, spec, args.repeat)
        else:
            shared = create_shared_heavy_path(
                shared_path_file, spec, args.repeat, cooperative_seconds
            )
            generated_now = True
    else:
        require(method["shared_heavy_path_role"] == "consumer", "unknown shared path role")
        shared = wait_for_shared_heavy_path(
            shared_path_file,
            spec,
            args.repeat,
            args.output,
            cooperative_seconds,
        )
    path_value, sliced, planner = run_arctn_harness(
        args, method, shared_path_file, shared, target_size
    )
    planner["heavy_path_physically_executed_in_this_worker"] = generated_now
    return path_value, sliced, planner


def run_cotengra(
    spec: Mapping[str, Any],
    method: Mapping[str, Any],
    repeat: int,
    target_size: int,
    cooperative_seconds: float,
) -> tuple[list[list[int]], list[int], dict[str, Any]]:
    import cotengra as ctg

    inputs, output, sizes, reverse = symbolic_network(spec)
    started_cpu = time.process_time()
    started = time.perf_counter()
    require(method["id"] == "cotengra-autohq-post-search-slicing", "unknown Cotengra method")
    constructor_args = {
        "max_repeats": int(method["max_repeats"]),
        "max_time": cooperative_seconds,
        "optlib": str(method["optlib"]),
        "parallel": int(method["parallel_workers"]),
        "progbar": False,
        "minimize": "flops",
    }
    optimizer = ctg.AutoHQOptimizer(**constructor_args)
    require(type(optimizer).__name__ == "AutoHQOptimizer", "AutoHQ resolved class drifted")
    require(optimizer.minimize == "flops", "AutoHQ objective drifted")
    resolved_kwargs = dict(optimizer.kwargs)
    require(int(resolved_kwargs["max_repeats"]) == 128, "AutoHQ max_repeats drifted")
    require(float(resolved_kwargs["max_time"]) == cooperative_seconds, "AutoHQ max-time drifted")
    require(int(resolved_kwargs["parallel"]) == 8, "AutoHQ worker count drifted")
    require(str(resolved_kwargs["optlib"]) == "optuna", "AutoHQ optlib drifted")
    resolved_configuration = {
        "constructor_arguments": constructor_args,
        "auto_hq_resolved_kwargs": resolved_kwargs,
    }
    search_started = time.perf_counter()
    unsliced_tree = optimizer.search(inputs, output, sizes)
    search_wall = time.perf_counter() - search_started
    unsliced_ssa = [
        [int(left), int(right)] for left, right in unsliced_tree.get_ssa_path()
    ]
    slice_started = time.perf_counter()
    tree = unsliced_tree.slice(
        target_size=target_size,
        minimize="flops",
        allow_outer=bool(method["allow_outer"]),
    )
    slicing_wall = time.perf_counter() - slice_started
    path = [[int(left), int(right)] for left, right in tree.get_ssa_path()]
    require(path == unsliced_ssa, "Cotengra post-search slicing changed the tree")
    stage_summary = {
        "search_wall_seconds": search_wall,
        "post_search_fixed_tree_slicing_wall_seconds": slicing_wall,
        "post_search_slicing_changed_ssa_path": False,
        "allow_outer": bool(method["allow_outer"]),
    }
    api_wall = time.perf_counter() - started
    api_cpu = time.process_time() - started_cpu
    sliced_symbols = sorted(str(symbol) for symbol in tree.sliced_inds)
    require(all(symbol in reverse for symbol in sliced_symbols), "Cotengra returned an unknown index")
    sliced = [int(reverse[symbol]) for symbol in sliced_symbols]
    return path, sliced, {
        "entrypoint": method["entrypoint"],
        "mechanism": method["mechanism"],
        "repeat_ordinal": repeat,
        "public_seed": None,
        "objective": "flops",
        "planner_api_wall_seconds": api_wall,
        "planner_api_parent_process_cpu_seconds": api_cpu,
        "stages": stage_summary,
        "resolved_configuration": json_safe(resolved_configuration),
        "native_tree_summary": {
            "sliced_symbols": sliced_symbols,
            "reported_max_size": json_safe(tree.max_size()),
            "reported_contraction_cost": json_safe(tree.contraction_cost()),
            "reported_total_flops": json_safe(tree.total_flops()),
            "reported_n_slices": json_safe(tree.nslices),
        },
        "package_version": importlib.metadata.version("cotengra"),
        "optuna_version": importlib.metadata.version("optuna"),
        "kahypar_version": importlib.metadata.version("kahypar"),
        "repeat_boundary": "unseeded repetition; not a matched cross-library seed",
    }


def omeco_tree_to_ssa(tree: Mapping[str, Any], n_inputs: int) -> list[list[int]]:
    leaves: list[int] = []

    def collect(node: Mapping[str, Any]) -> None:
        children = node.get("args")
        if children:
            for child in children:
                collect(child)
            return
        key = "tensor_index" if "tensor_index" in node else "tensorindex"
        require(key in node, "OMECO leaf has no tensor index")
        leaves.append(int(node[key]))

    collect(tree)
    observed = sorted(leaves)
    if observed == list(range(n_inputs)):
        offset = 0
    elif observed == list(range(1, n_inputs + 1)):
        offset = 1
    else:
        raise RuntimeError("OMECO leaves are not a permutation of input tensors")

    ssa: list[list[int]] = []
    next_id = n_inputs

    def visit(node: Mapping[str, Any]) -> int:
        nonlocal next_id
        children = node.get("args")
        if not children:
            key = "tensor_index" if "tensor_index" in node else "tensorindex"
            return int(node[key]) - offset
        require(len(children) == 2, "OMECO tree is not binary")
        left = visit(children[0])
        right = visit(children[1])
        require(left != right, "OMECO tree reuses one operand")
        ssa.append([left, right])
        result = next_id
        next_id += 1
        return result

    root = visit(tree)
    require(root == 2 * n_inputs - 2, "OMECO root SSA id drifted")
    require(len(ssa) == n_inputs - 1, "OMECO path length drifted")
    return ssa


def run_omeco(
    spec: Mapping[str, Any],
    method: Mapping[str, Any],
    repeat: int,
    target_size: int,
) -> tuple[list[list[int]], list[int], dict[str, Any]]:
    import omeco

    initial_cfg = dict(method["initial_tree"])
    slicer_cfg = dict(method["slicer"])
    initial_score_cfg = dict(initial_cfg["score"])
    initial_score = omeco.ScoreFunction(**initial_score_cfg)
    optimizer = omeco.TreeSA(
        ntrials=int(initial_cfg["ntrials"]),
        niters=int(initial_cfg["niters"]),
        score=initial_score,
    )
    slicer_score = omeco.ScoreFunction(
        tc_weight=1.0,
        sc_weight=1.0,
        rw_weight=0.0,
        sc_target=math.log2(target_size),
    )
    slicer = omeco.TreeSASlicer(
        ntrials=int(slicer_cfg["ntrials"]),
        niters=int(slicer_cfg["niters"]),
        optimization_ratio=float(slicer_cfg["optimization_ratio"]),
        score=slicer_score,
    )
    require(int(optimizer.ntrials) == 8, "OMECO ntrials drifted")
    require(int(optimizer.niters) == 50, "OMECO niters drifted")
    require(len(optimizer.betas) == 300, "OMECO beta schedule length drifted")
    require(int(slicer.ntrials) == 8, "OMECO slicer ntrials drifted")
    require(int(slicer.niters) == 10, "OMECO slicer niters drifted")
    require(abs(float(slicer.optimization_ratio) - 2.0) <= 1e-12, "OMECO ratio drifted")
    require(float(slicer.score.rw_weight) == 0.0, "OMECO slicer objective drifted")
    inputs = [list(term) for term in spec["inputs"]]
    output = list(spec["output"])
    sizes = dict(spec["size_dict"])
    started_cpu = time.process_time()
    started = time.perf_counter()
    initial_started = time.perf_counter()
    initial_tree = omeco.optimize_code(inputs, output, sizes, optimizer)
    initial_wall = time.perf_counter() - initial_started
    require(initial_tree.is_binary(), "OMECO returned a non-binary initial tree")
    slicer_started = time.perf_counter()
    sliced_result = omeco.slice_code(initial_tree, inputs, sizes, slicer)
    slicer_wall = time.perf_counter() - slicer_started
    api_wall = time.perf_counter() - started
    api_cpu = time.process_time() - started_cpu
    final_tree = sliced_result.tree()
    require(final_tree.is_binary(), "OMECO returned a non-binary sliced tree")
    require(final_tree.leaf_count() == len(inputs), "OMECO leaf count drifted")
    path = omeco_tree_to_ssa(final_tree.to_dict(), len(inputs))
    sliced = [int(leg) for leg in sliced_result.slicing()]
    return path, sliced, {
        "entrypoint": "omeco.optimize_code followed by omeco.slice_code",
        "repeat_ordinal": repeat,
        "public_seed": None,
        "objective": "FLOPs-directed target-aware native score",
        "allow_outer": bool(method["allow_outer"]),
        "planner_api_wall_seconds": api_wall,
        "planner_api_process_cpu_seconds": api_cpu,
        "initial_tree_wall_seconds": initial_wall,
        "slicer_wall_seconds": slicer_wall,
        "parameters": {
            "initial_tree": {
                "ntrials": int(optimizer.ntrials),
                "niters": int(optimizer.niters),
                "beta_count": len(optimizer.betas),
                "score": initial_score_cfg,
            },
            "slicer": {
                "ntrials": int(slicer.ntrials),
                "niters": int(slicer.niters),
                "beta_count": len(slicer.betas),
                "optimization_ratio": float(slicer.optimization_ratio),
                "score": {
                    "tc_weight": float(slicer.score.tc_weight),
                    "sc_weight": float(slicer.score.sc_weight),
                    "rw_weight": float(slicer.score.rw_weight),
                    "sc_target": float(slicer.score.sc_target),
                },
            },
        },
        "package_version": importlib.metadata.version("omeco"),
        "repeat_boundary": (
            "deterministic repeated execution for runtime variability only; quality is counted "
            "once per circuit because OMECO 0.2.6 uses fixed internal trial seeds"
        ),
    }


def run(args: argparse.Namespace) -> None:
    protocol = load_protocol(args.protocol)
    methods = {row["id"]: row for row in protocol["methods"]}
    require(args.method in methods, "method is not in the frozen protocol")
    require(args.repeat in protocol["repeat_ordinals"], "repeat ordinal is not frozen")
    expected_affinity = int(protocol["resources"]["physical_cores_per_cell"])
    affinity = sorted(os.sched_getaffinity(0))
    require(len(affinity) == expected_affinity, "worker is not pinned to exactly eight CPUs")
    spec = load_spec(args.instance)
    target = load_target(args.protocol, protocol, spec)
    target_size = int(target["effective_target_elements"])
    cooperative_seconds = float(
        protocol["resources"]["cooperative_planner_seconds_for_supported_methods"]
    )
    method = methods[args.method]
    full_started = time.perf_counter()
    if args.method.startswith("arctn-"):
        path, sliced, planner = run_arctn(
            spec, method, args, target_size, cooperative_seconds
        )
    elif args.method.startswith("cotengra-"):
        path, sliced, planner = run_cotengra(
            spec, method, args.repeat, target_size, cooperative_seconds
        )
    else:
        require(args.method == "omeco-treesa8-treesaslicer8", "unknown method")
        path, sliced, planner = run_omeco(spec, method, args.repeat, target_size)
    replay_started = time.perf_counter()
    metrics = exact_sliced_metrics(
        spec,
        path,
        sliced,
        target_size,
        outer_slicing_allowed=bool(method["allow_outer"]),
    )
    require(metrics["n_slices"] > 1 and metrics["sliced_outer_indices"],
            "supplement requires an actual sliced output index")
    require(metrics["output_block_count"] >= target["minimum_output_blocks"],
            "too few output blocks for the frozen target")
    linear = ssa_to_linear(path, len(spec["inputs"]))
    replay_wall = time.perf_counter() - replay_started
    usage = resource.getrusage(resource.RUSAGE_SELF)
    artifact = {
        "schema": "qc32-open-output-slicing-cell",
        "status": "success",
        "cell_id": args.cell_id,
        "instance": spec["name"],
        "instance_ordinal": spec["ordinal"],
        "method": args.method,
        "repeat_ordinal": args.repeat,
        "objective": "total_sliced_flops",
        "target": target,
        "thread_contract": {
            "physical_cpu_affinity": affinity,
            "physical_cores_per_cell": expected_affinity,
            "RAYON_NUM_THREADS": os.environ.get("RAYON_NUM_THREADS"),
            "COTENGRA_NUM_WORKERS": os.environ.get("COTENGRA_NUM_WORKERS"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "planner": planner,
        "ssa_path": path,
        "linear_path": linear,
        "sliced_indices": metrics["sliced_indices"],
        "slicing_metrics": metrics,
        "exact_replay_wall_seconds_before_write": replay_wall,
        "worker_full_wall_seconds_before_write": time.perf_counter() - full_started,
        "resource_self": {
            "user_cpu_seconds": float(usage.ru_utime),
            "system_cpu_seconds": float(usage.ru_stime),
            "max_rss_linux_kib": int(usage.ru_maxrss),
        },
    }
    atomic_write_json(args.output, artifact)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--protocol", type=Path, required=True)
    value.add_argument("--instance", type=Path, required=True)
    value.add_argument("--method", choices=METHODS, required=True)
    value.add_argument("--repeat", type=int, required=True)
    value.add_argument("--cell-id", required=True)
    value.add_argument("--output", type=Path, required=True)
    return value


def main() -> int:
    args = parser().parse_args()
    try:
        run(args)
    except Exception as exc:
        print(f"worker failed: {type(exc).__name__}: {exc}", file=__import__("sys").stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
