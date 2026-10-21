#!/usr/bin/env python3
"""Run one planner on one network and save the returned order with its costs.

run_benchmark.py starts one worker per run, pinned to eight physical cores.
Every planner setting comes from the protocol file. Planning time is the wall
time of the planner API call only; reading the network and scoring the order
are not included. The order is scored with scoring.py on the same network.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

from scoring import evaluate_ssa_path, load_network, require

SCHEMA = "arctn-benchmarks-run-v1"
if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)


def json_safe(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if hasattr(value, "item"):
        return json_safe(value.item())
    return repr(value)


def save_new(path, value):
    """Write the result once; a second worker for the same run cannot replace it."""
    path = Path(path)
    require(not path.exists(), f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, allow_nan=False, sort_keys=True)
            handle.write("\n")
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def package_version(package, expected, strict):
    actual = importlib.metadata.version(package)
    if expected is not None and actual != expected:
        message = f"{package} {actual} is installed; the protocol lists {expected}"
        require(not strict, message)
        print(f"warning: {message}", file=sys.stderr)
    return actual


def einsum_symbol(index):
    if index < 52:
        return "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"[index]
    codepoint = index + 140
    if codepoint >= 0xD800:
        codepoint += 0x800
    require(codepoint <= 0x10FFFF, "too many indices for single-character symbols")
    return chr(codepoint)


def symbolic_network(spec):
    symbols = {label: einsum_symbol(i) for i, label in enumerate(sorted(spec["size_dict"]))}
    return (tuple("".join(symbols[x] for x in term) for term in spec["inputs"]),
            "".join(symbols[x] for x in spec["output"]),
            {symbols[x]: size for x, size in spec["size_dict"].items()})


def run_arctn(spec, method, seed):
    import arctn

    kwargs = dict(preset=method["preset"], seed=seed, target_size=None, max_time=float(method["max_time"]),
                  use_ssa=True, flops_weight=1.0, read_write_weight=0.0, rate_enabled=bool(method["rate_enabled"]))
    started = time.perf_counter()
    report = dict(arctn.arctn_schedule(spec["inputs"], spec["output"], spec["size_dict"], **kwargs))
    seconds = time.perf_counter() - started
    require(report.get("path_format") == "ssa-v1", "ArcTN returned a path that is not in SSA form")
    path = [[int(a), int(b)] for a, b in report["path"]]
    return path, {"call": "arctn.arctn_schedule", "parameters": kwargs, "planning_seconds": seconds}


def run_cotengra(spec, method, seed):
    import cotengra as ctg
    from cotengra.presets import estimate_optimal_hardness

    name, kwargs = method["optimizer"], dict(method["kwargs"])
    require(name in {"AutoOptimizer", "AutoHQOptimizer", "HyperOptimizer"}, "unknown Cotengra optimizer")
    optimizer = getattr(ctg, name)(**kwargs)
    inputs, output, sizes = symbolic_network(spec)
    details = {"call": f"cotengra.{name}.search", "parameters": kwargs,
               "seed": "none; Cotengra takes no seed here, the run number is only a label"}
    if name != "HyperOptimizer":
        # The Auto presets switch to an exact search below this hardness.
        hardness, cutoff = int(estimate_optimal_hardness(inputs)), int(optimizer.optimal_cutoff)
        details["dispatch"] = "exact search" if hardness < cutoff else "HyperOptimizer"
    started = time.perf_counter()
    tree = optimizer.search(inputs, output, sizes)
    details["planning_seconds"] = time.perf_counter() - started
    return [[int(a), int(b)] for a, b in tree.get_ssa_path()], details


def omeco_tree_to_ssa(tree, n_inputs):
    """Read the binary contractions of an OMECO tree in post-order; unary nodes are skipped."""
    leaves, postorder, pending = [], [], [(tree, False)]
    while pending:
        node, expanded = pending.pop()
        children = node.get("args")
        if expanded:
            postorder.append(node)
        elif children:
            require(len(children) in (1, 2), "OMECO node is neither unary nor binary")
            pending.append((node, True))
            pending.extend((child, False) for child in reversed(children))
        else:
            key = "tensor_index" if "tensor_index" in node else "tensorindex"
            leaves.append(int(node[key]))
            postorder.append(node)
    observed = sorted(leaves)
    offset = 0 if observed == list(range(n_inputs)) else 1
    require(observed == list(range(offset, n_inputs + offset)), "OMECO leaves are not the input tensors")
    values, path = [], []
    for node in postorder:
        children = node.get("args")
        if not children:
            key = "tensor_index" if "tensor_index" in node else "tensorindex"
            values.append(int(node[key]) - offset)
        elif len(children) == 2:
            right, left = values.pop(), values.pop()
            path.append([left, right])
            values.append(n_inputs + len(path) - 1)
    require(len(path) == n_inputs - 1 and len(values) == 1, "OMECO order is incomplete")
    return path


def run_omeco(spec, method, seed):
    import omeco

    ntrials, niters = int(method["ntrials"]), int(method["niters"])
    score = omeco.ScoreFunction(tc_weight=1.0, sc_weight=0.0, rw_weight=0.0, sc_target=0.0)
    optimizer = omeco.TreeSA(ntrials=ntrials, niters=niters, score=score)
    # to_dict() recurses over the tree.
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 4 * len(spec["inputs"]) + 10000))
    started = time.perf_counter()
    tree = omeco.optimize_code(spec["inputs"], spec["output"], spec["size_dict"], optimizer)
    seconds = time.perf_counter() - started
    path = omeco_tree_to_ssa(tree.to_dict(), len(spec["inputs"]))
    return path, {"call": "omeco.optimize_code with TreeSA", "parameters": {"ntrials": ntrials, "niters": niters},
                  "seed": "none; OMECO takes no seed here, the run number is only a label", "planning_seconds": seconds}


def run_tnco(spec, method, seed):
    from tnco_adapter import run_tnco as adapter
    return adapter(spec, method, seed)


PLANNERS = {"ArcTN": ("arctn", run_arctn), "Cotengra": ("cotengra", run_cotengra),
            "OMECO": ("omeco", run_omeco), "TNCO": ("tnco", run_tnco)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--network", type=Path, required=True)
    parser.add_argument("--configuration", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--cell-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strict-versions", action="store_true",
                        help="stop if an installed package version differs from the protocol")
    args = parser.parse_args()

    started = time.perf_counter()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    methods = {m["id"]: m for m in protocol["configurations"]}
    require(args.configuration in methods, f"unknown configuration {args.configuration}")
    method = methods[args.configuration]
    spec = load_network(args.network)
    if hasattr(os, "sched_getaffinity"):
        affinity = sorted(os.sched_getaffinity(0))
        require(len(affinity) == protocol["resources"]["physical_cores_per_run"],
                "the worker must be pinned to the protocol's number of physical cores")
    else:
        affinity = None
    package, planner = PLANNERS[method["library"]]
    version = package_version(package, protocol["versions"].get(package), args.strict_versions)
    if len(spec["inputs"]) == 1:
        path, details = [], {"planning_seconds": 0.0, "note": "one tensor; no search"}
    else:
        path, details = planner(spec, method, args.seed)
    metrics = evaluate_ssa_path(spec["inputs"], spec["output"], spec["size_dict"], path)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    save_new(args.output, json_safe({
        "schema": SCHEMA, "status": "success", "cell_id": args.cell_id, "network": spec["name"],
        "configuration": args.configuration, "seed": args.seed, "library": method["library"],
        "package_version": version, "planning_seconds": details["planning_seconds"], "planner": details,
        "ssa_path": path, "metrics": metrics, "cpu_affinity": affinity,
        "threads": {key: os.environ.get(key) for key in ("RAYON_NUM_THREADS", "COTENGRA_NUM_WORKERS",
                    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "worker_wall_seconds": time.perf_counter() - started,
        "max_rss_kib": int(usage.ru_maxrss),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
