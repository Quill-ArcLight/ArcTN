#!/usr/bin/env python3
"""Contract one network with NumPy along a saved order, once, and time it.

The same executor is used for every planner: opt_einsum.contract_expression
with the saved order as a fixed path, then one call with the NumPy backend and
no warm-up. Before reading the arrays, the order is replayed to check that its
tensors fit the protocol's memory limits; runs that do not fit are recorded as
resource-skip. The result is checked by shape and by its sum against the
reference value stored with the network.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import time

import numpy as np
import opt_einsum as oe

from scoring import evaluate_ssa_path, load_network, require, ssa_to_linear
from worker import json_safe, save_new

SCHEMA = "arctn-benchmarks-execution-v1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--network", type=Path, required=True)
    parser.add_argument("--configuration", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--cell-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--orders", type=Path, required=True)
    parser.add_argument("--arrays", type=Path, required=True)
    parser.add_argument("--strict-versions", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    raw = json.loads(args.network.read_text(encoding="utf-8"))
    spec = load_network(args.network)
    record = {"schema": SCHEMA, "cell_id": args.cell_id, "network": spec["name"], "configuration": args.configuration,
              "seed": args.seed, "backend": "numpy", "numeric_threads": os.environ.get("OPENBLAS_NUM_THREADS")}
    order_file = args.orders / f"{args.cell_id}.json"
    if not order_file.exists():
        # The planner did not return an order for this run.
        save_new(args.output, {**record, "status": "no-order"})
        return 0
    ssa_path = json.loads(order_file.read_text())["ssa_path"]
    metrics = evaluate_ssa_path(spec["inputs"], spec["output"], spec["size_dict"], ssa_path)
    reference = raw["reference"]
    itemsize = np.dtype(reference["dtype"]).itemsize
    output_elements = int(np.prod([spec["size_dict"][i] for i in spec["output"]], dtype=object))
    limits = protocol["admission"]
    checks = {"result": metrics["max_size"] * itemsize <= limits["max_result_bytes"],
              "output": output_elements * itemsize <= limits["max_output_bytes"],
              "live": metrics["peak_size"] * itemsize <= limits["max_live_bytes"],
              "step": metrics["max_step_size"] * itemsize <= limits["max_step_bytes"]}
    record.update(flops=metrics["flops"], dtype=reference["dtype"], admission=checks)
    if not all(checks.values()):
        save_new(args.output, {**record, "status": "resource-skip"})
        return 0

    with np.load(args.arrays / f"{spec['name']}.npz", allow_pickle=False) as archive:
        arrays = [archive[f"a{i:06d}"] for i in range(len(spec["inputs"]))]
    require(all(a.dtype == np.dtype(reference["dtype"]) for a in arrays), "array dtype differs from the reference")
    symbols = {label: oe.get_symbol(i) for i, label in enumerate(sorted(spec["size_dict"]))}
    equation = ",".join("".join(symbols[x] for x in term) for term in spec["inputs"]) + "->" + \
        "".join(symbols[x] for x in spec["output"])
    linear = ssa_to_linear(ssa_path, len(spec["inputs"]))

    compile_started = time.perf_counter()
    expression = oe.contract_expression(equation, *[a.shape for a in arrays], optimize=linear)
    compile_seconds = time.perf_counter() - compile_started
    contract_started = time.perf_counter()
    result = np.asarray(expression(*arrays, backend="numpy"))
    contraction_seconds = time.perf_counter() - contract_started

    tolerance = protocol["validation"][reference["dtype"]]
    expected = complex(reference["sum_real"], reference["sum_imag"])
    observed = complex(result.sum())
    passed = (result.size == output_elements and result.dtype == np.dtype(reference["dtype"])
              and bool(np.isclose(observed, expected, rtol=tolerance["rtol"], atol=tolerance["atol"])))
    usage = resource.getrusage(resource.RUSAGE_SELF)
    save_new(args.output, json_safe({
        **record, "status": "success" if passed else "numerical-mismatch",
        "compile_seconds": compile_seconds, "contraction_seconds": contraction_seconds,
        "observed_sum": [observed.real, observed.imag], "reference_sum": [expected.real, expected.imag],
        "worker_wall_seconds": time.perf_counter() - started, "max_rss_kib": int(usage.ru_maxrss),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
