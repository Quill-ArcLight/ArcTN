"""Google TNCO adapter.

TNCO receives the network in its index-list form. With fusion enabled it first
merges small tensors and then searches the reduced network, so each returned
order is mapped back to the original tensors (fusion steps first) before it is
scored. TNCO runs eight annealing runs; the run with the lowest exact FLOPs on
the original network is kept. Planning time covers only the optimize() call;
mapping and scoring the eight orders afterwards are not included.
"""
from __future__ import annotations

from collections import Counter
import math
import time
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

from scoring import evaluate_ssa_path, require

SETTINGS = {"method": "sa", "cost_type": "float64", "betas": (0.0, 100000.0), "n_steps": 1000,
            "n_runs": 8, "n_jobs": 8, "max_width": None, "decompose_hyper_inds": False}


def run_tnco(spec, method, seed):
    from tnco.app import Optimizer, load_tn

    fuse = method.get("fuse")
    require(fuse in (None, 4), "this benchmark uses fuse=None or fuse=4")
    for key, value in SETTINGS.items():
        if key in method:
            require(tuple(method[key]) == value if isinstance(value, tuple) else method[key] == value,
                    f"TNCO setting {key} differs from the benchmark")
    encoded = tnco_input(spec["inputs"], spec["output"], spec["size_dict"])
    optimizer = Optimizer(method="sa", max_width=None, n_jobs=8, cost_type="float64", seed=seed, verbose=False)
    started = time.perf_counter()
    loaded, raw_results = optimizer.optimize(encoded, betas=(0.0, 100000.0), n_steps=1000, n_runs=8,
                                             fuse=fuse, decompose_hyper_inds=False, simplify_circuit=True)
    planning_seconds = time.perf_counter() - started
    require(len(raw_results) == 8, "TNCO did not return eight runs")
    if fuse is None:
        unfused = loaded
        # Without fusion the loader may omit the fuse_path tag; use an empty prefix.
        loaded = SimpleNamespace(ts_tags=loaded.ts_tags, ts_inds=loaded.ts_inds, output_inds=loaded.output_inds,
                                 dims=loaded.dims, tags={**loaded.tags, "fuse_path": []})
    else:
        unfused = load_tn(encoded, fuse=None, decompose_hyper_inds=False, simplify_circuit=True,
                          seed=seed, verbose=False)
    reduction = fusion_metadata(loaded, unfused, spec)
    runs = []
    for ordinal, result in enumerate(raw_results):
        native = [[int(a), int(b)] for a, b in result.path]
        ssa = fused_linear_to_ssa(reduction["fuse_path"], native, len(spec["inputs"]), reduction["leaf_order"])
        flops = evaluate_ssa_path(spec["inputs"], spec["output"], spec["size_dict"], ssa)["flops"]
        cost = float(result.cost)
        require(not math.isnan(cost), "TNCO returned a NaN cost")
        runs.append({"run": ordinal, "native_cost": cost if math.isfinite(cost) else str(cost),
                     "flops": flops, "ssa_path": ssa})
    best = min(runs, key=lambda row: (row["flops"], row["ssa_path"], row["run"]))
    return best["ssa_path"], {
        "settings": {**{k: list(v) if isinstance(v, tuple) else v for k, v in SETTINGS.items()},
                     "fuse": fuse, "seed": seed},
        "planning_seconds": planning_seconds,
        "selected_run": best["run"],
        "fused_tensors": len(reduction["fuse_path"]),
        "runs": [{k: v for k, v in row.items() if k != "ssa_path"} for row in runs],
    }


def replay_fuse_prefix(
    path: Sequence[Sequence[int]],
    n_inputs: int,
    leaf_order: Sequence[int] | None = None,
) -> tuple[list[list[int]], list[int]]:
    """Map a possibly partial live-position path and return surviving SSA IDs."""

    require(n_inputs >= 1, "TNCO network has no inputs")
    resolved = list(range(n_inputs)) if leaf_order is None else [int(value) for value in leaf_order]
    require(len(resolved) == n_inputs, "TNCO leaf-order length drifted")
    require(sorted(resolved) == list(range(n_inputs)), "TNCO leaf order is not a permutation")
    live = resolved
    result: list[list[int]] = []
    for step, pair in enumerate(path):
        require(len(pair) == 2, "TNCO contraction step is not binary")
        left, right = int(pair[0]), int(pair[1])
        require(left != right, "TNCO contraction step reuses an operand")
        require(0 <= left < len(live) and 0 <= right < len(live), "TNCO path position is invalid")
        result.append([live[left], live[right]])
        for position in sorted((left, right), reverse=True):
            live.pop(position)
        live.append(n_inputs + step)
    return result, live

def linear_to_ssa(
    path: Sequence[Sequence[int]],
    n_inputs: int,
    leaf_order: Sequence[int] | None = None,
) -> list[list[int]]:
    """Convert TNCO live-position steps to SSA IDs in original tensor order."""

    result, live = replay_fuse_prefix(path, n_inputs, leaf_order)
    require(len(result) == n_inputs - 1 and len(live) == 1, "TNCO path is incomplete")
    return result

def fused_linear_to_ssa(
    fuse_path: Sequence[Sequence[int]],
    path: Sequence[Sequence[int]],
    n_inputs: int,
    leaf_order: Sequence[int],
) -> list[list[int]]:
    """Restore fusion plus reduced-network search in original tensor order.

    TNCO appends each fusion result after removing its two live operands,
    exactly as it does for search paths. Its returned search path therefore
    continues the fusion path in the same live-position coordinate system.
    """

    return linear_to_ssa([*fuse_path, *path], n_inputs, leaf_order)

def product_dimensions(legs: Iterable[int], size_dict: Mapping[int, int]) -> int:
    value = 1
    for leg in legs:
        value *= size_dict[leg]
    return value

def tnco_input(
    inputs: Sequence[Sequence[int]], output: Sequence[int], sizes: Mapping[int, int]
) -> list[tuple[Any, ...]]:
    """Encode the original hypergraph with TNCO's documented index-list form."""

    occurrences: dict[int, list[str]] = {int(label): [] for label in sizes}
    for tensor, term in enumerate(inputs):
        for label in term:
            occurrences[int(label)].append(f"t{tensor}")
    output_set = {int(label) for label in output}
    rows: list[tuple[Any, ...]] = []
    for label in sorted(sizes):
        members: list[Any] = [int(sizes[label]), *occurrences[int(label)]]
        if int(label) in output_set:
            members.append("*")
        rows.append(tuple(members))
    return rows

def loaded_leaf_order(loaded: Any, n_inputs: int) -> list[int]:
    """Recover TNCO leaf positions from stable ``t<ordinal>`` input tags."""

    tags = list(loaded.ts_tags)
    require(len(tags) == n_inputs, "TNCO loaded tensor count drifted")
    result: list[int] = []
    for tag in tags:
        require(isinstance(tag, dict), "TNCO loaded tensor lost its source-name tag")
        name = str(tag.get("name", ""))
        require(name.startswith("t") and name[1:].isdigit(), "TNCO source-name tag is malformed")
        result.append(int(name[1:]))
    require(sorted(result) == list(range(n_inputs)), "TNCO loaded tensor tags are not a permutation")
    return result

def source_leaves_from_tag(tag: Any) -> list[int]:
    """Read original tensor names from TNCO's nested ``x``/``y`` fusion tags."""

    pending = [tag]
    leaves: list[int] = []
    while pending:
        current = pending.pop()
        require(isinstance(current, dict), "TNCO fused tensor lost its source tags")
        if set(current) == {"x", "y"}:
            pending.extend((current["x"], current["y"]))
        else:
            name = str(current.get("name", ""))
            require(
                name.startswith("t") and name[1:].isdigit(),
                "TNCO fused source-name tag is malformed",
            )
            leaves.append(int(name[1:]))
    require(len(leaves) == len(set(leaves)), "TNCO fused tag repeats an original tensor")
    return sorted(leaves)

def fusion_metadata(
    loaded: Any, unfused: Any, spec: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate the actual TNCO reduction against its saved fusion prefix."""

    inputs, output, sizes = spec["inputs"], spec["output"], spec["size_dict"]
    leaf_order = loaded_leaf_order(unfused, len(inputs))
    require("fuse_path" in loaded.tags, "TNCO did not save its fusion path")
    fuse_path = [[int(left), int(right)] for left, right in loaded.tags["fuse_path"]]
    prefix_ssa, remaining = replay_fuse_prefix(fuse_path, len(inputs), leaf_order)

    # The index-list loader numbers rows; translate those row IDs back to the
    # frozen instance's labels before saving or checking a reduced network.
    labels = sorted(sizes)

    def original_label(label: Any) -> int:
        require(isinstance(label, int) and 0 <= label < len(labels), "TNCO index label drifted")
        return labels[label]

    require(
        [sorted(original_label(label) for label in term) for term in unfused.ts_inds]
        == [sorted(inputs[tensor]) for tensor in leaf_order],
        "TNCO unfused network differs from the original input order mapping",
    )
    reduced_inputs = [sorted(original_label(label) for label in term) for term in loaded.ts_inds]
    reduced_output = sorted(original_label(label) for label in loaded.output_inds)
    reduced_sizes = sorted([original_label(label), int(size)] for label, size in loaded.dims.items())
    reduced_leaves = [source_leaves_from_tag(tag) for tag in loaded.ts_tags]

    # Independently replay only the prefix to check the returned tensor order,
    # retained hyper-indices, dimensions, and source-leaf membership.
    slots = {tensor: set(term) for tensor, term in enumerate(inputs)}
    members = {tensor: {tensor} for tensor in range(len(inputs))}
    counts: Counter[int] = Counter(label for term in inputs for label in term)
    output_set = set(output)
    prefix_flops = 0
    for step, (left, right) in enumerate(prefix_ssa):
        left_inds, right_inds = slots.pop(left), slots.pop(right)
        union = left_inds | right_inds
        prefix_flops += product_dimensions(union, sizes)
        counts.subtract(left_inds)
        counts.subtract(right_inds)
        result = {label for label in union if counts[label] > 0 or label in output_set}
        counts.update(result)
        slots[len(inputs) + step] = result
        members[len(inputs) + step] = members.pop(left) | members.pop(right)
    require(
        reduced_inputs == [sorted(slots[tensor]) for tensor in remaining],
        "TNCO reduced tensor indices/order differ from fusion prefix replay",
    )
    require(
        reduced_leaves == [sorted(members[tensor]) for tensor in remaining],
        "TNCO reduced tensor source tags/order differ from fusion prefix replay",
    )
    require(reduced_output == sorted(output), "TNCO fusion changed output indices")
    used = set(reduced_output).union(*(set(term) for term in reduced_inputs))
    require(
        reduced_sizes == [[label, sizes[label]] for label in sorted(used)],
        "TNCO reduced dimensions differ from the original input",
    )
    return {"leaf_order": leaf_order, "fuse_path": fuse_path, "fuse_prefix_flops": prefix_flops}
