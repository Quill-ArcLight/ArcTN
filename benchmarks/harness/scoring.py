"""Common cost model for contraction orders given as SSA paths.

All planners are scored here on the network they received, with exact
integers. FLOPs count scalar multiplications of the pairwise contractions.
Writes sum the sizes of all contraction results. Largest is the size of the
largest result. Peak is the largest total size of live tensors (inputs and
results) at any step, counting a result while its operands are still alive.
Step is the largest combined size of the two inputs and the result of one
contraction.
"""

from __future__ import annotations

from collections import Counter
import math
from typing import Any, Iterable, Mapping, Sequence


class ProtocolError(RuntimeError):
    """An input network or a returned path breaks the benchmark contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProtocolError(message)


def ssa_to_linear(ssa_path: Sequence[Sequence[int]], n_inputs: int) -> list[tuple[int, int]]:
    """Convert SSA ids to the position-based form used by opt_einsum."""
    require(n_inputs >= 1, "network has no tensors")
    live = list(range(n_inputs))
    linear: list[tuple[int, int]] = []
    for step, pair in enumerate(ssa_path):
        require(len(pair) == 2, f"SSA step {step} is not binary")
        left_id, right_id = int(pair[0]), int(pair[1])
        require(left_id != right_id, f"SSA step {step} repeats a node")
        try:
            left = live.index(left_id)
            right = live.index(right_id)
        except ValueError as exc:
            raise ProtocolError(f"SSA step {step} references a node that is not live") from exc
        linear.append((left, right))
        for position in sorted((left, right), reverse=True):
            live.pop(position)
        live.append(n_inputs + step)
    require(len(ssa_path) == n_inputs - 1 and len(live) == 1, "SSA path is incomplete")
    return linear


def _product(legs: Iterable[int], size_dict: Mapping[int, int]) -> int:
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
    """Check an SSA path and return its exact costs."""
    n = len(inputs)
    require(n >= 1 and len(ssa_path) == n - 1, "SSA path length is not n_tensors - 1")
    output_set = set(output)
    slots: list[tuple[int, ...] | None] = []
    raw_sizes: list[int] = []
    refcount: Counter[int] = Counter()
    for term in inputs:
        counts = Counter(term)
        require(max(counts.values(), default=0) <= 1, "a tensor repeats an index")
        unique = tuple(sorted(counts))
        slots.append(unique)
        refcount.update(unique)
        raw_sizes.append(_product(term, size_dict))

    alive = sum(raw_sizes)
    peak = alive
    flops = 0
    writes = 0
    largest = _product(output, size_dict) if n == 1 else 1
    max_step_size = 0  # largest two inputs plus result of one contraction
    for step, pair in enumerate(ssa_path):
        require(len(pair) == 2, f"SSA step {step} is not binary")
        left_id, right_id = int(pair[0]), int(pair[1])
        require(left_id != right_id, f"SSA step {step} contracts a tensor with itself")
        require(0 <= left_id < len(slots) and 0 <= right_id < len(slots), f"SSA step {step} is out of range")
        left, right = slots[left_id], slots[right_id]
        require(left is not None and right is not None, f"SSA step {step} reuses a tensor")
        slots[left_id] = None
        slots[right_id] = None
        union = tuple(sorted(set(left).union(right)))
        flops += _product(union, size_dict)
        for leg in left:
            refcount[leg] -= 1
        for leg in right:
            refcount[leg] -= 1
        result = tuple(leg for leg in union if refcount[leg] > 0 or leg in output_set)
        refcount.update(result)
        size = _product(result, size_dict)
        writes += size
        largest = max(largest, size)
        max_step_size = max(max_step_size, raw_sizes[left_id] + raw_sizes[right_id] + size)
        peak = max(peak, alive + size)
        alive += size - raw_sizes[left_id] - raw_sizes[right_id]
        slots.append(result)
        raw_sizes.append(size)

    remaining = [value for value in slots if value is not None]
    require(len(remaining) == 1, "SSA path does not leave one tensor")
    require(tuple(sorted(remaining[0])) == tuple(sorted(output)), "final indices differ from the output")
    return {
        "flops": flops,
        "total_write": writes,
        "max_size": largest,
        "peak_size": peak,
        "max_step_size": max_step_size,
        "log10_flops": None if flops == 0 else math.log10(flops),
        "log2_max_size": math.log2(largest),
    }


def load_network(path: str | Any) -> dict[str, Any]:
    """Read a network file: inputs and output as integer labels, size_dict as label -> dimension."""
    import json
    from pathlib import Path

    value = json.loads(Path(path).read_text(encoding="utf-8"))
    sizes = {int(k): int(v) for k, v in value["size_dict"].items()}
    inputs = [list(map(int, term)) for term in value["inputs"]]
    output = list(map(int, value["output"]))
    require(inputs, "network has no input tensors")
    require(len(output) == len(set(output)), "repeated output label")
    require(all(len(term) == len(set(term)) for term in inputs), "an input repeats a label")
    used = set(output).union(*map(set, inputs))
    require(used == set(sizes), "size_dict differs from the labels in use")
    require(all(size > 0 for size in sizes.values()), "non-positive dimension")
    occurrences = Counter(label for term in inputs for label in term)
    require(set(output) <= set(occurrences), "an output label is absent from the inputs")
    # A pairwise order cannot sum away an index that appears on one input only.
    require(all(count >= 2 or label in output for label, count in occurrences.items()),
            "network has an index on one input only that is not an output")
    return {"name": value["name"], "inputs": inputs, "output": output, "size_dict": sizes}


def evaluate_sliced_plan(inputs, output, size_dict, ssa_path, sliced_indices, target_size, output_slicing_allowed):
    """Cost of a sliced plan: every slice fixes the sliced indices and runs the same order.

    Total FLOPs are the per-slice FLOPs times the number of slices; summing the
    slices and assembling output blocks are not counted. Every contraction
    result of a slice must fit target_size.
    """
    legs = sorted(int(leg) for leg in sliced_indices)
    require(len(legs) == len(set(legs)), "an index is sliced twice")
    outer = sorted(set(legs) & set(output))
    require(not outer or output_slicing_allowed, "the plan slices an output index")
    sliced_sizes = dict(size_dict)
    n_slices = 1
    for leg in legs:
        require(leg in sliced_sizes, "unknown sliced index")
        n_slices *= sliced_sizes[leg]
        sliced_sizes[leg] = 1
    per_slice = evaluate_ssa_path(inputs, output, sliced_sizes, ssa_path)
    require(per_slice["max_size"] <= target_size, "a slice has a result larger than the target")
    return {"flops": per_slice["flops"] * n_slices, "n_slices": n_slices, "per_slice_max_size": per_slice["max_size"],
            "output_indices_sliced": len(outer)}
