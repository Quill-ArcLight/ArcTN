"""Convert OpenQASM 2 circuits into simplified task networks with quimb.

For each circuit and task in protocol.json this writes `<stem>--<task>.json`
(inputs / output / size_dict with integer labels, readable by ArcTN and the
other planners) and `<stem>--<task>.npz` (tensors a000000, ...), plus one
`<stem>--<task>.meta.json` with counts, timing and hashes. Planner timing
starts from these files; quimb parsing and simplification time is recorded
here and never added to planner time.

Usage:
    python convert_tasks.py --protocol ../protocol-networks.json --out work/networks circuit.qasm [...]
"""
import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import quimb
import quimb.tensor as qtn

PAULI_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
_MEASURE = re.compile(r"^\s*measure\s+(\S+?)\s*->")
_QUBIT = re.compile(r"([A-Za-z_]\w*)\s*\[\s*(\d+)\s*\]")


# OpenQASM 2 gates used by MQT Bench that quimb 1.15 cannot parse, written
# with gates it can. rzz and rxx equal exp(-i theta/2 ZZ) and exp(-i theta/2 XX)
# exactly; ch is the exact controlled Hadamard; rccx is the qelib1.inc
# definition of the relative-phase Toffoli gate.
EXTRA_GATES = {
    "rzz": "gate rzz(theta) a,b { cx a,b; rz(theta) b; cx a,b; }",
    "rxx": "gate rxx(theta) a,b { h a; h b; cx a,b; rz(theta) b; cx a,b; h a; h b; }",
    "ch": "gate ch a,b { ry(pi/4) b; cx a,b; ry(-pi/4) b; }",
    "rccx": "gate rccx a,b,c { u2(0,pi) c; u1(pi/4) c; cx b,c; u1(-pi/4) c; cx a,c; "
            "u1(pi/4) c; cx b,c; u1(-pi/4) c; u2(0,pi) c; }",
}


class NonUnitary(ValueError):
    """The circuit has a reset or a measurement before later gates."""


def _operands(statement):
    """Return the operand list of a gate statement such as `cx q[0],q[1];`."""
    body = statement.rstrip(";").strip()
    # The operands start at the first space outside the parameter list.
    depth, cut = 0, None
    for position, char in enumerate(body):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char.isspace() and depth == 0:
            cut = position
            break
    if cut is None:
        return []
    return [token.strip() for token in body[cut:].split(",") if token.strip()]


def clean_qasm(text):
    """Drop final measurements, barriers and classical registers.

    Raises NonUnitary when a reset appears or a measured qubit (or a measured
    whole register) is used by a later gate, because such circuits are not
    unitary.
    """
    kept, measured_qubits, measured_regs = [], set(), set()
    in_gate_definition = False
    for line in text.splitlines():
        stripped = line.strip()
        if in_gate_definition:
            kept.append(line)
            in_gate_definition = "}" not in stripped
            continue
        if not stripped or stripped.startswith("//"):
            kept.append(line)
            continue
        head = stripped.split(None, 1)[0].split("(", 1)[0]
        if head in ("gate", "opaque"):
            kept.append(line)
            in_gate_definition = head == "gate" and "}" not in stripped
            continue
        if head == "reset":
            raise NonUnitary("reset")
        if head in ("creg", "barrier"):
            continue
        match = _MEASURE.match(stripped)
        if match:
            qubit = _QUBIT.fullmatch(match.group(1))
            if qubit:
                measured_qubits.add((qubit.group(1), int(qubit.group(2))))
            else:
                measured_regs.add(match.group(1))
            continue
        if head not in ("OPENQASM", "include", "qreg") and (measured_qubits or measured_regs):
            for operand in _operands(stripped):
                qubit = _QUBIT.fullmatch(operand)
                register = qubit.group(1) if qubit else operand
                if register in measured_regs or (qubit and (register, int(qubit.group(2))) in measured_qubits):
                    raise NonUnitary("gate after measurement")
                if not qubit and any(reg == register for reg, _ in measured_qubits):
                    raise NonUnitary("gate after measurement")
        kept.append(line)
    used = {line.strip().split(None, 1)[0].split("(", 1)[0] for line in kept if line.strip()}
    defined = {line.strip().split()[1].split("(", 1)[0] for line in kept if line.strip().startswith("gate ")}
    extra = [EXTRA_GATES[name] for name in EXTRA_GATES if name in used and name not in defined]
    if extra:
        # Definitions go right after the include line so every later use sees them.
        at = next((i + 1 for i, line in enumerate(kept) if line.strip().startswith("include")), 0)
        kept[at:at] = extra
    return "\n".join(kept) + "\n"


def load_circuit(text):
    return qtn.Circuit.from_openqasm2_str(clean_qasm(text))


def middle_site(n):
    return (n - 1) // 2


def open_qubits(n, task):
    if n <= task["full_state_max_qubits"]:
        return list(range(n))
    k = task["open_qubits_otherwise"]
    sites = sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})
    if len(sites) != k:
        raise ValueError("evenly spaced open qubits collide")
    return sites


def simplify_options(pre):
    return {"seq": pre["simplify_sequence"], "atol": pre["simplify_atol"],
            "equalize_norms": pre["simplify_equalize_norms"]}


def build_task(text, task_name, task, pre):
    """Return (tensor network, output index names, task details).

    Each task parses the circuit afresh so quimb's per-circuit caches cannot
    move work between tasks and the recorded time covers one task only.
    """
    circ = load_circuit(text)
    n = circ.N
    fs = simplify_options(pre)
    if task_name == "amplitude":
        tn = circ.amplitude_tn("0" * n, simplify_sequence=fs["seq"], simplify_atol=fs["atol"],
                               simplify_equalize_norms=fs["equalize_norms"], dtype="complex128")
        return tn, (), {"bitstring": "0" * n}
    if task_name in ("expectation_z", "expectation_zz"):
        c = middle_site(n)
        where = (c,) if task_name == "expectation_z" else (c, c + 1)
        G = PAULI_Z if len(where) == 1 else np.kron(PAULI_Z, PAULI_Z)
        tn = circ.local_expectation_tn(G, where, simplify_sequence=fs["seq"], simplify_atol=fs["atol"],
                                       simplify_equalize_norms=fs["equalize_norms"], dtype="complex128")
        return tn, (), {"observable": "Z" * len(where), "where": list(where)}
    if task_name == "state":
        sites = open_qubits(n, task)
        psi = circ.get_psi_simplified(**fs)
        fixed = [i for i in range(n) if i not in set(sites)]
        psi.isel_({psi.site_ind(i): 0 for i in fixed})
        out = tuple(psi.site_ind(i) for i in sites)
        psi.full_simplify_(output_inds=out, **fs)
        psi.astype_("complex128")
        return psi, out, {"open_qubits": sites, "fixed_qubits_value": 0}
    raise ValueError(f"unknown task {task_name}")


def export(tn, output_inds):
    """Map a quimb network to integer-labelled inputs, output and arrays."""
    labels = {}

    def label(ix):
        return labels.setdefault(ix, len(labels))

    tensors = list(tn.tensors)
    inputs = [[label(ix) for ix in t.inds] for t in tensors]
    output = [label(ix) for ix in output_inds]
    size_dict = {}
    for t in tensors:
        for ix, dim in zip(t.inds, t.shape):
            size_dict[str(labels[ix])] = int(dim)
    for ix in output_inds:
        size_dict.setdefault(str(labels[ix]), int(tn.ind_size(ix)))
    arrays = [np.asarray(t.data, dtype=complex) for t in tensors]
    counts = {}
    for term in inputs:
        for leg in set(term):
            counts[leg] = counts.get(leg, 0) + 1
    hyper = sum(1 for leg, c in counts.items() if c > 2 or (c > 1 and leg in set(output)))
    spec = {"inputs": inputs, "output": output, "size_dict": size_dict}
    return spec, arrays, {"tensors": len(inputs), "indices": len(size_dict), "hyperedges": hyper,
                          "output_order": len(output)}


def structure_sha256(spec):
    return hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def convert(path, protocol, out_dir, only_tasks=None):
    pre = protocol["preprocessing"]
    text = Path(path).read_text(encoding="utf-8")
    stem = Path(path).stem
    source_sha = hashlib.sha256(text.encode()).hexdigest()
    records = []
    try:
        raw = load_circuit(text)
    except NonUnitary as error:
        return [{"circuit": stem, "status": "excluded", "reason": f"non-unitary: {error}"}]
    except Exception as error:
        return [{"circuit": stem, "status": "excluded", "reason": f"parse failed: {type(error).__name__}: {error}"}]
    n, raw_gates = raw.N, len(raw.gates)
    for task_name, task in protocol["tasks"].items():
        if only_tasks and task_name not in only_tasks:
            continue
        record = {"circuit": stem, "task": task_name, "num_qubits": n, "gates": raw_gates,
                  "source_sha256": source_sha}
        if n < task.get("min_qubits", 1):
            records.append({**record, "status": "not-applicable", "reason": "too few qubits"})
            continue
        try:
            started = time.perf_counter()
            try:
                tn, out, details = build_task(text, task_name, task, pre)
            except ZeroDivisionError:
                # quimb divides by the network norm when equalizing norms, which
                # fails when the task value is exactly zero. Norm equalization
                # only rescales tensors, so repeat without it and record why.
                tn, out, details = build_task(text, task_name, task, {**pre, "simplify_equalize_norms": False})
                details = {**details, "zero_norm_retry": True}
            seconds = time.perf_counter() - started
        except Exception as error:
            records.append({**record, "status": "excluded",
                            "reason": f"network build failed: {type(error).__name__}: {error}"})
            continue
        spec, arrays, shape = export(tn, out)
        status = "ready" if shape["tensors"] > 2 else "trivial"
        name = f"{stem}--{task_name}"
        meta = {**record, **details, **shape, "status": status, "quimb_seconds": seconds,
                "exponent": float(getattr(tn, "exponent", 0.0)), "structure_sha256": structure_sha256(spec),
                "simplify": pre["simplify_sequence"], "quimb_version": quimb.__version__}
        (out_dir / f"{name}.json").write_text(json.dumps({"name": name, **spec}) + "\n", encoding="utf-8")
        np.savez(out_dir / f"{name}.npz", **{f"a{i:06d}": a for i, a in enumerate(arrays)})
        (out_dir / f"{name}.meta.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
        records.append(meta)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary", help="JSON-lines summary path (default: <out>/conversion-summary.jsonl)")
    parser.add_argument("--tasks", nargs="*", help="subset of protocol tasks (default: all)")
    parser.add_argument("qasm", nargs="+")
    args = parser.parse_args()
    protocol = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for path in args.qasm:
        records = convert(path, protocol, out_dir, args.tasks)
        summary.extend(records)
        for r in records:
            print(f"{r['circuit']}\t{r.get('task', '-')}\t{r['status']}\t{r.get('tensors', '')}\t{r.get('reason', '')}",
                  file=sys.stderr)
    summary_path = Path(args.summary) if args.summary else out_dir / "conversion-summary.jsonl"
    with open(summary_path, "a", encoding="utf-8") as handle:
        for r in summary:
            handle.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
