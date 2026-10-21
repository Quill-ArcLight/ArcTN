#!/usr/bin/env python3
"""Count qubits, depth and gates of the selected MQT Bench circuits.

Measurements, barriers, register declarations and gate definitions are not
gates. Every other statement is one gate. Depth places each gate directly
after the last gate on any of its qubits.

Example:
    python mqtbench/tools/circuit_table.py --circuits work/circuits --out mqtbench/circuits/circuits.csv
"""
import argparse
import csv
import re
from pathlib import Path

QUBIT = re.compile(r"([A-Za-z_]\w*)\s*\[\s*(\d+)\s*\]")
NAME = re.compile(r"^(?P<family>.+?)_indep_qiskit_(?P<n>\d+)\.qasm$")
SKIP = ("OPENQASM", "include", "qreg", "creg", "barrier", "measure", "reset")


def operands(statement):
    """Qubits of a gate statement such as `cx q[0],q[1];`; the parameter list is skipped."""
    depth = 0
    for position, char in enumerate(statement):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char.isspace() and depth == 0:
            return QUBIT.findall(statement[position:])
    return []


def count(text):
    layers, gates, in_definition = {}, 0, False
    for statement in text.replace("\n", " ").split(";"):
        statement = statement.strip()
        if in_definition:
            in_definition = "}" not in statement
            continue
        if statement.startswith("gate ") or statement.startswith("opaque "):
            in_definition = "}" not in statement
            continue
        if not statement or statement.startswith("//") or statement.startswith(SKIP):
            continue
        qubits = operands(statement)
        if not qubits:
            continue
        level = 1 + max(layers.get(q, 0) for q in qubits)
        for q in qubits:
            layers[q] = level
        gates += 1
    return max(layers.values(), default=0), gates


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--circuits", type=Path, required=True, help="directory with the selected .qasm files")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.circuits.glob("*.qasm")):
        match = NAME.match(path.name)
        depth, gates = count(path.read_text(encoding="utf-8"))
        rows.append({"file": path.name, "family": match["family"], "qubits": int(match["n"]),
                     "depth": depth, "gates": gates})
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} circuits -> {args.out}")


if __name__ == "__main__":
    main()
