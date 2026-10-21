# Benchmarks

Scripts, inputs and recorded results for the experiments of the ArcTN paper:
QC32 planning, execution and slicing (Sections 6.2 to 6.4) and MQT Bench
planning (Section 6.6). The Hygon deployment (Section 6.5) runs on a vendor
platform and is not included.

```text
benchmarks/
├── harness/        scheduler, workers, cost model, collection and rescoring
├── qc32/           32 networks with input arrays, protocols, recorded results, analysis
│   └── slicing-scripts/   the scripts that produced the recorded slicing plans
└── mqtbench/       320 circuits, network construction, 672 networks, protocol, results, analysis
```

## Check the recorded results

These steps need only Python 3.10 or later and no planner. `rescore.py`
replays every recorded order (and, for slicing, every set of sliced indices)
on its network with the cost model in `harness/scoring.py` and compares the
costs with `runs.csv`. The analysis scripts print the numbers reported in the
paper.

```bash
cd benchmarks
python harness/rescore.py --networks qc32/networks --results qc32/results/planning
python harness/rescore.py --networks qc32/networks --results qc32/results/slicing
python harness/rescore.py --networks mqtbench/networks/networks.jsonl.xz --results mqtbench/results/planning
python qc32/analyze_planning.py
python qc32/analyze_execution.py
python qc32/analyze_slicing.py
python mqtbench/analyze_planning.py
```

## Run the planners

Requirements: Linux, `taskset`, and Python 3.12 with the packages in
`requirements.txt`. All configurations of a protocol go through one queue.
Each run is a separate process pinned to eight physical cores of one CPU
package, with eight Rayon threads and eight Cotengra workers and one BLAS
thread. The protocol files fix every planner setting, the number of runs at
once (12 for QC32, 16 for MQT Bench), the number of TNCO runs at once (8) and
the wall-time limit of a run (3600 s for QC32, 3900 s for MQT Bench). The
recorded runs used a server with eight Intel Xeon Platinum 8253 CPUs of 16
cores each. Finished runs are skipped, so a stopped run can be continued with
the same command.

**Cotengra Hyper128 takes most of the time.** In the recorded runs it accounts
for 77% of the summed planning time on QC32 (19.6 of 25.6 hours) and 84% on
MQT Bench (162 of 192 hours). Without it, the planning runs need about half an
hour on 12 slots for QC32 and about two hours on 16 slots for MQT Bench; with
it, about two and twelve hours. Run the other configurations first and
Hyper128 afterwards, as below, or leave it out if you do not need it.

QC32 (8 configurations, 32 networks, seeds 0 to 4):

```bash
python harness/run_benchmark.py --protocol qc32/protocol-planning.json --networks qc32/networks \
    --results work/qc32-planning --skip cotengra-hyper128-optuna
python harness/run_benchmark.py --protocol qc32/protocol-planning.json --networks qc32/networks \
    --results work/qc32-planning --only cotengra-hyper128-optuna
python harness/collect.py work/qc32-planning --out work/qc32-planning-collected
python qc32/analyze_planning.py --runs work/qc32-planning-collected/runs.csv
```

MQT Bench (9 configurations, 672 networks, seed 0):

```bash
python harness/unpack_networks.py mqtbench/networks/networks.jsonl.xz work/mqtbench-networks
python harness/run_benchmark.py --protocol mqtbench/protocol-planning.json --networks work/mqtbench-networks \
    --results work/mqtbench-planning --skip cotengra-hyper128-optuna
python harness/run_benchmark.py --protocol mqtbench/protocol-planning.json --networks work/mqtbench-networks \
    --results work/mqtbench-planning --only cotengra-hyper128-optuna
python harness/collect.py work/mqtbench-planning --out work/mqtbench-planning-collected
python mqtbench/analyze_planning.py --runs work/mqtbench-planning-collected/runs.csv
```

`--max-concurrent N` runs fewer runs at once on a smaller machine, and
`--networks-subset a,b` or `--max-tensors N` restricts the networks. On QC32,
Light and Heavy stop by the rate rule and Cotengra Auto and AutoHQ by their
own rate settings, so planning times and, through them, the returned orders
depend on the machine. Cotengra and OMECO take no seed; their five runs differ
only by chance.

## Execution (QC32)

The execution runs contract each saved order once with
`opt_einsum.contract_expression` and NumPy, on eight physical cores with eight
OpenBLAS threads and no warm-up. A run is skipped when the order needs a
result above 64 GiB, an output above 32 GiB, or more than 128 GiB of live
tensors or for one contraction. The result is checked against the reference
sum stored with each network. Total time adds the wall time of the planning
process and that of the execution process.

```bash
python harness/unpack_orders.py qc32/results/planning/paths.jsonl.xz work/qc32-orders
mkdir -p work/qc32-arrays && tar -xJf qc32/networks/arrays.tar.xz -C work/qc32-arrays
python harness/run_benchmark.py --protocol qc32/protocol-execution.json --networks qc32/networks \
    --results work/qc32-execution --orders work/qc32-orders --arrays work/qc32-arrays
python harness/collect.py work/qc32-execution --out work/qc32-execution-collected
python qc32/analyze_execution.py --runs work/qc32-execution-collected/runs.csv \
    --planning-runs qc32/results/planning/runs.csv
```

To execute your own orders, unpack `paths.jsonl.xz` from your collected
planning results instead and pass that collected `runs.csv` as
`--planning-runs`.

## Slicing (QC32)

`qc32/results/slicing` holds every recorded plan: the order, the sliced
indices and the size target. Fixed and Dynamic start from the same saved
Heavy order. To make that possible, the scripts in `qc32/slicing-scripts`
pass the order to a small Rust program that calls ArcTN's slicing functions
directly (`scalar/` for scalar outputs with target 2^26 elements, `tensor/`
for tensor outputs, where output indices may be sliced). They are the scripts
that produced the recorded plans. The public crate in this repository does not
export the output-slicing functions that `tensor/` needs, so these scripts are
included as a record and do not build against it unchanged.

## Building the MQT Bench networks

`mqtbench/circuits/circuits.tar.xz` holds the 320 selected circuit files and
`circuits.csv` their qubit counts, depths and gate counts. The selection rule
uses file names only: for the 14 families provided at every qubit count,
n = 10, 20, ..., 130; for the 17 families provided at fixed sizes, all files.

```bash
pip install -r mqtbench/requirements-networks.txt
mkdir -p work/circuits && tar -xJf mqtbench/circuits/circuits.tar.xz -C work/circuits
ls work/circuits > work/circuit-names.txt
python mqtbench/tools/select_circuits.py --protocol mqtbench/protocol-networks.json \
    --listing work/circuit-names.txt --out work/manifest.json
python mqtbench/tools/run_conversion.py --protocol mqtbench/protocol-networks.json \
    --manifest work/manifest.json --circuits work/circuits --out work/networks --workers 4
```

With the original archive, `select_circuits.py --archive MQTBench_all.zip
--extract-to work/circuits` selects and extracts the same files. Each
(circuit, task) pair is built in its own process with a 3600 s limit.
`networks/conversion-summary.csv` lists the outcome for all 1280 pairs;
the 672 networks with 3 to 4999 tensors are planned and stored in
`networks/networks.jsonl.xz`.

## Recorded results

| Experiment | Records |
|---|---|
| QC32 planning (Section 6.2) | `qc32/results/planning` |
| QC32 execution (Section 6.3) | `qc32/results/execution`, which executes the orders of the planning runs |
| QC32 slicing (Section 6.4) | `qc32/results/slicing`, groups `scalar` and `tensor` |
| MQT Bench planning (Section 6.6) | `mqtbench/results/planning` |

The baselines are Cotengra 0.8.2, OMECO 0.2.6 and TNCO 0.4.0. The scripts in
`harness/` use the ArcTN 1.0.0 Python interface.

File formats:

- A network file has `name`, `inputs` (one list of integer index labels per
  tensor), `output` and `size_dict` (label to dimension). QC32 files also have
  `reference`: the data type and the sum of the exact output.
- `runs.csv` has one row per run. `planning_seconds` is the wall time of the
  planner call; `process_seconds` is the wall time of the whole process (QC32
  only). FLOPs, writes, largest result and peak live elements are exact
  integers from `harness/scoring.py`.
- `paths.jsonl.xz` has one order per line as an SSA path: inputs are numbered
  0 to n-1 and the k-th contraction creates tensor n+k. `plans.jsonl.xz` adds
  `sliced_indices`, `target_size` and `output_slicing_allowed`.

## Licenses

The QC32 networks and arrays come from Einsum Benchmark and the circuit files
from MQT Bench; both are MIT-licensed, see
`qc32/networks/LICENSE-einsum-benchmark.txt` and
`mqtbench/circuits/LICENSE-mqt-bench.txt`. Everything else here is under the
license of this repository.
