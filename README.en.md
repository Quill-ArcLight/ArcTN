# ArcTN

[简体中文](https://github.com/Quill-ArcLight/ArcTN/blob/main/README.md) | **English**

ArcTN is a tensor network library written in Rust for contraction order optimization, slicing, and numerical contraction. Its Python interface connects to Quimb, Cotengra, and opt_einsum. ArcTN can supply contraction paths to other tools or execute tensor network contractions directly.

[Documentation (Chinese)](https://quantumquill.arclightquantum.com/docs/arctn/index.html)

<a id="algorithms-and-execution"></a>

## Features

- Contraction order optimization: greedy search, random-greedy search, subset dynamic programming, fixed-leaf-order dynamic programming, and hypergraph bisection.
- Contraction tree optimization: subtree reconfiguration, simulated annealing, and parallel tempering.
- Network simplification and reconstruction of an original-network path from a simplified-network path.
- Slicing with a fixed contraction path, or dynamic slicing with local changes to the contraction path.
- Dense real and complex tensor contraction on CPUs.
- Saving contraction paths and slicing information, compiling paths for repeated execution, and path caching.
- NumPy and explicitly selected external array backends; optional MPI execution of independent slices.

A network is specified by the indices of each input tensor, the output indices, and the index dimensions. Numerical execution also requires the corresponding arrays. Frontends such as Quimb convert quantum circuits into tensor networks; ArcTN takes the resulting networks as input.

<a id="source-and-availability"></a>

## Source code and Light / Heavy

This repository provides the source for standalone algorithms, network and path types, slicing, numerical execution, Python bindings, and command-line tools. Code that Arclight has the right to license is distributed under the **[Arclight Noncommercial Source-Available License 1.0](LICENSE)**, which prohibits commercial use and closed-source integration. This is not an OSI-approved open-source license.

**The Light and Heavy implementations are proprietary; this repository provides their calling interfaces.** Rust crates and Python wheels built from this repository do not include the engine. The standalone Rust algorithms, the CLI's `--method greedy`, and numerical tensor contraction work without it. Python users can execute existing contraction paths with or without specified sliced indices, and simplify networks.

To use Light/Heavy through interfaces such as `arctn_path`, `arctn_schedule`, and `arctn_plan`, install a complete wheel containing the engine or configure a compatible, separately licensed shared library. See [Python usage](#python) for installation instructions.

[CI](https://github.com/Quill-ArcLight/ArcTN/actions/workflows/test.yml) runs Rust and CPython 3.13 tests on Linux, macOS, and Windows, plus Open MPI multiprocess tests on Linux.

MPI is an optional feature for parallel slice execution. `tnmpi` distributes slices across processes, contracts them along a saved path, and sums the results. Standard single-machine Rust/Python functionality does not require MPI. See the [MPI documentation (Chinese)](docs/mpi.md) for installation and runtime requirements.

<a id="rust"></a>

## Rust usage

Building from source requires Rust 1.82 or later:

```sh
git clone https://github.com/Quill-ArcLight/ArcTN.git
cd ArcTN
cargo build --release
cargo test
cargo run --example contraction
```

`RAYON_NUM_THREADS` sets the size of the default Rayon thread pool. Applications can also use their own Rayon pools. The Cargo `mt` feature enables a separate matrix multiplication thread pool. When combining it with parallel slice execution, limit the total number of threads to avoid oversubscribing the available CPU cores.

<a id="python"></a>

## Python usage

<a id="python-installation"></a>

### Installation

A complete ArcTN 1.0.0 wheel containing the Light/Heavy engine can be installed without Rust. Obtain a wheel matching your operating system, CPU architecture, and Python version, open a terminal in the directory containing that file, and run:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install --find-links=. arctn==1.0.0
```

`arctn==1.0.0` selects the version to install. `--find-links=.` also searches the current directory for compatible wheels, so you do not need to enter the filename. Dependencies such as NumPy are installed automatically.

The activation command above is for macOS and Linux. In Windows PowerShell, use `.venv\Scripts\Activate.ps1`. Supported combinations depend on the released wheels.

Building the Python package from this repository requires Python 3.9 or later and Rust 1.83 or later. Follow the [source installation instructions (Chinese)](pybind/README.md#source-installation). A source installation does not include the Light/Heavy shared library; [configure it separately](#engine-library) to use Light/Heavy.

<a id="light-and-heavy-interface"></a>

### Call Light / Heavy

Follow the [installation instructions](#python-installation) to install a complete wheel containing the Light/Heavy engine, then run the code below in the same Python environment. With a complete wheel, you do not need to clone the source, install Rust, or load the shared library manually.

**Import the `arctn_path` function and select Light or Heavy with `preset`; they are not separate Python modules to import.** On import, the Python interface detects the bundled engine library. The library is loaded and used when you first call a search function. This example uses Light:

```python
from arctn import arctn_path

inputs = [["a", "b"], ["b", "c"], ["c", "d"]]
output = ["a", "d"]
size_dict = {"a": 2, "b": 3, "c": 4, "d": 2}

path = arctn_path(
    inputs, output, size_dict,
    preset="light",
    seed=0,
)
print(path)
```

**To use Heavy, change only `preset="light"` to `preset="heavy"`. No reinstallation or change to the import is needed.** If `preset` is omitted, Heavy is used.

`inputs` lists the indices of each tensor in input order, `output` gives the indices to retain and their order, and `size_dict` gives each index's dimension. This example represents a product of three matrices with shapes `(2, 3)`, `(3, 4)`, and `(4, 2)`, producing a `(2, 2)` result.

`arctn_path` searches for a contraction order without performing the numerical contraction, so it does not need array values. It returns a list of pairs identifying the tensors to contract at each step. The default is opt_einsum's linear path format; set `use_ssa=True` to return an SSA path.

`seed` sets the random seed for the search. Both modes use the same objective parameters, with defaults `flops_weight=1, read_write_weight=64`; set `read_write_weight=0` to optimize FLOPs alone.

To obtain path metrics as well, use `arctn_schedule` instead. Reusing the network definition above:

```python
from arctn import arctn_schedule

result = arctn_schedule(inputs, output, size_dict, preset="heavy", seed=0)
print("Path:", result["path"])
print("log10 FLOPs:", result["log10_flops"])
print("log2 largest intermediate:", result["log2_max_size"])
```

`arctn_schedule` returns a dictionary containing the final path, objective weights, path metrics, optional slicing results, and contraction order optimization time. Here, `log10_flops` is the base-10 logarithm of the FLOP count, and `log2_max_size` is the base-2 logarithm of the number of elements in the largest intermediate tensor.

Choose the function for the result you need; these are not sequential steps:

| Task | Import from `arctn` | Return value |
| --- | --- | --- |
| Search for a contraction order | `arctn_path` | A contraction path |
| Search and obtain path metrics, or select sliced indices | `arctn_schedule` | A dictionary containing the path and metrics |
| Search and execute the contraction | `arctn_contract` | The result array, or `(result, info)` with `return_info=True` |

All three functions select Light or Heavy through `preset`. `arctn_contract` also requires `arrays` in the same order as `inputs` and uses the Rust CPU executor by default. See the [Python interface guide (Chinese)](pybind/README.md#interfaces) for other interfaces and Quimb usage.

`arctn_schedule` and `arctn_contract` accept `target_size`, which limits the number of elements in any single intermediate tensor produced within each slice, not the process's total memory use. `max_time` is measured in seconds and checked by the search itself; it does not cause the operating system to terminate the process.

### Execute an existing contraction path

If you already have a contraction path, you can compile and execute it without calling Light/Heavy. This is a separate matrix multiplication example:

```python
import numpy as np
from arctn import ArcTNCompiledContraction

inputs, output = [["a", "b"], ["b", "c"]], ["a", "c"]
sizes = {"a": 2, "b": 3, "c": 2}
compiled = ArcTNCompiledContraction.compile(inputs, output, sizes, ssa_path=[(0, 1)])
a = np.arange(6, dtype=np.float64).reshape(2, 3)
b = np.arange(6, dtype=np.float64).reshape(3, 2)
np.testing.assert_allclose(compiled([a, b]), a @ b)
```

<a id="engine-library"></a>

### Configure a separate Light/Heavy shared library

To call Light/Heavy from Rust or a Python source installation, or to select another compatible shared library, set the library's absolute path before the first call:

```sh
export ARCTN_ENGINE_LIBRARY=/absolute/path/to/libarctn_engine.so
```

Use the corresponding `.dylib` on macOS or `.dll` on Windows. PowerShell instructions are in the [engine interface documentation](docs/engine-interface.md). An explicit `ARCTN_ENGINE_LIBRARY` setting takes precedence over the shared library bundled with the Python package.

The shared library must match the operating system, CPU architecture, and [ABI version](docs/engine-interface.md#c-abi-version-1), and must come from a trusted source. If no usable library is available, Light/Heavy calls report an installation error. Standalone algorithms and execution of existing paths remain available.

<a id="license"></a>

## License

Company-owned code is covered by the [Arclight Noncommercial Source-Available License 1.0](LICENSE). Third-party dependencies retain their own licenses. See the [third-party and historical licensing notices](THIRD_PARTY_NOTICES.md) for dependency notices and an explanation of the license texts retained from earlier development revisions. The Light/Heavy shared library is licensed separately. This source license does not grant permission to use or distribute it.
