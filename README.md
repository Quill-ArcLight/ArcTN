# ArcTN

**English** | [简体中文](README.zh-CN.md)

ArcTN is a tensor network library written in Rust with a Python interface for contraction path search, slicing, and numerical contraction. It also integrates with Quimb, Cotengra, and opt_einsum.

[Complete installation packages](https://github.com/Quill-ArcLight/ArcTN/releases/tag/v1.0.1) · [Documentation](https://quantumquill.arclightquantum.com/en/docs/arctn/index.html)

<a id="algorithms-and-execution"></a>

## Features

- Search for contraction paths using the Light and Heavy presets.
- Simplify tensor networks and use slicing to reduce intermediate tensor sizes.
- Contract real and complex tensors on CPUs.
- Save and reuse contraction paths, and compile them for repeated execution.
- Integrate with other Python tensor network tools; optionally execute slices in parallel with MPI.

<a id="python"></a>

## Python usage

<a id="python-installation"></a>

### Installation

Under **Assets** on the [ArcTN 1.0.1 release page](https://github.com/Quill-ArcLight/ArcTN/releases/tag/v1.0.1), download one `.whl` matching your system and Python version. Complete wheels include the Light/Heavy engine; you do not need Rust or manual shared-library configuration.

| System and processor | Platform text in the filename |
| --- | --- |
| macOS, Apple silicon | `macosx_..._arm64` |
| macOS, Intel | `macosx_..._x86_64` |
| Windows, x86-64 | `win_amd64` |
| Linux, x86-64 | `manylinux_2_28_x86_64` |
| Linux, ARM64 | `manylinux_2_28_aarch64` |

Each platform has wheels for CPython 3.12 (`cp312`) and 3.13 (`cp313`). Check your version with `python --version`. Linux requires glibc 2.28 or later; the macOS minimum version is indicated in the wheel filename.

Open a terminal in the directory containing the downloaded wheel, then create an environment and install:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install --only-binary=arctn --find-links=. arctn==1.0.1
```

In Windows PowerShell, replace the second line with `.venv\Scripts\Activate.ps1`. Dependencies such as NumPy are installed automatically. Run the examples below in this environment.

<a id="light-and-heavy-interface"></a>

### Search for a contraction path

This example multiplies three matrices with shapes `(2, 3)`, `(3, 4)`, and `(4, 2)`, producing a `(2, 2)` result:

```python
from arctn import arctn_path

inputs = [["a", "b"], ["b", "c"], ["c", "d"]]
output = ["a", "d"]
size_dict = {"a": 2, "b": 3, "c": 4, "d": 2}

path = arctn_path(inputs, output, size_dict, preset="light", seed=0)
print(path)
```

`inputs` lists each tensor's indices, `output` specifies the output indices and their order, and `size_dict` gives the index dimensions. The returned `path` lists which two tensors to contract at each step. This call searches for a path without requiring array values.

**To use Heavy, change `preset="light"` to `preset="heavy"`.** Both use the same import. Heavy is the default when `preset` is omitted.

### Execute a contraction

Using the network definition above, supply floating-point or complex arrays in the same order as `inputs`:

```python
import numpy as np
from arctn import arctn_contract

arrays = [
    np.arange(6, dtype=float).reshape(2, 3),
    np.arange(12, dtype=float).reshape(3, 4),
    np.arange(8, dtype=float).reshape(4, 2),
]
result = arctn_contract(inputs, output, size_dict, arrays, preset="light", seed=0)
print(result)
```

`arctn_contract` searches for a path and executes it, returning the result array. This example is equivalent to `arrays[0] @ arrays[1] @ arrays[2]`.

Choose the function for the result you need; you do not need to call them in sequence:

| Task | Function | Return value |
| --- | --- | --- |
| Search for a path | `arctn_path` | A contraction path |
| Obtain a path, cost metrics, and slicing information | `arctn_schedule` | A dictionary containing the path and metrics |
| Search and execute | `arctn_contract` | The result array |

### Common settings

- `preset="light"` or `"heavy"`: select the search preset.
- `seed=0`: set the search's random seed.
- `max_time=10.0`: set the search time budget in seconds; this is not a hard timeout for the full numerical computation.
- `read_write_weight=0`: optimize paths for FLOPs alone; the default also considers read/write costs.

`arctn_schedule` and `arctn_contract` also accept `target_size`, an upper limit on the number of elements in each intermediate tensor within a slice, rather than on total process memory. To allow output-index slicing, also set `allow_output_slicing=True`; execution still returns the complete output.

See the [Python interface guide (Chinese)](pybind/README.md#interfaces) for other parameters, path reuse, and Quimb examples. See the [MPI guide (Chinese)](docs/mpi.md) for parallel execution; MPI currently supports internal-index slicing only.

<a id="rust"></a>

## Rust usage

Requires Rust 1.82 or later:

```sh
git clone https://github.com/Quill-ArcLight/ArcTN.git
cd ArcTN
cargo run --release --example contraction
```

<a id="source-and-availability"></a>
<a id="engine-library"></a>

This repository provides source for standalone algorithms, slicing, numerical execution, and calling interfaces. The Light/Heavy engine is proprietary and included in complete wheels; source builds require [separate engine configuration](docs/engine-interface.md). For Python source builds, see the [build instructions (Chinese)](pybind/README.md#source-installation).

<a id="license"></a>

## License

ArcTN is provided for noncommercial use. Public source is covered by the [Arclight Noncommercial Source-Available License 1.0](LICENSE); Light/Heavy is covered by the separate [engine license](LICENSE-ARCTN-ENGINE.txt). Third-party dependencies retain their own licenses; see the [third-party notices](THIRD_PARTY_NOTICES.md).
