# ArcTN

[简体中文](https://github.com/Quill-ArcLight/ArcTN/blob/main/README.md) | **English**

ArcTN is a tensor network library written in Rust for contraction order optimization, slicing, and numerical contraction. Its Python interface connects to Quimb, Cotengra, and opt_einsum. ArcTN can supply contraction paths to other tools or execute tensor network contractions directly.

[Documentation (Chinese)](https://quantumquill.arclightquantum.com/docs/arctn/index.html)

<a id="algorithms-and-execution"></a>

## Features

- Contraction order optimization: greedy search, random-greedy search, subset dynamic programming, fixed-leaf-order dynamic programming, and hypergraph bisection.
- Contraction tree optimization: subtree reconfiguration, simulated annealing, and parallel tempering.
- Network simplification and reconstruction of an original-network path from a simplified-network path.
- Slicing with a fixed contraction path, or dynamic slicing with local path improvements.
- Dense real and complex tensor contraction on CPUs, contraction path compilation, serialization and repeated execution, and path caching.
- NumPy and explicitly selected external array backends; optional MPI execution of independent slices.

A network is specified by the indices of each input tensor, the output indices, and the index dimensions. Numerical execution also requires the corresponding arrays. Frontends such as Quimb convert quantum circuits into tensor networks; ArcTN takes the resulting networks as input.

<a id="source-and-availability"></a>

## Source code and Light / Heavy

This repository provides the source for standalone algorithms, network and path types, slicing, numerical execution, Python bindings, and command-line tools. Code that Arclight has the right to license is distributed under the **[Arclight Noncommercial Source-Available License 1.0](LICENSE)**, which prohibits commercial use and closed-source integration. This is not an OSI-approved open-source license.

**The Light and Heavy implementations are proprietary; this repository provides their calling interfaces.** Rust crates and Python wheels built from this repository do not include the engine. The standalone Rust algorithms, the CLI's `--method greedy`, and numerical tensor contraction work without it. Python users can execute existing contraction paths, slice networks, and simplify networks. Calling Light/Heavy through interfaces such as `arctn_path`, `arctn_schedule`, and `arctn_plan` requires a compatible, separately licensed shared library.

The complete Python wheel containing the engine is currently for internal use only and is not distributed through PyPI or public GitHub Releases. After installation, the Python interface automatically detects the bundled shared library and loads it when Light/Heavy is called. The shared library is licensed separately and is not covered by this repository's source license; external use or distribution requires separate authorization.

[CI](https://github.com/Quill-ArcLight/ArcTN/actions/workflows/test.yml) runs Rust and CPython 3.13 tests on Linux, macOS, and Windows, plus Open MPI multiprocess tests on Linux. MPI remains an optional experimental feature outside the current supported release scope. Standard single-machine Rust/Python functionality does not require MPI. See the [MPI documentation (Chinese)](docs/mpi.md) for usage and dependency maintenance details.

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

This random-greedy example does not require the Light/Heavy shared library:

```rust
use arctn::{random_greedy, TensorNetwork};

fn main() -> Result<(), String> {
    let net = TensorNetwork {
        name: "matrix_product".into(),
        inputs: vec![vec![0, 1], vec![1, 2]],
        output: vec![0, 2],
        size_dict: [(0, 2), (1, 3), (2, 2)].into_iter().collect(),
    };
    let (path, stats) = random_greedy(&net, 16, 0)?;
    println!("path: {path:?}; log10 FLOPs: {}", stats.log10_flops);
    Ok(())
}
```

`RAYON_NUM_THREADS` sets the size of the default Rayon thread pool. Applications can also use their own Rayon pools. The Cargo `mt` feature enables a separate matrix multiplication thread pool. When combining it with parallel slice execution, limit the total number of threads to avoid oversubscribing the available CPU cores.

<a id="python"></a>

## Python usage

### Installation

Building the Python package from this repository requires Python 3.9 or later and Rust 1.83 or later. Follow the [source installation instructions (Chinese)](pybind/README.md#source-installation). A source installation does not include the Light/Heavy shared library.

If you have received a complete wheel containing the engine, you can install it without Rust. Choose a wheel matching your operating system, CPU architecture, and Python version, and replace the placeholder below with its actual file path:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install /path/to/arctn-...whl
```

The activation command above is for macOS and Linux. In Windows PowerShell, use `.venv\Scripts\Activate.ps1`. Different operating systems, CPU architectures, and Python versions require different wheels; availability depends on the packages provided.

### Execute an existing contraction path

This example compiles and executes a matrix multiplication path without Light/Heavy:

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

<a id="light-and-heavy-interface"></a>

### Call Light / Heavy

A complete installation automatically loads its bundled shared library; no additional configuration is required. Reusing the network definition above:

```python
from arctn import arctn_schedule

result = arctn_schedule(inputs, output, sizes, preset="heavy", seed=0)
print(result["path"], result["log10_flops"])
```

Set `preset="light"` to use Light. The result includes the final path, objective weights, path metrics, optional slicing results, and contraction order optimization time. See the [Python interface guide (Chinese)](pybind/README.md) for other interfaces and Quimb usage.

`target_size` limits the number of elements in any single intermediate tensor produced within each slice, not the process's total memory use. `max_time` is checked by the search itself; it is not an operating-system timeout that forcibly terminates the process.

### Configure a separate Light/Heavy shared library

For Rust, Python source installations, or an alternative compatible shared library, set its absolute path before the first Light/Heavy call:

```sh
export ARCTN_ENGINE_LIBRARY=/absolute/path/to/libarctn_engine.so
```

Use the corresponding `.dylib` on macOS or `.dll` on Windows. PowerShell instructions are in the [engine interface documentation](docs/engine-interface.md). An explicit `ARCTN_ENGINE_LIBRARY` setting takes precedence over the shared library bundled with the Python package.

The shared library must match the operating system, CPU architecture, and [ABI version](docs/engine-interface.md#c-abi-version-1), and must come from a trusted source. If no usable library is available, Light/Heavy calls report an installation error. Standalone algorithms and execution of existing paths remain available.

<a id="license"></a>

## License

Company-owned code is covered by the [Arclight Noncommercial Source-Available License 1.0](LICENSE). Third-party dependencies retain their own licenses. See the [third-party and historical licensing notices](THIRD_PARTY_NOTICES.md) for dependency notices and an explanation of the license texts retained from earlier development revisions. The Light/Heavy shared library is licensed separately. This source license does not grant permission to use or distribute it.
