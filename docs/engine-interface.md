# Light and Heavy engine interface

The crate loads a native library implementing Light and Heavy. Network
validation, final path validation, metrics, and numerical execution remain
in this repository's source. Only libraries from a trusted provider should
be loaded.

## Installation and discovery

A complete Python wheel includes the engine under `arctn/_lib`. Importing
`arctn` discovers this library automatically; the high-level Python interfaces
do not require an engine path argument or environment configuration.

An explicitly set `ARCTN_ENGINE_LIBRARY` takes precedence over the bundled
library. Rust callers and Python installations built from this repository's
source must supply the engine separately and set this variable to its absolute
path before the first Light/Heavy call. For example, on Linux:

```sh
export ARCTN_ENGINE_LIBRARY=/absolute/path/to/libarctn_engine.so
```

On macOS, use the corresponding `.dylib`. On Windows PowerShell:

```powershell
$env:ARCTN_ENGINE_LIBRARY = "C:\path\to\arctn_engine.dll"
```

The engine must match the caller's operating system, CPU architecture, and
the ABI version below. A wheel must also be compatible with the installed
Python version. These requirements do not imply that binaries for every
platform are currently available. Without an engine, Light/Heavy calls
return an installation error; independent algorithms and execution of supplied
paths remain available.

Complete wheels containing the engine are currently supplied for internal use
only, with no PyPI or public GitHub Release distribution. The engine is licensed
separately and is not covered by this repository's source license. External use
or redistribution requires separate authorization; see
[source and engine availability](../README.md#source-and-availability).

## C ABI, version 1

```c
#include <stddef.h>
#include <stdint.h>

uint32_t arctn_engine_abi_version(void);
char *arctn_engine_run_v1(const uint8_t *request, size_t length);
void arctn_engine_free_v1(char *response);
```

The version function must return `1`. `run_v1` is synchronous. Its input is
`length` bytes of UTF-8 JSON, owned by the caller and valid until the call
returns. Its output is an owned, NUL-terminated UTF-8 JSON string. The caller
releases each non-null response exactly once using `free_v1` from the same
library. The engine must not unwind across the C boundary. Calls may occur
concurrently; the engine must synchronize any shared mutable state.

## Request

```json
{
  "inputs": [[10, 20], [20, 30]],
  "output": [10, 30],
  "size_dict": [[10, 2], [20, 3], [30, 2]],
  "preset": "heavy",
  "seed": 0,
  "max_time": null,
  "flops_weight": 1.0,
  "read_write_weight": 64.0,
  "rate_enabled": true,
  "target_size": null,
  "slicing_mode": "fixed",
  "threads": 8
}
```

Index identifiers are unsigned 32-bit integers and need not be contiguous.
The order of tensors, tensor axes, and output axes must be preserved. Index
dimensions are positive integers; `size_dict` contains no duplicate IDs.
`seed` is an unsigned 64-bit integer.

`preset` is `light` or `heavy`; `slicing_mode` is `fixed` or `dynamic`.
Dynamic slicing requires a positive `target_size`. Objective weights are
finite, non-negative, and not both zero. A non-null `max_time` is finite and
positive. `rate_enabled` enables the preset's cooperative rate stopping rule.
`threads` is the caller's current Rayon pool width and must be positive.

## Response

```json
{"path": [[0, 1]], "sliced_legs": null, "wall_s": 0.1}
```

`path` uses SSA tensor IDs. Inputs occupy `0..n-1`; step `k` produces `n+k`.
`sliced_legs` is `null` when no target was requested. With a target, it is an
array of index IDs, possibly empty when no slicing is needed. Output indices
and repeated indices within an input cannot be sliced by this executor.
`wall_s` is the engine's finite, non-negative planning time in seconds.

An unsuccessful call returns:

```json
{"error": "description of the error"}
```

The public adapter validates the returned path and sliced indices, recomputes
metrics, and checks the exact integer target. These checks do not protect
against malicious native code: the engine is part of the caller's trusted
process. The interface exposes the final path, slice selection, and planning
time; the public adapter supplies validated metrics.
