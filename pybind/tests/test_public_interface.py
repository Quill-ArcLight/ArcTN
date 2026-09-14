"""Public execution and optional engine integration."""

import os
from pathlib import Path
import shutil
import subprocess
import sys

import arctn
import numpy as np
import pytest

from arctn import (
    ArcTNCompiledContraction,
    ArcTNExecutionPlan,
    ArcTNOptimizer,
    arctn_contract,
    arctn_plan,
    arctn_schedule,
    arctn_simplify,
)
from arctn import arctn_py


@pytest.fixture
def network():
    inputs = [["a", "b"], ["b", "c"], ["c", "d"]]
    sizes = {"a": 2, "b": 3, "c": 4, "d": 2}
    arrays = [np.arange(6.).reshape(2, 3), np.arange(12.).reshape(3, 4),
              np.arange(8.).reshape(4, 2)]
    return inputs, ["a", "d"], sizes, arrays


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
@pytest.mark.parametrize("backend", ["native", "numpy"])
def test_execute_supplied_path_without_engine(network, dtype, backend, monkeypatch):
    monkeypatch.delenv("ARCTN_ENGINE_LIBRARY", raising=False)
    inputs, output, sizes, arrays = network
    arrays = [x.astype(dtype) for x in arrays]
    if np.issubdtype(dtype, np.complexfloating):
        arrays = [x + 0.125j * x for x in arrays]
    compiled = ArcTNCompiledContraction.compile(
        inputs, output, sizes, ssa_path=[(1, 2), (0, 3)], backend=backend)
    np.testing.assert_allclose(compiled(arrays), arrays[0] @ arrays[1] @ arrays[2], rtol=1e-6)


def test_explicit_slices_without_engine(monkeypatch):
    monkeypatch.delenv("ARCTN_ENGINE_LIBRARY", raising=False)
    a = np.arange(6.).reshape(2, 3)
    b = np.arange(6.).reshape(3, 2)
    value = arctn_py.contract_sliced(
        [[10, 20], [20, 30]], [10, 30], {10: 2, 20: 3, 30: 2},
        [a, b], [20], ssa_path=[(0, 1)])
    np.testing.assert_array_equal(value, a @ b)


def test_simplification_without_engine(network, monkeypatch):
    monkeypatch.delenv("ARCTN_ENGINE_LIBRARY", raising=False)
    inputs, output, sizes, _ = network
    result = arctn_simplify(inputs, output, sizes)
    assert result is not None


@pytest.fixture
def source_only_package(tmp_path):
    # Exercise the source-only distribution even when pytest uses a full wheel.
    shutil.copytree(
        Path(arctn.__file__).parent, tmp_path / "arctn",
        ignore=shutil.ignore_patterns("_lib", "__pycache__"),
    )
    return tmp_path


def test_missing_engine_is_explicit(source_only_package):
    env = dict(os.environ)
    env.pop("ARCTN_ENGINE_LIBRARY", None)
    env["PYTHONPATH"] = str(source_only_package)
    code = """
import os
from pathlib import Path
import arctn
from arctn import arctn_path
assert Path(arctn.__file__).parent == Path.cwd() / 'arctn'
assert 'ARCTN_ENGINE_LIBRARY' not in os.environ
assert all(hasattr(arctn, name) for name in arctn.__all__)
try:
    arctn_path([[0, 1], [1, 2]], [0, 2], {0: 2, 1: 3, 2: 2})
except ValueError as error:
    assert 'ARCTN_ENGINE_LIBRARY' in str(error), str(error)
else:
    raise AssertionError('missing engine was accepted')
"""
    subprocess.run([sys.executable, "-c", code], env=env,
                   cwd=source_only_package, check=True, timeout=30)


@pytest.mark.parametrize("override", [None, "missing", ""])
def test_import_with_bundled_engine(source_only_package, override):
    filename = {
        "linux": "libarctn_engine.so",
        "darwin": "libarctn_engine.dylib",
        "win32": "arctn_engine.dll",
    }.get(sys.platform)
    if filename is None:
        pytest.skip("no engine binary is packaged for this platform")
    library = source_only_package / "arctn" / "_lib" / filename
    library.parent.mkdir()
    # Discovery must not load the binary until a planning call. An invalid file
    # makes that boundary testable without access to private engine artifacts.
    library.touch()
    env = dict(os.environ)
    env.pop("ARCTN_ENGINE_LIBRARY", None)
    env["PYTHONPATH"] = str(source_only_package)
    expected = str(library)
    if override is not None:
        expected = str(source_only_package / "missing-engine") if override else ""
        env["ARCTN_ENGINE_LIBRARY"] = expected
    code = """
import os
import sys
import arctn
assert os.environ['ARCTN_ENGINE_LIBRARY'] == sys.argv[1]
assert all(hasattr(arctn, name) for name in arctn.__all__)
arctn.arctn_simplify([[0, 1], [1, 2]], [0, 2], {0: 2, 1: 3, 2: 2})
try:
    arctn.arctn_path([[0, 1], [1, 2]], [0, 2], {0: 2, 1: 3, 2: 2})
except ValueError as error:
    assert 'ARCTN_ENGINE_LIBRARY' in str(error), str(error)
    if sys.argv[1]:
        assert 'cannot load' in str(error), str(error)
else:
    raise AssertionError('invalid engine was accepted')
assert os.environ['ARCTN_ENGINE_LIBRARY'] == sys.argv[1]
"""
    subprocess.run([sys.executable, "-c", code, expected], env=env,
                   cwd=source_only_package, check=True, timeout=30)


engine_required = pytest.mark.skipif(
    not os.environ.get("ARCTN_ENGINE_LIBRARY"), reason="private engine not supplied")


@engine_required
@pytest.mark.parametrize("preset", ["light", "heavy"])
@pytest.mark.parametrize("mode,target", [("fixed", None), ("fixed", 4), ("dynamic", 4)])
@pytest.mark.parametrize("weights", [(1., 64.), (1., 0.), (0., 1.)])
def test_engine_plan_and_execution(network, tmp_path, preset, mode, target, weights):
    inputs, output, sizes, arrays = network
    opts = dict(preset=preset, seed=7, rate_enabled=False, slicing_mode=mode,
                target_size=target, flops_weight=weights[0], read_write_weight=weights[1])
    report = arctn_schedule(inputs, output, sizes, use_ssa=True, **opts)
    assert len(report["path"]) == 2
    assert not ({"multi_start", "circuit_schedule", "mem_report", "generator_best_source",
                 "chosen_path_stage"} & report.keys())
    assert report["flops_weight"] == weights[0]
    assert report["read_write_weight"] == weights[1]
    assert report["sliced"] == (target is not None)
    if target:
        assert report["sliced_log2_max_size"] <= np.log2(target)
    plan = arctn_plan(inputs, output, sizes, **opts)
    filename = tmp_path / "plan.json"
    plan.save(filename)
    loaded = ArcTNExecutionPlan.load(filename)
    expected = arrays[0] @ arrays[1] @ arrays[2]
    for backend in ("native", "numpy"):
        np.testing.assert_allclose(loaded.execute(arrays, backend=backend), expected)


@engine_required
def test_quimb_tree_integration(network):
    import quimb.tensor as qtn

    inputs, output, _, arrays = network
    tn = qtn.TensorNetwork([qtn.Tensor(a, inds=inds) for a, inds in zip(arrays, inputs)])
    optimizer = ArcTNOptimizer(preset="light", target_size=4)
    result = tn.contract(all, output_inds=output, optimize=optimizer)
    np.testing.assert_allclose(result.data, arrays[0] @ arrays[1] @ arrays[2])


@engine_required
@pytest.mark.parametrize("backend", ["native", "numpy"])
def test_contract_with_slicing(network, backend):
    inputs, output, sizes, arrays = network
    result = arctn_contract(inputs, output, sizes, arrays, preset="light",
                            target_size=4, backend=backend)
    np.testing.assert_allclose(result, arrays[0] @ arrays[1] @ arrays[2])
