"""Output-index slicing with explicit plans and optional engine planning.

Explicit execution tests need only the public extension and arctn[execution].
Their references are complete NumPy einsum expressions, not a reproduction
of the slice enumeration or output-block assembly code.
"""

import inspect
import os
from pathlib import Path
import shutil
import subprocess
import sys

import arctn
import numpy as np
import pytest

from arctn import (
    ArcTNExecutionPlan,
    ArcTNOptimizer,
    arctn_contract,
    arctn_plan,
    arctn_schedule,
    arctn_tree,
)
from arctn import arctn_py


DTYPES = (np.float32, np.float64, np.complex64, np.complex128)
ENTRY_POINTS = (arctn_contract, arctn_plan, arctn_schedule, arctn_tree, ArcTNOptimizer)


def _case(name, equation, dimensions, sliced_labels, ssa_path):
    terms, output = equation.split("->")
    terms = terms.split(",")
    labels = {label: index for index, label in enumerate(dict.fromkeys("".join(terms)))}
    return {
        "name": name,
        "equation": equation,
        "inputs": [list(labels[label] for label in term) for term in terms],
        "output": [labels[label] for label in output],
        "sizes": {labels[label]: dimension for label, dimension in dimensions.items()},
        "sliced_legs": [labels[label] for label in sliced_labels],
        "ssa_path": [list(pair) for pair in ssa_path],
    }


CASES = (
    _case("permuted output", "ab,bc->ca", dict(a=2, b=3, c=4), "c", [(0, 1)]),
    _case("two output legs", "ab,bc->ca", dict(a=2, b=3, c=4), "ac", [(0, 1)]),
    _case("internal and output legs", "ab,bc->ca", dict(a=2, b=3, c=4), "cb", [(0, 1)]),
    _case(
        "output shared by three tensors", "ab,bc,bd->dbca",
        dict(a=2, b=3, c=4, d=2), "bd", [(0, 1), (3, 2)],
    ),
    _case(
        "internal hyperedge and two output legs", "ab,bc,bd->dac",
        dict(a=2, b=3, c=4, d=2), "dba", [(0, 1), (3, 2)],
    ),
    _case("unary reduction with empty SSA", "abc->ca", dict(a=2, b=3, c=4), "cb", []),
    _case("unary scalar blocks", "ab->ba", dict(a=2, b=3), "ab", []),
    _case("scalar input", ",a->a", dict(a=5), "a", [(0, 1)]),
    _case("internal-only regression", "ab,bc->ca", dict(a=2, b=3, c=4), "b", [(0, 1)]),
)
PLANNING_CASE = CASES[0]
TARGET = 2  # The full output has eight elements.


def _arrays(spec, dtype, layout="c"):
    rng = np.random.default_rng(871)
    arrays = []
    for term in spec["inputs"]:
        shape = tuple(spec["sizes"][leg] for leg in term)
        storage_shape = tuple(2 * dimension for dimension in shape) if layout == "strided" else shape
        data = rng.standard_normal(storage_shape)
        if np.issubdtype(dtype, np.complexfloating):
            data = data + 1j * rng.standard_normal(storage_shape)
        array = np.asarray(data, dtype=dtype)
        if layout == "strided" and shape:
            array = array[tuple(slice(None, None, -2) for _ in shape)]
        elif layout == "fortran" and shape:
            array = np.asfortranarray(array)
        arrays.append(array)
    return arrays


def _artifact(spec):
    # This is the public v2 plan format, with a caller-supplied SSA order and
    # exact slice set. It makes no assertion about any planner's search result.
    tensors = ";".join(",".join(map(str, term)) for term in spec["inputs"])
    output = ",".join(map(str, spec["output"]))
    sizes = sorted(spec["sizes"].items())
    dimensions = ",".join(f"{leg}={dimension}" for leg, dimension in sizes)
    return {
        "schema": "arctn-execution-plan",
        "schema_version": 2,
        "path_format": "ssa-v1",
        "network_canon": f"t:{tensors}|o:{output}|d:{dimensions}",
        "network": {
            "inputs": spec["inputs"],
            "output": spec["output"],
            "size_dict": [list(item) for item in sizes],
        },
        "ssa_path": spec["ssa_path"],
        "sliced": {"legs": spec["sliced_legs"]},
        "metrics": {},
        "planning": {},
        "provenance": {"source": "explicit-test-plan"},
    }


def _assert_result(actual, reference, dtype):
    assert actual.shape == reference.shape
    assert actual.dtype == np.dtype(dtype)
    single_precision = np.dtype(dtype) in (np.dtype(np.float32), np.dtype(np.complex64))
    tolerance = 2e-4 if single_precision else 1e-11
    np.testing.assert_allclose(actual, reference, rtol=tolerance, atol=tolerance)


@pytest.fixture
def no_planning_engine(monkeypatch):
    monkeypatch.delenv("ARCTN_ENGINE_LIBRARY", raising=False)

    def unexpected_search(*args, **kwargs):
        pytest.fail("explicit sliced execution must not invoke the planning engine")

    monkeypatch.setattr(arctn_py, "optimize_auto", unexpected_search)
    monkeypatch.setattr(arctn_py, "optimize_auto_full", unexpected_search)


@pytest.mark.parametrize("spec", CASES, ids=lambda spec: spec["name"])
@pytest.mark.parametrize("dtype", DTYPES)
def test_explicit_native_output_slices_without_engine(spec, dtype, no_planning_engine):
    arrays = _arrays(spec, dtype)
    reference = np.einsum(spec["equation"], *arrays, optimize=False)
    actual = arctn_py.contract_sliced(
        spec["inputs"], spec["output"], spec["sizes"], arrays,
        spec["sliced_legs"], ssa_path=[tuple(pair) for pair in spec["ssa_path"]],
    )
    _assert_result(actual, reference, dtype)


@pytest.mark.parametrize("spec", CASES, ids=lambda spec: spec["name"])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("backend", ("native", "numpy"))
@pytest.mark.parametrize("layout", ("strided", "fortran"))
def test_explicit_plan_from_dict_save_load_without_engine(
    spec, dtype, backend, layout, tmp_path, no_planning_engine,
):
    arrays = _arrays(spec, dtype, layout)
    originals = [array.copy() for array in arrays]
    reference = np.einsum(spec["equation"], *arrays, optimize=False)
    plan = ArcTNExecutionPlan.from_dict(_artifact(spec))
    assert plan.output == tuple(spec["output"])
    assert plan.ssa_path == tuple(tuple(pair) for pair in spec["ssa_path"])
    assert plan.sliced_legs == tuple(spec["sliced_legs"])
    filename = tmp_path / "plan.json"
    plan.save(filename)
    loaded = ArcTNExecutionPlan.load(filename)
    assert loaded == plan
    for executable in (plan, loaded):
        _assert_result(executable.execute(arrays, backend=backend), reference, dtype)
    for array, original in zip(arrays, originals):
        np.testing.assert_array_equal(array, original)


def _call_entry(function, options):
    spec = PLANNING_CASE
    args = (spec["inputs"], spec["output"], spec["sizes"])
    if function is arctn_contract:
        return function(*args, _arrays(spec, np.float64), **options)
    if function is ArcTNOptimizer:
        return function(**options).search(*args)
    return function(*args, **options)


@pytest.mark.parametrize("function", ENTRY_POINTS, ids=lambda function: function.__name__)
def test_allow_output_slicing_defaults_to_false(function):
    assert inspect.signature(function).parameters["allow_output_slicing"].default is False


@pytest.mark.parametrize("function", ENTRY_POINTS, ids=lambda function: function.__name__)
@pytest.mark.parametrize("value", (None, 0, 1, "true"))
def test_allow_output_slicing_requires_bool(function, value, no_planning_engine):
    with pytest.raises(TypeError, match="allow_output_slicing"):
        _call_entry(function, {"target_size": TARGET, "allow_output_slicing": value})


@pytest.mark.parametrize("function", ENTRY_POINTS, ids=lambda function: function.__name__)
@pytest.mark.parametrize("enabled", (False, True))
@pytest.mark.parametrize("mode", ("fixed", "dynamic"))
def test_public_option_reaches_native_planning(function, enabled, mode, monkeypatch):
    class PlanningProbe(Exception):
        pass

    def capture_options(*args, **kwargs):
        assert kwargs.get("allow_output_slicing", False) is enabled
        assert kwargs["slicing_mode"] == mode
        assert kwargs["target_size"] == TARGET
        raise PlanningProbe

    monkeypatch.setattr(arctn_py, "optimize_auto_full", capture_options)
    with pytest.raises(PlanningProbe):
        _call_entry(function, {
            "target_size": TARGET, "slicing_mode": mode,
            "allow_output_slicing": enabled,
        })


def test_output_slicing_planning_reports_missing_engine(tmp_path):
    # Remove bundled engines even when this test suite is run from a full wheel.
    shutil.copytree(
        Path(arctn.__file__).parent, tmp_path / "arctn",
        ignore=shutil.ignore_patterns("_lib", "__pycache__"),
    )
    env = dict(os.environ)
    env.pop("ARCTN_ENGINE_LIBRARY", None)
    env["PYTHONPATH"] = str(tmp_path)
    code = """
import os
from pathlib import Path
import arctn
assert Path(arctn.__file__).parent == Path.cwd() / 'arctn'
assert 'ARCTN_ENGINE_LIBRARY' not in os.environ
try:
    arctn.arctn_plan([[0, 1], [1, 2]], [2, 0], {0: 2, 1: 3, 2: 4},
                     preset='light', target_size=2, allow_output_slicing=True)
except ValueError as error:
    assert 'ARCTN_ENGINE_LIBRARY' in str(error), str(error)
else:
    raise AssertionError('planning without an engine was accepted')
"""
    subprocess.run([sys.executable, "-c", code], env=env, cwd=tmp_path,
                   check=True, timeout=30)


engine_required = pytest.mark.skipif(
    not os.environ.get("ARCTN_ENGINE_LIBRARY"), reason="planning engine not supplied",
)


@engine_required
@pytest.mark.parametrize("preset", ("light", "heavy"))
@pytest.mark.parametrize("mode", ("fixed", "dynamic"))
def test_engine_output_slice_planning_and_execution(preset, mode, tmp_path):
    spec = PLANNING_CASE
    args = (spec["inputs"], spec["output"], spec["sizes"])
    options = dict(preset=preset, seed=5, target_size=TARGET,
                   slicing_mode=mode, allow_output_slicing=True)
    report = arctn_schedule(*args, use_ssa=True, **options)
    assert report["sliced"]
    assert set(report["sliced_legs"]).intersection(spec["output"])
    assert report["sliced_log2_max_size"] <= np.log2(TARGET) + 1e-9
    plan = arctn_plan(*args, **options)
    filename = tmp_path / "searched-plan.json"
    plan.save(filename)
    loaded = ArcTNExecutionPlan.load(filename)
    assert loaded == plan
    for dtype in DTYPES:
        arrays = _arrays(spec, dtype, "strided")
        reference = np.einsum(spec["equation"], *arrays, optimize=False)
        for backend in ("native", "numpy"):
            _assert_result(loaded.execute(arrays, backend=backend), reference, dtype)
            actual = arctn_contract(*args, arrays, backend=backend, **options)
            _assert_result(actual, reference, dtype)


@engine_required
@pytest.mark.parametrize("mode", ("fixed", "dynamic"))
def test_engine_tree_and_optimizer_preserve_output_blocks(mode):
    spec = PLANNING_CASE
    args = (spec["inputs"], spec["output"], spec["sizes"])
    arrays = _arrays(spec, np.complex128, "strided")
    reference = np.einsum(spec["equation"], *arrays, optimize=False)
    options = dict(preset="light", seed=5, target_size=TARGET,
                   slicing_mode=mode, allow_output_slicing=True)
    tree = arctn_tree(*args, **options)
    assert tree.sliced_inds
    _assert_result(tree.contract(arrays), reference, np.complex128)
    optimizer = ArcTNOptimizer(**options)
    optimized_tree = optimizer.search(*args)
    assert optimized_tree.sliced_inds
    _assert_result(optimized_tree.contract(arrays), reference, np.complex128)
    with pytest.raises(NotImplementedError):
        optimizer(*args)


@engine_required
@pytest.mark.parametrize("function", ENTRY_POINTS, ids=lambda function: function.__name__)
@pytest.mark.parametrize("mode", ("fixed", "dynamic"))
@pytest.mark.parametrize("explicit_false", (False, True))
def test_engine_default_output_protection(function, mode, explicit_false):
    options = dict(preset="light", seed=5, target_size=TARGET, slicing_mode=mode)
    if explicit_false:
        options["allow_output_slicing"] = False
    with pytest.raises(ValueError):
        _call_entry(function, options)
