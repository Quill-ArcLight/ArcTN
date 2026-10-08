"""Plan input snapshots, with public execution and optional engine planning."""

from collections.abc import Mapping
import os
from types import MappingProxyType

import numpy as np
import pytest

from arctn import _to_int_labels, arctn_plan, arctn_py


@pytest.fixture(params=["supplied-path", "engine"])
def planner(request, monkeypatch):
    if request.param == "engine":
        if not os.environ.get("ARCTN_ENGINE_LIBRARY"):
            pytest.skip("planning engine not supplied")
        return

    def supplied_path(inputs, output, sizes, **options):
        # Keep the public Python normalization and native plan validation real.
        # Only replace path search, which needs the separately supplied engine.
        assert options["use_ssa"] is True
        assert len(inputs) in (1, 2)
        return {"path": [] if len(inputs) == 1 else [(0, 1)]}

    monkeypatch.setattr(arctn_py, "optimize_auto_full", supplied_path)


def _iterable_form(form, inputs, output, sizes):
    if form == "tuples":
        return tuple(map(tuple, inputs)), tuple(output), MappingProxyType(sizes)
    if form == "output-iterator":
        return inputs, iter(output), sizes
    if form == "outer-generator":
        return (term for term in inputs), output, sizes
    if form == "term-iterators":
        return [iter(term) for term in inputs], output, sizes
    if form == "all-iterators":
        return (iter(term) for term in inputs), iter(output), MappingProxyType(sizes)
    assert form == "lists"
    return inputs, output, sizes


@pytest.mark.parametrize("backend", ["native", "numpy"])
@pytest.mark.parametrize("network", ["unary", "binary"])
@pytest.mark.parametrize("form", [
    "lists", "tuples", "output-iterator", "outer-generator",
    "term-iterators", "all-iterators",
])
def test_plan_iterables_preserve_network_and_execution(planner, backend, network, form):
    if network == "unary":
        inputs, output, sizes = [["a"]], ["a"], {"a": 2}
        arrays = [np.array([1., 2.])]
        expected = arrays[0]
        expected_inputs, expected_output = ((0,),), (0,)
        expected_labels = ("a",)
    else:
        inputs, output = [["a", "b"], ["b", "c"]], ["c", "a"]
        sizes = {"a": 2, "b": 3, "c": 4}
        arrays = [np.arange(6.).reshape(2, 3), np.arange(12.).reshape(3, 4)]
        expected = np.einsum("ab,bc->ca", *arrays)
        expected_inputs, expected_output = ((0, 1), (1, 2)), (2, 0)
        expected_labels = ("a", "b", "c")

    plan = arctn_plan(
        *_iterable_form(form, inputs, output, sizes),
        preset="light", seed=0, rate_enabled=False,
    )
    assert plan.inputs == expected_inputs
    assert plan.output == expected_output
    assert plan.leg_labels == expected_labels
    result = plan.execute(arrays, backend=backend)
    # In particular, an exhausted unary output iterator used to silently turn
    # the vector [1, 2] into the scalar sum 3 on both execution backends.
    assert result.shape == expected.shape
    np.testing.assert_array_equal(result, expected)


class SingleReadMapping(Mapping):
    """Detect repeated reads of caller-owned dimension values."""

    def __init__(self, values):
        self.values = values
        self.reads = {key: 0 for key in values}

    def __getitem__(self, key):
        self.reads[key] += 1
        assert self.reads[key] == 1
        return self.values[key]

    def __iter__(self):
        return iter(self.values)

    def __len__(self):
        return len(self.values)


def test_plan_snapshots_size_mapping_once(planner):
    sizes = SingleReadMapping({"a": 2})
    plan = arctn_plan([["a"]], ["a"], sizes, preset="light", rate_enabled=False)
    assert dict(plan.size_dict) == {0: 2}
    assert sizes.reads == {"a": 1}


class MappingWithoutKeys:
    """A supported mapping-like object without the optional keys() method."""

    def __init__(self, values):
        self.values = values

    def __getitem__(self, key):
        return self.values[key]

    def __iter__(self):
        return iter(self.values)

    def items(self):
        return self.values.items()


def test_plan_preserves_mapping_without_keys(planner):
    sizes = MappingWithoutKeys({"a": 2})
    # This mapping already satisfies the shared label normalizer. Snapshotting
    # must not additionally require keys(), as dict(sizes) would do here.
    assert _to_int_labels([["a"]], ["a"], sizes)[2] == {0: 2}
    plan = arctn_plan([["a"]], ["a"], sizes, preset="light", rate_enabled=False)
    assert dict(plan.size_dict) == {0: 2}
    np.testing.assert_array_equal(plan.execute([np.array([1., 2.])]), [1., 2.])


@pytest.mark.parametrize("sizes,error,match", [
    ([("a", 2)], TypeError, "size_dict"),
    ({}, ValueError, "size_dict"),
    ({"a": 2, "unused": 3}, ValueError, "size_dict"),
    ({"a": True}, ValueError, "bool"),
    ({"a": 0}, ValueError, "正整数"),
    ({"a": 1.5}, ValueError, "正整数"),
])
def test_plan_preserves_size_validation(sizes, error, match, monkeypatch):
    def unexpected_planning(*args, **kwargs):
        pytest.fail("invalid dimensions must be rejected before planning")

    monkeypatch.setattr(arctn_py, "optimize_auto_full", unexpected_planning)
    with pytest.raises(error, match=match):
        arctn_plan([["a"]], ["a"], sizes, preset="light")


@pytest.mark.skipif(
    not os.environ.get("ARCTN_ENGINE_LIBRARY"), reason="planning engine not supplied",
)
@pytest.mark.parametrize("backend", ["native", "numpy"])
def test_plan_combined_iterators_with_output_slicing(backend):
    inputs, output = [["a", "b"], ["b", "c"]], ["c", "a"]
    sizes = {"a": 2, "b": 3, "c": 4}
    arrays = [np.arange(6.).reshape(2, 3), np.arange(12.).reshape(3, 4)]
    plan = arctn_plan(
        *_iterable_form("all-iterators", inputs, output, sizes),
        preset="light", seed=0, rate_enabled=False, target_size=2,
        allow_output_slicing=True,
    )
    assert set(plan.output).intersection(plan.sliced_legs)
    actual = plan.execute(arrays, backend=backend)
    expected = np.einsum("ab,bc->ca", *arrays)
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(actual, expected)
