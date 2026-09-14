"""Stable, backend-neutral contraction plans for the Python interface."""

from __future__ import annotations

import json
import math
import os
import operator
import tempfile
import time
from pathlib import Path
from types import MappingProxyType

from . import (
    ArcTNCompiledContraction,
    __version__,
    _compile_external_sliced_expression,
    _contraction_tree_from_info,
    _execute_external_sliced_expression,
    _execution_dtype_name,
    _prepare_contract_arrays,
    _ssa_to_linear_path,
    _to_int_labels,
    _validate_auto_preset,
    _validate_bool,
    _validate_execution_backend,
    _validate_max_time,
    _validate_planner_weights,
    _validate_slicing_mode,
    _validate_target_size,
    arctn_schedule,
    arctn_py as _rs,
)


_SCHEMA = "arctn-execution-plan"
_SCHEMA_VERSION = 2
_PATH_FORMAT = "ssa-v1"
_MEMORY_CONSTRAINT = "max_intermediate_elements_per_slice"
_METRIC_KEYS = (
    "log10_flops",
    "log2_max_size",
    "log2_max_contraction_size",
    "log2_total_size",
    "log2_read_write",
    "log2_peak_size",
    "log2_n_slices",
    "sliced_log10_flops_total",
    "sliced_log2_max_size",
    "sliced_log2_max_contraction_size",
    "sliced_log2_peak_size",
    "max_intermediate_log2_elements_per_slice",
)
_PLANNING_KEYS = (
    "preset",
    "method",
    "seed",
    "max_time",
    "rate_enabled",
    "planner_objective",
    "flops_weight",
    "read_write_weight",
    "slicing_mode",
)


class PlanValidationError(ValueError):
    """Raised when an execution plan is incomplete or internally inconsistent."""


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value):
    if isinstance(value, MappingProxyType):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _network_canon(inputs, output, size_items):
    tensors = ";".join(
        ",".join(str(leg) for leg in tensor)
        for tensor in inputs
    )
    output_text = ",".join(str(leg) for leg in output)
    dimensions = ",".join(f"{leg}={dimension}" for leg, dimension in size_items)
    return f"t:{tensors}|o:{output_text}|d:{dimensions}"


def _original_labels(label_of):
    labels = [None] * len(label_of)
    for label, leg in label_of.items():
        labels[leg] = label
    return tuple(labels)


def _dense_network(inputs, output, size_dict):
    inputs_i, output_i, size_i, label_of = _to_int_labels(
        inputs, output, size_dict
    )
    return (
        tuple(tuple(leg for leg in tensor) for tensor in inputs_i),
        tuple(output_i),
        tuple(sorted(size_i.items())),
        _original_labels(label_of),
    )


class ArcTNExecutionPlan:
    """An immutable contraction path and exact slice plan.

    The persisted v2 artifact contains a dense integer network, so execution
    does not depend on serializing arbitrary Python label objects.  A plan made
    in memory retains those original labels for inspection; a loaded plan uses
    the embedded dense labels.
    """

    __slots__ = (
        "_inputs",
        "_output",
        "_size_dict",
        "_ssa_path",
        "_sliced_legs",
        "_target_size",
        "_metrics",
        "_planning",
        "_provenance",
        "_leg_labels",
        "_network_canon",
        "_artifact",
    )
    __hash__ = None

    def __init__(self, *args, **kwargs):
        raise TypeError(
            "use arctn_plan(), ArcTNExecutionPlan.from_schedule(), or "
            "ArcTNExecutionPlan.load()"
        )

    @classmethod
    def _from_artifact(cls, artifact, *, leg_labels=None):
        try:
            encoded = json.dumps(
                artifact,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
            _rs._validate_execution_plan_v2_json(encoded)
            artifact = json.loads(encoded)
        except (TypeError, ValueError) as exc:
            raise PlanValidationError(f"invalid execution plan: {exc}") from exc

        if "metrics" not in artifact:
            artifact["metrics"] = {
                key: artifact[key]
                for key in _METRIC_KEYS
                if key in artifact
            }
        if "planning" not in artifact:
            artifact["planning"] = {
                key: artifact[key]
                for key in _PLANNING_KEYS
                if key in artifact
            }
        if "provenance" not in artifact:
            artifact["provenance"] = {
                key: artifact[key]
                for key in ("arctn_version", "source")
                if key in artifact
            }
        for name in ("metrics", "planning", "provenance"):
            if not isinstance(artifact[name], dict):
                raise PlanValidationError(
                    f"execution plan {name} must be an object"
                )

        network = artifact["network"]
        inputs = tuple(tuple(tensor) for tensor in network["inputs"])
        output = tuple(network["output"])
        size_dict = MappingProxyType(
            {leg: dimension for leg, dimension in network["size_dict"]}
        )
        if leg_labels is None:
            leg_labels = artifact.get("network_leg_labels")
        if leg_labels is None:
            leg_labels = tuple(size_dict)
        else:
            leg_labels = tuple(leg_labels)

        value = object.__new__(cls)
        object.__setattr__(value, "_inputs", inputs)
        object.__setattr__(value, "_output", output)
        object.__setattr__(value, "_size_dict", size_dict)
        object.__setattr__(
            value,
            "_ssa_path",
            tuple(tuple(pair) for pair in artifact["ssa_path"]),
        )
        object.__setattr__(
            value, "_sliced_legs", tuple(artifact["sliced"]["legs"])
        )
        object.__setattr__(value, "_target_size", artifact.get("target_size"))
        object.__setattr__(value, "_metrics", _freeze(artifact["metrics"]))
        object.__setattr__(value, "_planning", _freeze(artifact["planning"]))
        object.__setattr__(value, "_provenance", _freeze(artifact["provenance"]))
        object.__setattr__(value, "_leg_labels", leg_labels)
        object.__setattr__(value, "_network_canon", artifact["network_canon"])
        object.__setattr__(value, "_artifact", _freeze(artifact))
        return value

    def __setattr__(self, name, value):
        raise AttributeError("ArcTNExecutionPlan is immutable")

    @property
    def schema(self):
        return _SCHEMA

    @property
    def schema_version(self):
        return _SCHEMA_VERSION

    @property
    def path_format(self):
        return _PATH_FORMAT

    @property
    def inputs(self):
        return self._inputs

    @property
    def output(self):
        return self._output

    @property
    def size_dict(self):
        return self._size_dict

    @property
    def ssa_path(self):
        return self._ssa_path

    @property
    def sliced_legs(self):
        return self._sliced_legs

    @property
    def target_size(self):
        return self._target_size

    @property
    def metrics(self):
        return self._metrics

    @property
    def planning(self):
        return self._planning

    @property
    def provenance(self):
        return self._provenance

    @property
    def leg_labels(self):
        return self._leg_labels

    @property
    def network_canon(self):
        return self._network_canon

    @property
    def is_sliced(self):
        return bool(self._sliced_legs)

    @classmethod
    def from_schedule(cls, report, *, inputs, output, size_dict):
        """Create a plan from an SSA-form :func:`arctn_schedule` report."""
        if not isinstance(report, dict):
            raise TypeError("report must be a mapping returned by arctn_schedule")
        if "path" not in report:
            raise PlanValidationError("schedule report is missing path")
        if report.get("path_format") != _PATH_FORMAT:
            raise PlanValidationError(
                "from_schedule requires arctn_schedule(..., use_ssa=True)"
            )
        inputs_i, output_i, size_items, leg_labels = _dense_network(
            inputs, output, size_dict
        )
        sliced_legs = report.get("sliced_legs") or ()
        metrics = (
            {
                key: report[key]
                for key in _METRIC_KEYS
                if key in report
                and not (
                    isinstance(report[key], float)
                    and not math.isfinite(report[key])
                )
            }
            if report["path"]
            else {}
        )
        planning = {
            key: report[key]
            for key in _PLANNING_KEYS
            if key in report
        }
        provenance = {
            "arctn_version": __version__,
            "source": "arctn_schedule",
        }
        artifact = {
            "schema": _SCHEMA,
            "schema_version": _SCHEMA_VERSION,
            "path_format": _PATH_FORMAT,
            "network_canon": _network_canon(inputs_i, output_i, size_items),
            "network": {
                "inputs": [list(tensor) for tensor in inputs_i],
                "output": list(output_i),
                "size_dict": [list(item) for item in size_items],
            },
            "ssa_path": [list(pair) for pair in report["path"]],
            "sliced": {"legs": list(sliced_legs)},
            "metrics": metrics,
            "planning": planning,
            "provenance": provenance,
        }
        target_size = report.get("target_size")
        if target_size is not None:
            artifact.update(
                {
                    "target_size": target_size,
                    "memory_target_log2_elements": math.log2(target_size),
                    "memory_constraint_metric": _MEMORY_CONSTRAINT,
                }
            )
        return cls._from_artifact(artifact, leg_labels=leg_labels)

    @classmethod
    def from_dict(cls, artifact):
        """Validate and load one self-contained v2 artifact mapping."""
        if not isinstance(artifact, dict):
            raise PlanValidationError("execution plan root must be an object")
        return cls._from_artifact(artifact)

    @classmethod
    def load(cls, path):
        """Load and validate one self-contained v2 JSON plan."""
        try:
            with open(os.fspath(path), "r", encoding="utf-8") as stream:
                artifact = json.load(stream)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise PlanValidationError(f"cannot load execution plan: {exc}") from exc
        return cls.from_dict(artifact)

    def to_dict(self):
        """Return a portable JSON-compatible v2 artifact mapping."""
        return _thaw(self._artifact)

    def save(self, path):
        """Atomically write the portable v2 JSON representation."""
        destination = Path(path)
        parent = destination.parent
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
                json.dump(
                    self.to_dict(),
                    stream,
                    ensure_ascii=False,
                    allow_nan=False,
                    indent=2,
                    sort_keys=True,
                )
                stream.write("\n")
            os.replace(temporary, destination)
        except (OSError, TypeError, ValueError) as exc:
            if temporary is not None:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass
            raise PlanValidationError(f"cannot save execution plan: {exc}") from exc
        return destination

    def validate(self, inputs=None, output=None, size_dict=None, *, arrays=None):
        """Validate an optional caller network and array metadata against the plan."""
        provided_network = (inputs is not None, output is not None, size_dict is not None)
        if any(provided_network) and not all(provided_network):
            raise TypeError(
                "inputs, output, and size_dict must be provided together"
            )
        if all(provided_network):
            candidate = _dense_network(inputs, output, size_dict)
            if candidate[:3] != (
                self._inputs,
                self._output,
                tuple(self._size_dict.items()),
            ):
                raise PlanValidationError(
                    "caller network does not match the embedded plan network"
                )

        if arrays is not None:
            try:
                values = tuple(arrays)
            except TypeError as exc:
                raise PlanValidationError("arrays must be a sequence") from exc
            if len(values) != len(self._inputs):
                raise PlanValidationError(
                    f"arrays length {len(values)} does not match "
                    f"{len(self._inputs)} plan inputs"
                )
            dtype_name = None
            for index, (array, tensor) in enumerate(zip(values, self._inputs)):
                if not hasattr(array, "shape") or not hasattr(array, "dtype"):
                    raise PlanValidationError(
                        f"arrays[{index}] must provide shape and dtype"
                    )
                try:
                    current_dtype = _execution_dtype_name(array.dtype)
                    actual_shape = tuple(operator.index(dim) for dim in array.shape)
                except (TypeError, ValueError) as exc:
                    raise PlanValidationError(
                        f"arrays[{index}] has invalid shape or dtype: {exc}"
                    ) from exc
                expected_shape = tuple(self._size_dict[leg] for leg in tensor)
                if actual_shape != expected_shape:
                    raise PlanValidationError(
                        f"arrays[{index}] shape {actual_shape} does not match "
                        f"{expected_shape}"
                    )
                if dtype_name is None:
                    dtype_name = current_dtype
                elif current_dtype != dtype_name:
                    raise PlanValidationError("all arrays must use the same dtype")
        return None

    def to_linear_path(self):
        """Return the opt_einsum recycled linear path."""
        return tuple(_ssa_to_linear_path(self._ssa_path, len(self._inputs)))

    def to_tree(self):
        """Build a cotengra ``ContractionTree`` without running search again."""
        label_of = {leg: leg for leg in self._size_dict}
        tree, _ = _contraction_tree_from_info(
            self._inputs,
            self._output,
            self._size_dict,
            label_of,
            {
                "path": self._ssa_path,
                "sliced_legs": self._sliced_legs,
            },
        )
        return tree

    def compile(self, *, backend="native"):
        """Compile an unsliced plan for repeated in-process execution."""
        if self._sliced_legs:
            raise PlanValidationError(
                "sliced plans cannot be compiled by ArcTNCompiledContraction; "
                "call plan.execute(...) or plan.to_tree()"
            )
        return ArcTNCompiledContraction.compile(
            self._inputs,
            self._output,
            self._size_dict,
            ssa_path=self._ssa_path,
            backend=backend,
        )

    def execute(self, arrays, *, backend="native", return_info=False):
        """Execute this exact path and slice set without running the planner."""
        backend = _validate_execution_backend(backend)
        try:
            arrays = tuple(arrays)
        except TypeError as exc:
            raise PlanValidationError("arrays must be a sequence") from exc
        self.validate(arrays=arrays)
        if not self._sliced_legs:
            compiled = self.compile(backend=backend)
            result = compiled.execute(arrays)
            if not return_info:
                return result
            stats = compiled.stats()
            stats.update(
                {
                    "plan_schema": _SCHEMA,
                    "plan_schema_version": _SCHEMA_VERSION,
                    "execution_backend": backend,
                    "sliced_legs": [],
                }
            )
            return result, stats

        setup_started = time.perf_counter()
        values = _prepare_contract_arrays(
            self._inputs, self._size_dict, arrays, backend=backend
        )
        expression_compile_wall_s = None
        if backend == "native":
            setup_wall_s = time.perf_counter() - setup_started
            execution_started = time.perf_counter()
            result = _rs.contract_sliced(
                self._inputs,
                self._output,
                dict(self._size_dict),
                values,
                self._sliced_legs,
                ssa_path=self._ssa_path,
            )
            dispatch_wall_s = time.perf_counter() - execution_started
            implementation = "arctn.contract_sliced"
        else:
            compile_started = time.perf_counter()
            expression, index_plans, slice_dimensions = (
                _compile_external_sliced_expression(
                    self._inputs,
                    self._output,
                    self._size_dict,
                    sliced_legs=self._sliced_legs,
                    ssa_path=self._ssa_path,
                )
            )
            expression_compile_wall_s = time.perf_counter() - compile_started
            setup_wall_s = time.perf_counter() - setup_started
            execution_started = time.perf_counter()
            result = _execute_external_sliced_expression(
                expression,
                values,
                index_plans,
                slice_dimensions,
                backend=backend,
            )
            dispatch_wall_s = time.perf_counter() - execution_started
            implementation = "arctn.sliced_contract_expression"
        if not return_info:
            return result
        synchronous = backend in {"native", "numpy"}
        return result, {
            "plan_schema": _SCHEMA,
            "plan_schema_version": _SCHEMA_VERSION,
            "execution_backend": backend,
            "execution_implementation": implementation,
            "expression_compile_wall_s": expression_compile_wall_s,
            "execution_setup_wall_s": setup_wall_s,
            "execution_dispatch_wall_s": dispatch_wall_s,
            "execution_wall_s": setup_wall_s + dispatch_wall_s if synchronous else None,
            "execution_timing_scope": (
                "synchronous_host_wall"
                if synchronous
                else "external_backend_completion_not_verified"
            ),
            "sliced_legs": list(self._sliced_legs),
        }

    def __eq__(self, other):
        if not isinstance(other, ArcTNExecutionPlan):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    def __repr__(self):
        return (
            "ArcTNExecutionPlan("
            f"n_inputs={len(self._inputs)}, n_steps={len(self._ssa_path)}, "
            f"n_sliced_legs={len(self._sliced_legs)}, "
            f"target_size={self._target_size!r})"
        )


def arctn_plan(
    inputs,
    output,
    size_dict,
    *,
    preset="heavy",
    seed=0,
    target_size=None,
    slicing_mode="fixed",
    max_time=None,
    flops_weight=1.0,
    read_write_weight=64.0,
    rate_enabled=True,
):
    """Plan once and return an immutable backend-neutral execution plan."""
    preset = _validate_auto_preset(preset)
    max_time = _validate_max_time(max_time)
    flops_weight, read_write_weight = _validate_planner_weights(
        flops_weight, read_write_weight
    )
    target_size = _validate_target_size(target_size)
    slicing_mode = _validate_slicing_mode(slicing_mode, target_size)
    rate_enabled = _validate_bool(rate_enabled, "rate_enabled")
    report = arctn_schedule(
        inputs,
        output,
        size_dict,
        preset=preset,
        seed=seed,
        target_size=target_size,
        slicing_mode=slicing_mode,
        max_time=max_time,
        use_ssa=True,
        flops_weight=flops_weight,
        read_write_weight=read_write_weight,
        rate_enabled=rate_enabled,
    )
    return ArcTNExecutionPlan.from_schedule(
        report,
        inputs=inputs,
        output=output,
        size_dict=size_dict,
    )
