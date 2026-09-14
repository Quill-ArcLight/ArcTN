//! Python bindings for ArcTN.
//!
//! The extension provides planning, slicing, execution, and diagnostic APIs.
//!
//! Python maps arbitrary labels to ArcTN's `u32` leg identifiers before
//! entering this module. Path-only calls return opt_einsum-style recycled
//! linear paths by default and return SSA paths when `use_ssa=True`.
//!
//! Build the wheel from `pybind/` with `python -m maturin build --release`.
//! The native module is installed as `arctn.arctn_py`.

use std::collections::{HashMap, HashSet};

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyDict};

use arctn::network::TensorNetwork;

use arctn::auto::{auto_path_preset_with_objective, AutoPreset, SlicingMode};

use arctn::PlannerObjective;

fn pick_auto_preset(label: &str) -> PyResult<AutoPreset> {
    AutoPreset::from_label(label).ok_or_else(|| {
        PyValueError::new_err(format!("preset 只支持 'light' 或 'heavy'，收到 {label:?}"))
    })
}

fn pick_slicing_mode(label: &str) -> PyResult<SlicingMode> {
    SlicingMode::from_label(label).ok_or_else(|| {
        PyValueError::new_err(format!(
            "slicing_mode 只支持 'fixed' 或 'dynamic'，收到 {label:?}"
        ))
    })
}

fn validate_max_time(max_time: Option<f64>) -> PyResult<()> {
    if max_time.is_some_and(|secs| !secs.is_finite() || secs <= 0.0) {
        return Err(PyValueError::new_err(
            "max_time must be finite and greater than zero",
        ));
    }
    Ok(())
}

fn pick_planner_objective(flops_weight: f64, read_write_weight: f64) -> PyResult<PlannerObjective> {
    PlannerObjective::new(flops_weight, read_write_weight).map_err(PyValueError::new_err)
}

/// Preserve the public `target_size` as an exact element count.
///
/// Converting it to a floating-point log2 value would lose the distinction
/// between adjacent integers above 2^53 and could accept an oversized
/// intermediate.
#[derive(Clone, Copy, Debug)]
struct TargetRequest(usize);

impl TargetRequest {
    fn log2_elements(self) -> f64 {
        (self.0 as f64).log2()
    }

    fn exact_elements(self) -> usize {
        self.0
    }
}

/// Parse the product API's exact intermediate-element limit.
fn pick_target_request(target_size: Option<Bound<'_, PyAny>>) -> PyResult<Option<TargetRequest>> {
    let Some(value) = target_size else {
        return Ok(None);
    };
    if value.is_instance_of::<PyBool>() {
        return Err(PyTypeError::new_err(
            "target_size 必须是严格正整数，不能是 bool",
        ));
    }
    let value = value.extract::<usize>()?;
    if value == 0 {
        return Err(PyValueError::new_err("target_size 必须大于 0"));
    }
    Ok(Some(TargetRequest(value)))
}

/// Add the same planner-objective provenance used by the CLI save contract.
fn set_planner_metadata(
    d: &Bound<'_, pyo3::types::PyDict>,
    stats: &arctn::PathStats,
    slice: Option<&arctn::slice::SliceResult>,
    objective: PlannerObjective,
) -> PyResult<()> {
    let objective_log2 = slice.map_or_else(
        || objective.score_path_log2(stats),
        |sr| objective.score_sliced_log2(&sr.per_slice, sr.log2_n_slices),
    );
    let log2_total_writes = slice.map_or(stats.log2_total_size, |sr| {
        sr.per_slice.log2_total_size + sr.log2_n_slices
    });
    let log2_read_write = slice.map_or(stats.log2_read_write, |sr| {
        sr.per_slice.log2_read_write + sr.log2_n_slices
    });
    d.set_item("planner_objective", objective.as_str())?;
    d.set_item("flops_weight", objective.flops_weight())?;
    d.set_item("read_write_weight", objective.read_write_weight())?;
    d.set_item("planner_objective_score_log2", objective_log2)?;
    d.set_item("planner_log2_read_write", log2_read_write)?;
    d.set_item("planner_log2_total_writes", log2_total_writes)?;
    Ok(())
}

fn set_memory_constraint_metadata<'py>(
    d: &Bound<'py, pyo3::types::PyDict>,
    py: Python<'py>,
    target: Option<TargetRequest>,
    max_intermediate_per_slice: Option<f64>,
) -> PyResult<()> {
    d.set_item(
        "memory_constraint_metric",
        "max_intermediate_elements_per_slice",
    )?;
    match target.map(TargetRequest::exact_elements) {
        Some(value) => d.set_item("target_size", value)?,
        None => d.set_item("target_size", py.None())?,
    }
    match target {
        Some(value) => d.set_item("memory_target_log2_elements", value.log2_elements())?,
        None => d.set_item("memory_target_log2_elements", py.None())?,
    }
    match max_intermediate_per_slice {
        Some(value) => d.set_item("max_intermediate_log2_elements_per_slice", value)?,
        None => d.set_item("max_intermediate_log2_elements_per_slice", py.None())?,
    }
    Ok(())
}

/// Convert an SSA path to an opt_einsum-style recycled linear path.
fn ssa_to_linear(ssa_path: &[(usize, usize)], n: usize) -> Vec<(usize, usize)> {
    let mut ids: Vec<usize> = (0..n).collect();
    let mut out = Vec::with_capacity(ssa_path.len());
    for (next_ssa, &(a, b)) in (n..).zip(ssa_path.iter()) {
        let ia = ids.binary_search(&a).expect("SSA 路径引用了已消费的张量");
        let ib = ids.binary_search(&b).expect("SSA 路径引用了已消费的张量");
        let (lo, hi) = if ia < ib { (ia, ib) } else { (ib, ia) };
        out.push((lo, hi));
        ids.remove(hi);
        ids.remove(lo);
        ids.push(next_ssa);
    }
    out
}

/// Convert a recycled linear path to SSA, the inverse of [`ssa_to_linear`].
fn linear_to_ssa(linear_path: &[(usize, usize)], n: usize) -> PyResult<Vec<(usize, usize)>> {
    let mut ids: Vec<usize> = (0..n).collect();
    let mut out = Vec::with_capacity(linear_path.len());
    for (next_ssa, &(a, b)) in (n..).zip(linear_path.iter()) {
        let (lo, hi) = if a < b { (a, b) } else { (b, a) };
        if hi >= ids.len() || lo == hi {
            return Err(PyValueError::new_err(format!(
                "linear 路径下标越界或自收缩: ({a},{b})，当前存活 {} 个张量。\
                 （如果这本来就是 SSA 路径，请改用 ssa_path= 传入）",
                ids.len()
            )));
        }
        out.push((ids[lo], ids[hi]));
        ids.remove(hi);
        ids.remove(lo);
        ids.push(next_ssa);
    }
    Ok(out)
}

/// Find a contraction path with Light or Heavy.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (inputs, output, size_dict, *, preset="heavy", seed=0, use_ssa=false,
                    max_time=None, flops_weight=1.0, read_write_weight=64.0))]
fn optimize_auto(
    py: Python<'_>,
    inputs: Vec<Vec<u32>>,
    output: Vec<u32>,
    size_dict: HashMap<u32, usize>,
    preset: &str,
    seed: u64,
    use_ssa: bool,
    max_time: Option<f64>,
    flops_weight: f64,
    read_write_weight: f64,
) -> PyResult<Vec<(usize, usize)>> {
    let preset = pick_auto_preset(preset)?;
    validate_max_time(max_time)?;
    let objective = pick_planner_objective(flops_weight, read_write_weight)?;
    let net = build_net(inputs, output, size_dict)?;
    let n = net.n_tensors();
    if n == 1 {
        return Ok(Vec::new());
    }
    let result = py
        .detach(|| auto_path_preset_with_objective(&net, preset, seed, max_time, objective))
        .map_err(PyValueError::new_err)?;
    Ok(if use_ssa {
        result.path
    } else {
        ssa_to_linear(&result.path, n)
    })
}

// Python execution helpers.

fn tensor_from_pyarray<T>(arr: &Bound<'_, numpy::PyArrayDyn<T>>) -> PyResult<arctn::DenseTensor<T>>
where
    T: numpy::Element + Clone + arctn::Scalar,
{
    use numpy::PyArrayMethods;
    use numpy::PyUntypedArrayMethods;
    if !arr.is_c_contiguous() {
        return Err(PyValueError::new_err(
            "输入数组必须是 C 连续的（先 np.ascontiguousarray）",
        ));
    }
    let shape: Vec<usize> = arr.shape().to_vec();
    let ro = arr.readonly();
    // The explicit layout check above makes this conversion deterministic.
    let data: Vec<T> = ro
        .as_slice()
        .map(|s| s.to_vec())
        .map_err(|_| PyValueError::new_err("输入数组必须是 C 连续的（先 np.ascontiguousarray）"))?;
    arctn::DenseTensor::try_from_data(shape, data)
        .map_err(|e| PyValueError::new_err(format!("输入数组布局非法: {e}")))
}

#[derive(Clone, Copy)]
enum PyArrayDtype {
    Float32,
    Float64,
    Complex64,
    Complex128,
}

/// Validate dtype, C layout and every network-derived shape before copying any ndarray payload.
///
/// Keeping this as a complete first pass matters at the FFI boundary: a mixed-dtype or malformed
/// tensor near the end of `arrays` must not be discovered only after all valid prefixes have
/// already been copied into Rust-owned `Vec`s.
fn preflight_network_pyarrays<T>(
    net: &TensorNetwork,
    arrays: &[Bound<'_, PyAny>],
    dtype_name: &str,
    error_context: &str,
) -> PyResult<()>
where
    T: numpy::Element + Clone + arctn::Scalar,
{
    use numpy::{PyArrayDyn, PyUntypedArrayMethods};

    for (tensor_index, (value, legs)) in arrays.iter().zip(&net.inputs).enumerate() {
        let array = value.cast::<PyArrayDyn<T>>().map_err(|_| {
            PyValueError::new_err(format!(
                "{error_context}: arrays[{tensor_index}] 的 dtype 与首张量不一致（预期 {dtype_name}）"
            ))
        })?;
        if !array.is_c_contiguous() {
            return Err(PyValueError::new_err(format!(
                "{error_context}: arrays[{tensor_index}] 必须是 C 连续的（先 np.ascontiguousarray）"
            )));
        }
        let expected: Vec<usize> = legs
            .iter()
            .map(|leg| {
                *net.size_dict
                    .get(leg)
                    .expect("已校验 TensorNetwork 的每条腿都应有维度")
            })
            .collect();
        if array.shape() != expected.as_slice() {
            return Err(PyValueError::new_err(format!(
                "{error_context}: arrays[{tensor_index}] shape {:?} 与网络预期 {:?} 不一致（输入数组布局非法）",
                array.shape(),
                expected
            )));
        }
    }
    Ok(())
}

fn detect_and_preflight_network_dtype(
    net: &TensorNetwork,
    arrays: &[Bound<'_, PyAny>],
    error_context: &str,
) -> PyResult<PyArrayDtype> {
    use numpy::PyArrayDyn;

    let first = arrays.first().expect("调用 dtype 分派前已校验 arrays 非空");
    if first
        .cast::<PyArrayDyn<num_complex::Complex<f64>>>()
        .is_ok()
    {
        preflight_network_pyarrays::<num_complex::Complex<f64>>(
            net,
            arrays,
            "complex128",
            error_context,
        )?;
        return Ok(PyArrayDtype::Complex128);
    }
    if first
        .cast::<PyArrayDyn<num_complex::Complex<f32>>>()
        .is_ok()
    {
        preflight_network_pyarrays::<num_complex::Complex<f32>>(
            net,
            arrays,
            "complex64",
            error_context,
        )?;
        return Ok(PyArrayDtype::Complex64);
    }
    if first.cast::<PyArrayDyn<f64>>().is_ok() {
        preflight_network_pyarrays::<f64>(net, arrays, "float64", error_context)?;
        return Ok(PyArrayDtype::Float64);
    }
    if first.cast::<PyArrayDyn<f32>>().is_ok() {
        preflight_network_pyarrays::<f32>(net, arrays, "float32", error_context)?;
        return Ok(PyArrayDtype::Float32);
    }
    Err(PyValueError::new_err(format!(
        "{error_context}: arrays 的 dtype 不支持；支持 float32 / float64 / complex64 / complex128"
    )))
}

fn copy_network_pyarrays<T>(
    arrays: &[Bound<'_, PyAny>],
    dtype_name: &str,
) -> PyResult<Vec<arctn::DenseTensor<T>>>
where
    T: numpy::Element + Clone + arctn::Scalar,
{
    use numpy::PyArrayDyn;

    arrays
        .iter()
        .enumerate()
        .map(|(tensor_index, value)| {
            let array = value.cast::<PyArrayDyn<T>>().map_err(|_| {
                PyValueError::new_err(format!(
                    "arrays[{tensor_index}] 的 dtype 与首张量不一致（预期 {dtype_name}）"
                ))
            })?;
            tensor_from_pyarray(array)
        })
        .collect()
}

/// A reusable exact contraction compiled from an explicitly SSA-form path.
///
/// The Python surface deliberately requires the `ssa_path=` keyword.  A
/// linear-recycled path uses a different coordinate system, so accepting an
/// unlabeled path here would make a long-lived compiled object easy to build
/// incorrectly.
#[pyclass(name = "CompiledContraction")]
struct PyCompiledContraction {
    net: TensorNetwork,
    compiled: arctn::CompiledContraction,
}

fn compiled_execute_typed<'py, T>(
    py: Python<'py>,
    compiled: &arctn::CompiledContraction,
    arrays: &[Bound<'py, PyAny>],
    dtype_name: &str,
) -> PyResult<Bound<'py, PyAny>>
where
    T: numpy::Element + Clone + arctn::Scalar,
{
    let tensors = copy_network_pyarrays::<T>(arrays, dtype_name)?;
    let output = py
        .detach(|| compiled.execute(&tensors))
        .map_err(|error| PyValueError::new_err(format!("预编译收缩失败: {error}")))?;
    dense_tensor_into_pyarray(py, output)
}

#[pymethods]
impl PyCompiledContraction {
    /// Compile a reusable exact contraction from a **SSA** path.
    #[staticmethod]
    #[pyo3(signature = (inputs, output, size_dict, *, ssa_path))]
    fn compile(
        inputs: Vec<Vec<u32>>,
        output: Vec<u32>,
        size_dict: HashMap<u32, usize>,
        ssa_path: Vec<(usize, usize)>,
    ) -> PyResult<Self> {
        let net = build_net(inputs, output, size_dict)?;
        let compiled = arctn::CompiledContraction::compile(&net, &ssa_path)
            .map_err(|error| PyValueError::new_err(format!("预编译路径非法: {error}")))?;
        Ok(Self { net, compiled })
    }

    /// Execute the already-compiled path on arrays with the frozen shapes.
    fn execute<'py>(
        &self,
        py: Python<'py>,
        arrays: Vec<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if arrays.len() != self.net.n_tensors() {
            return Err(PyValueError::new_err(format!(
                "arrays 个数 {} 与编译计划输入数 {} 不符",
                arrays.len(),
                self.net.n_tensors()
            )));
        }
        let dtype = detect_and_preflight_network_dtype(&self.net, &arrays, "预编译收缩失败")?;
        match dtype {
            PyArrayDtype::Float32 => {
                compiled_execute_typed::<f32>(py, &self.compiled, &arrays, "float32")
            }
            PyArrayDtype::Float64 => {
                compiled_execute_typed::<f64>(py, &self.compiled, &arrays, "float64")
            }
            PyArrayDtype::Complex64 => compiled_execute_typed::<num_complex::Complex<f32>>(
                py,
                &self.compiled,
                &arrays,
                "complex64",
            ),
            PyArrayDtype::Complex128 => compiled_execute_typed::<num_complex::Complex<f64>>(
                py,
                &self.compiled,
                &arrays,
                "complex128",
            ),
        }
    }

    /// Return small, stable metadata for diagnostics and cache ownership.
    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let value = PyDict::new(py);
        value.set_item("path_format", "ssa-v1")?;
        value.set_item("n_inputs", self.compiled.n_inputs())?;
        value.set_item("n_steps", self.compiled.n_steps())?;
        value.set_item("log10_flops", self.compiled.log10_flops)?;
        value.set_item("log2_peak_size", self.compiled.log2_peak_size)?;
        Ok(value)
    }

    fn __repr__(&self) -> String {
        format!(
            "CompiledContraction(path_format='ssa-v1', n_inputs={}, n_steps={})",
            self.compiled.n_inputs(),
            self.compiled.n_steps()
        )
    }
}

fn dense_tensor_into_pyarray<'py, T>(
    py: Python<'py>,
    tensor: arctn::DenseTensor<T>,
) -> PyResult<Bound<'py, PyAny>>
where
    T: numpy::Element + arctn::Scalar,
{
    use numpy::{IntoPyArray, PyArrayMethods};

    let (shape, data) = tensor.into_parts();
    let array = data.into_pyarray(py);
    Ok(array.reshape(shape)?.into_any())
}

/// Return the loaded extension's identity and repository state at compilation.
#[pyfunction]
fn build_info<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
    let features = PyDict::new(py);
    features.set_item("mt", env!("ARCTN_PY_BUILD_FEATURE_MT") == "1")?;
    let value = PyDict::new(py);
    value.set_item("version", env!("CARGO_PKG_VERSION"))?;
    value.set_item("commit", env!("ARCTN_PY_BUILD_COMMIT"))?;
    value.set_item("source_state", env!("ARCTN_PY_BUILD_SOURCE_STATE"))?;
    value.set_item("target_arch", std::env::consts::ARCH)?;
    value.set_item("target_os", std::env::consts::OS)?;
    value.set_item("target_pointer_width", usize::BITS)?;
    value.set_item("features", features)?;
    Ok(value)
}

#[pyfunction]
fn _validate_execution_plan_v2_json(text: &str) -> PyResult<()> {
    arctn::parse_embedded_execution_plan(text)
        .map(|_| ())
        .map_err(PyValueError::new_err)
}

#[pymodule]
fn arctn_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Planning.
    m.add_function(wrap_pyfunction!(optimize_auto, m)?)?;
    m.add_function(wrap_pyfunction!(optimize_auto_full, m)?)?;
    // Execution.
    m.add_function(wrap_pyfunction!(contract_sliced, m)?)?;
    m.add_class::<PyCompiledContraction>()?;
    m.add_function(wrap_pyfunction!(build_info, m)?)?;
    m.add_function(wrap_pyfunction!(_validate_execution_plan_v2_json, m)?)?;
    // Analysis.
    m.add_function(wrap_pyfunction!(simplify_stats, m)?)?;
    // Stable package metadata.
    m.add("AUTO_PRESETS", vec!["light", "heavy"])?;
    m.add("SLICING_MODES", vec!["fixed", "dynamic"])?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

// Planning and slicing entry points.

/// Return the final contraction path, metrics, and optional slicing result.
#[pyfunction]
#[pyo3(signature = (inputs, output, size_dict, *, preset="heavy", seed=0,
                    use_ssa=false, target_size=None, slicing_mode="fixed", max_time=None,
                    flops_weight=1.0, read_write_weight=64.0, rate_enabled=true))]
#[allow(clippy::too_many_arguments)]
fn optimize_auto_full<'py>(
    py: Python<'py>,
    inputs: Vec<Vec<u32>>,
    output: Vec<u32>,
    size_dict: HashMap<u32, usize>,
    preset: &str,
    seed: u64,
    use_ssa: bool,
    target_size: Option<Bound<'py, PyAny>>,
    slicing_mode: &str,
    max_time: Option<f64>,
    flops_weight: f64,
    read_write_weight: f64,
    rate_enabled: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let preset = pick_auto_preset(preset)?;
    let mode = pick_slicing_mode(slicing_mode)?;
    let objective = pick_planner_objective(flops_weight, read_write_weight)?;
    let target = pick_target_request(target_size)?;
    validate_max_time(max_time)?;
    let net = build_net(inputs, output, size_dict)?;
    let result = py
        .detach(|| {
            arctn::auto::optimize(
                &net,
                preset,
                seed,
                max_time,
                objective,
                rate_enabled,
                target.map(TargetRequest::exact_elements),
                mode,
            )
        })
        .map_err(PyValueError::new_err)?;
    let d = PyDict::new(py);
    d.set_item("preset", preset.as_str())?;
    d.set_item("slicing_mode", slicing_mode)?;
    d.set_item("wall_s", result.wall_s)?;
    d.set_item(
        "path",
        if use_ssa {
            result.path.clone()
        } else {
            ssa_to_linear(&result.path, net.n_tensors())
        },
    )?;
    let stats = &result.stats;
    d.set_item("log10_flops", stats.log10_flops)?;
    d.set_item("log2_max_size", stats.log2_max_size)?;
    d.set_item("log2_max_contraction_size", stats.log2_max_contraction_size)?;
    d.set_item("log2_total_size", stats.log2_total_size)?;
    d.set_item("log2_read_write", stats.log2_read_write)?;
    d.set_item("log2_peak_size", stats.log2_peak_size)?;
    set_planner_metadata(&d, stats, result.sliced.as_ref(), objective)?;
    d.set_item("sliced", result.sliced.is_some())?;
    let per_slice = if let Some(slice) = &result.sliced {
        d.set_item("sliced_legs", &slice.legs)?;
        d.set_item("log2_n_slices", slice.log2_n_slices)?;
        d.set_item("sliced_log10_flops_total", slice.log10_flops_total)?;
        d.set_item("sliced_log2_max_size", slice.per_slice.log2_max_size)?;
        d.set_item(
            "sliced_log2_max_contraction_size",
            slice.per_slice.log2_max_contraction_size,
        )?;
        d.set_item("sliced_log2_peak_size", slice.per_slice.log2_peak_size)?;
        Some(slice.per_slice.log2_max_size)
    } else {
        d.set_item("sliced_legs", Vec::<u32>::new())?;
        d.set_item("log2_n_slices", 0.0)?;
        None
    };
    set_memory_constraint_metadata(&d, py, target, per_slice)?;
    Ok(d)
}

fn build_net(
    inputs: Vec<Vec<u32>>,
    output: Vec<u32>,
    size_dict: HashMap<u32, usize>,
) -> PyResult<TensorNetwork> {
    if inputs.is_empty() {
        return Err(PyValueError::new_err("inputs 为空"));
    }
    let used: HashSet<u32> = inputs
        .iter()
        .flatten()
        .chain(output.iter())
        .copied()
        .collect();
    if let Some(extra) = size_dict.keys().find(|leg| !used.contains(leg)) {
        return Err(PyValueError::new_err(format!(
            "size_dict 含未在 inputs/output 使用的腿 {extra}；请删除它以避免标签拼写错误"
        )));
    }
    let net = TensorNetwork {
        name: "py".into(),
        inputs,
        output,
        size_dict,
    };
    net.validate()
        .map_err(|e| PyValueError::new_err(format!("网络结构非法: {e}")))?;
    Ok(net)
}

/// Return structural network simplification statistics.

#[pyfunction]
#[pyo3(signature = (inputs, output, size_dict))]
fn simplify_stats<'py>(
    py: Python<'py>,
    inputs: Vec<Vec<u32>>,
    output: Vec<u32>,
    size_dict: HashMap<u32, usize>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use pyo3::types::PyDict;
    let net = build_net(inputs, output, size_dict)?;
    let n_before = net.n_tensors();
    let sp = arctn::simplify::simplify(&net);
    let n_after = sp.reduced.n_tensors();

    let d = PyDict::new(py);
    d.set_item("n_tensors_before", n_before)?;
    d.set_item("n_tensors_after", n_after)?;
    d.set_item("simplify_ratio", n_after as f64 / n_before as f64)?;
    d.set_item("prefix_len", sp.prefix.len())?;
    d.set_item("prefix", sp.prefix.clone())?;
    d.set_item("reduced_inputs", sp.reduced.inputs.clone())?;
    d.set_item("reduced_output", sp.reduced.output.clone())?;
    d.set_item("map", sp.map.clone())?;
    Ok(d)
}

/// Execute a supplied path over the requested slice indices.
///
/// An empty `sliced_legs` list uses the ordinary unsliced execution path.
/// The outer parallelism is `min(number_of_slices, RAYON_NUM_THREADS)`, and
/// each in-flight slice owns a copy of the input tensors. `target_size`
/// constrains one intermediate tensor per slice, not total resident memory.
fn contract_sliced_typed<'py, T>(
    py: Python<'py>,
    net: &TensorNetwork,
    arrays: &[Bound<'py, PyAny>],
    ssa_path: Vec<(usize, usize)>,
    sliced_legs: Vec<u32>,
    dtype_name: &str,
) -> PyResult<Bound<'py, PyAny>>
where
    T: numpy::Element + Clone + arctn::Scalar,
{
    let tensors = copy_network_pyarrays::<T>(arrays, dtype_name)?;
    let output = py
        .detach(|| {
            arctn::slice::contract_network_sliced::<T>(net, &tensors, &ssa_path, &sliced_legs)
        })
        .map_err(|error| PyValueError::new_err(format!("切片收缩失败: {error}")))?;
    dense_tensor_into_pyarray(py, output)
}

#[pyfunction]
#[pyo3(signature = (inputs, output, size_dict, arrays, sliced_legs, path=None, ssa_path=None))]
#[allow(clippy::too_many_arguments)]
fn contract_sliced<'py>(
    py: Python<'py>,
    inputs: Vec<Vec<u32>>,
    output: Vec<u32>,
    size_dict: HashMap<u32, usize>,
    arrays: Vec<Bound<'py, PyAny>>,
    sliced_legs: Vec<u32>,
    path: Option<Vec<(usize, usize)>>,
    ssa_path: Option<Vec<(usize, usize)>>,
) -> PyResult<Bound<'py, PyAny>> {
    let n = inputs.len();
    if arrays.len() != n {
        return Err(PyValueError::new_err(format!(
            "arrays 个数 {} 与 inputs 个数 {n} 不符",
            arrays.len()
        )));
    }
    // Reject sliced output indices before entering the execution engine.
    for l in &sliced_legs {
        if output.contains(l) {
            return Err(PyValueError::new_err(format!(
                "切片腿 {l} 出现在 output 中：开放腿不能被切（它要出现在结果里）"
            )));
        }
    }
    if path.is_some() && ssa_path.is_some() {
        return Err(PyValueError::new_err(
            "path 与 ssa_path 互斥，只能给一个（path=linear 口径，ssa_path=SSA 口径）",
        ));
    }
    let net = build_net(inputs, output, size_dict)?;
    for leg in &sliced_legs {
        if !net.size_dict.contains_key(leg) {
            return Err(PyValueError::new_err(format!(
                "切片腿 {leg} 没有出现在网络 size_dict 中"
            )));
        }
    }
    // Complete dtype/layout/shape preflight happens before linear-path conversion and before
    // copying any ndarray payload.
    let dtype = detect_and_preflight_network_dtype(&net, &arrays, "切片收缩失败")?;

    // The two path formats are explicit and mutually exclusive.
    let ssa: Vec<(usize, usize)> = match (ssa_path, path) {
        (Some(s), _) => s,
        (None, Some(l)) => linear_to_ssa(&l, n)?,
        (None, None) => {
            return Err(PyValueError::new_err(
                "必须给 path=（linear 口径）或 ssa_path=（SSA 口径）之一。\
                 通常直接传 optimize_auto_full(..., use_ssa=True)['path'] 到 ssa_path=",
            ))
        }
    };

    match dtype {
        PyArrayDtype::Float32 => {
            contract_sliced_typed::<f32>(py, &net, &arrays, ssa, sliced_legs, "float32")
        }
        PyArrayDtype::Float64 => {
            contract_sliced_typed::<f64>(py, &net, &arrays, ssa, sliced_legs, "float64")
        }
        PyArrayDtype::Complex64 => contract_sliced_typed::<num_complex::Complex<f32>>(
            py,
            &net,
            &arrays,
            ssa,
            sliced_legs,
            "complex64",
        ),
        PyArrayDtype::Complex128 => contract_sliced_typed::<num_complex::Complex<f64>>(
            py,
            &net,
            &arrays,
            ssa,
            sliced_legs,
            "complex128",
        ),
    }
}
