//! Light and Heavy interfaces for an explicitly supplied private engine.
//!
//! Set `ARCTN_ENGINE_LIBRARY` to the engine library before calling these
//! functions. Python wheels containing an engine set this automatically.
//! The public crate validates inputs and replays returned paths; the engine
//! implements the preset search strategies.

use std::collections::HashMap;
use std::ffi::{c_char, CStr, OsString};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use libloading::Library;
use serde::Deserialize;

use crate::network::{LegId, TensorNetwork};
use crate::objective::PlannerObjective;
use crate::path::{simulate_path, PathStats, SsaPath};
use crate::slice::{slice_result_fits_target_size, validate_slice_legs, SliceResult};

/// Preset selected for a planning call.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum AutoPreset {
    Light,
    #[default]
    Heavy,
}

impl AutoPreset {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Light => "light",
            Self::Heavy => "heavy",
        }
    }

    pub fn from_label(label: &str) -> Option<Self> {
        match label {
            "light" => Some(Self::Light),
            "heavy" => Some(Self::Heavy),
            _ => None,
        }
    }
}

/// Slicing applied after ordinary preset path selection.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum SlicingMode {
    /// Preserve the selected path while choosing sliced legs.
    #[default]
    Fixed,
    /// Alternate slice selection and local path reconfiguration.
    Dynamic,
}

impl SlicingMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Fixed => "fixed",
            Self::Dynamic => "dynamic",
        }
    }

    pub fn from_label(label: &str) -> Option<Self> {
        match label {
            "fixed" => Some(Self::Fixed),
            "dynamic" => Some(Self::Dynamic),
            _ => None,
        }
    }
}

/// Final planning result, with metrics recomputed from the returned path.
#[derive(Clone, Debug)]
pub struct AutoResult {
    pub path: SsaPath,
    /// Metrics on the original, unsliced network.
    pub stats: PathStats,
    /// Exact sliced legs and the metrics of the corresponding sliced path.
    pub sliced: Option<SliceResult>,
    /// Planning wall time reported by the engine, in seconds.
    pub wall_s: f64,
}

type EngineRun = unsafe extern "C" fn(*const u8, usize) -> *mut c_char;
type EngineFree = unsafe extern "C" fn(*mut c_char);

struct Engine {
    // Retain the library for the lifetime of its function pointers.
    _library: Library,
    run: EngineRun,
    free: EngineFree,
}

impl Engine {
    fn load(path: &Path) -> Result<Self, String> {
        // The caller explicitly chooses this native library. Its exported
        // functions must implement the documented version 1 C ABI.
        unsafe {
            let library = Library::new(path)
                .map_err(|error| format!("cannot load ARCTN_ENGINE_LIBRARY: {error}"))?;
            let version = library
                .get::<unsafe extern "C" fn() -> u32>(b"arctn_engine_abi_version\0")
                .map_err(|error| format!("engine ABI version function is missing: {error}"))?;
            let actual_version = version();
            if actual_version != 1 {
                return Err(format!(
                    "unsupported ArcTN engine ABI version {actual_version}; expected 1"
                ));
            }
            let run = *library
                .get::<EngineRun>(b"arctn_engine_run_v1\0")
                .map_err(|error| format!("engine planning function is missing: {error}"))?;
            let free = *library
                .get::<EngineFree>(b"arctn_engine_free_v1\0")
                .map_err(|error| format!("engine release function is missing: {error}"))?;
            Ok(Self {
                _library: library,
                run,
                free,
            })
        }
    }

    fn run(&self, request: &[u8]) -> Result<EngineResponse, String> {
        // The request remains valid until the synchronous engine call returns.
        let pointer = unsafe { (self.run)(request.as_ptr(), request.len()) };
        if pointer.is_null() {
            return Err("ArcTN engine returned a null response".into());
        }
        let response = EngineBuffer {
            pointer,
            free: self.free,
        };
        // Version 1 returns an owned, NUL-terminated UTF-8 response. The guard
        // releases it through the same library on both success and failure.
        let text = unsafe { CStr::from_ptr(response.pointer) }
            .to_str()
            .map_err(|error| format!("ArcTN engine response is not UTF-8: {error}"))?;
        decode_response(text)
    }
}

struct EngineBuffer {
    pointer: *mut c_char,
    free: EngineFree,
}

impl Drop for EngineBuffer {
    fn drop(&mut self) {
        // This buffer is returned by the engine and must use its deallocator.
        unsafe { (self.free)(self.pointer) };
    }
}

fn engine() -> Result<Arc<Engine>, String> {
    let path = std::env::var_os("ARCTN_ENGINE_LIBRARY")
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            "Light/Heavy requires an ArcTN engine library; install a Python wheel containing the engine, or set ARCTN_ENGINE_LIBRARY to a compatible library path."
                .to_owned()
        })?;
    // Keep successfully loaded libraries alive, including any engine threads.
    // Failed loads are not cached, so a corrected installation can be retried.
    static ENGINES: OnceLock<Mutex<HashMap<OsString, Arc<Engine>>>> = OnceLock::new();
    let mut engines = ENGINES
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .map_err(|_| "ArcTN engine library cache is unavailable".to_owned())?;
    if let Some(engine) = engines.get(&path) {
        return Ok(Arc::clone(engine));
    }
    let loaded = Arc::new(Engine::load(&PathBuf::from(&path))?);
    engines.insert(path, Arc::clone(&loaded));
    Ok(loaded)
}

#[derive(Deserialize)]
struct EngineResponse {
    path: SsaPath,
    sliced_legs: Option<Vec<LegId>>,
    wall_s: f64,
}

fn decode_response(text: &str) -> Result<EngineResponse, String> {
    let value: serde_json::Value = serde_json::from_str(text)
        .map_err(|error| format!("ArcTN engine returned invalid JSON: {error}"))?;
    let object = value
        .as_object()
        .ok_or_else(|| "ArcTN engine response must be a JSON object".to_owned())?;
    if let Some(error) = object.get("error") {
        return Err(error
            .as_str()
            .map(|message| format!("ArcTN engine: {message}"))
            .unwrap_or_else(|| "ArcTN engine error must be a string".into()));
    }
    if !object.contains_key("sliced_legs") {
        return Err("ArcTN engine response is missing sliced_legs".into());
    }
    let response: EngineResponse = serde_json::from_value(value)
        .map_err(|error| format!("ArcTN engine response has invalid fields: {error}"))?;
    if !response.wall_s.is_finite() || response.wall_s < 0.0 {
        return Err("ArcTN engine wall_s must be finite and non-negative".into());
    }
    Ok(response)
}

fn validate_response(
    net: &TensorNetwork,
    response: EngineResponse,
    target_size: Option<usize>,
) -> Result<AutoResult, String> {
    let stats = simulate_path(net, &response.path)
        .map_err(|error| format!("ArcTN engine returned an invalid path: {error}"))?;
    let sliced = match response.sliced_legs {
        Some(legs) => {
            if target_size.is_none() {
                return Err("ArcTN engine returned slicing without a target_size request".into());
            }
            validate_slice_legs(net, &legs)
                .map_err(|error| format!("ArcTN engine returned invalid sliced legs: {error}"))?;
            let mut sliced_net = net.clone();
            let mut log2_n_slices = 0.0;
            for &leg in &legs {
                log2_n_slices += net.log2_dim(leg);
                sliced_net.size_dict.insert(leg, 1);
            }
            let per_slice = simulate_path(&sliced_net, &response.path)?;
            Some(SliceResult {
                legs,
                log2_n_slices,
                per_slice,
                log10_flops_total: per_slice.log10_flops
                    + log2_n_slices * std::f64::consts::LOG10_2,
            })
        }
        None => None,
    };
    if let Some(target) = target_size {
        let result = sliced.as_ref().ok_or_else(|| {
            "ArcTN engine did not return sliced_legs for the target_size request".to_owned()
        })?;
        if !slice_result_fits_target_size(net, &response.path, result, target)? {
            return Err(format!(
                "ArcTN engine returned a plan that exceeds target_size={target}"
            ));
        }
    }
    Ok(AutoResult {
        path: response.path,
        stats,
        sliced,
        wall_s: response.wall_s,
    })
}

/// Run a preset through the explicitly supplied private engine.
///
/// `max_time` is a cooperative planning limit. `target_size` constrains the
/// number of elements in each produced intermediate per slice; it does not
/// specify process memory or a strict wall-clock limit.
#[allow(clippy::too_many_arguments)]
pub fn optimize(
    net: &TensorNetwork,
    preset: AutoPreset,
    seed: u64,
    max_time: Option<f64>,
    objective: PlannerObjective,
    rate_enabled: bool,
    target_size: Option<usize>,
    mode: SlicingMode,
) -> Result<AutoResult, String> {
    net.validate()?;
    if max_time.is_some_and(|seconds| !seconds.is_finite() || seconds <= 0.0) {
        return Err("max_time must be finite and greater than zero".into());
    }
    if target_size == Some(0) {
        return Err("target_size must be greater than zero".into());
    }
    if mode == SlicingMode::Dynamic && target_size.is_none() {
        return Err("dynamic slicing requires target_size".into());
    }
    let objective = PlannerObjective::new(objective.flops_weight(), objective.read_write_weight())?;
    let mut dimensions: Vec<(LegId, usize)> = net
        .size_dict
        .iter()
        .map(|(&leg, &dimension)| (leg, dimension))
        .collect();
    dimensions.sort_unstable();
    let request = serde_json::json!({
        "inputs": net.inputs,
        "output": net.output,
        "size_dict": dimensions,
        "preset": preset.as_str(),
        "seed": seed,
        "max_time": max_time,
        "flops_weight": objective.flops_weight(),
        "read_write_weight": objective.read_write_weight(),
        "rate_enabled": rate_enabled,
        "target_size": target_size,
        "slicing_mode": mode.as_str(),
        "threads": rayon::current_num_threads(),
    });
    let request = serde_json::to_vec(&request)
        .map_err(|error| format!("cannot serialize ArcTN engine request: {error}"))?;
    let response = engine()?.run(&request)?;
    validate_response(net, response, target_size)
}

/// Run Light or Heavy with the default planner objective.
pub fn auto_path_preset(
    net: &TensorNetwork,
    preset: AutoPreset,
    seed: u64,
    max_time: Option<f64>,
) -> Result<AutoResult, String> {
    auto_path_preset_with_objective(net, preset, seed, max_time, PlannerObjective::FIXED)
}

/// Run Light or Heavy with an explicit planner objective.
pub fn auto_path_preset_with_objective(
    net: &TensorNetwork,
    preset: AutoPreset,
    seed: u64,
    max_time: Option<f64>,
    objective: PlannerObjective,
) -> Result<AutoResult, String> {
    optimize(
        net,
        preset,
        seed,
        max_time,
        objective,
        true,
        None,
        SlicingMode::Fixed,
    )
}

/// Select a preset path, then choose sliced legs for that fixed path.
pub fn auto_path_preset_to_size(
    net: &TensorNetwork,
    preset: AutoPreset,
    seed: u64,
    max_time: Option<f64>,
    target_size: usize,
) -> Result<AutoResult, String> {
    auto_path_preset_to_size_with_objective(
        net,
        preset,
        seed,
        max_time,
        target_size,
        PlannerObjective::FIXED,
        true,
    )
}

/// Select a preset path, then apply the requested slicing mode.
pub fn auto_path_preset_to_size_with_mode(
    net: &TensorNetwork,
    preset: AutoPreset,
    seed: u64,
    max_time: Option<f64>,
    target_size: usize,
    slicing_mode: SlicingMode,
) -> Result<AutoResult, String> {
    auto_path_preset_to_size_with_mode_and_objective(
        net,
        preset,
        seed,
        max_time,
        target_size,
        slicing_mode,
        PlannerObjective::FIXED,
        true,
    )
}

/// Fixed-path slicing with an explicit planner objective and rate selection.
#[allow(clippy::too_many_arguments)]
pub fn auto_path_preset_to_size_with_objective(
    net: &TensorNetwork,
    preset: AutoPreset,
    seed: u64,
    max_time: Option<f64>,
    target_size: usize,
    objective: PlannerObjective,
    rate_enabled: bool,
) -> Result<AutoResult, String> {
    auto_path_preset_to_size_with_mode_and_objective(
        net,
        preset,
        seed,
        max_time,
        target_size,
        SlicingMode::Fixed,
        objective,
        rate_enabled,
    )
}

/// Slicing with an explicit mode, planner objective, and rate selection.
#[allow(clippy::too_many_arguments)]
pub fn auto_path_preset_to_size_with_mode_and_objective(
    net: &TensorNetwork,
    preset: AutoPreset,
    seed: u64,
    max_time: Option<f64>,
    target_size: usize,
    slicing_mode: SlicingMode,
    objective: PlannerObjective,
    rate_enabled: bool,
) -> Result<AutoResult, String> {
    optimize(
        net,
        preset,
        seed,
        max_time,
        objective,
        rate_enabled,
        Some(target_size),
        slicing_mode,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn network() -> TensorNetwork {
        TensorNetwork {
            name: "sparse-leg".into(),
            inputs: vec![vec![u32::MAX], vec![u32::MAX]],
            output: vec![],
            size_dict: [(u32::MAX, 2)].into_iter().collect(),
        }
    }

    #[test]
    fn response_replay_preserves_sparse_sliced_legs() {
        let response =
            decode_response(r#"{"path":[[0,1]],"sliced_legs":[4294967295],"wall_s":0.25}"#)
                .unwrap();
        let result = validate_response(&network(), response, Some(1)).unwrap();
        let sliced = result.sliced.unwrap();
        assert_eq!(sliced.legs, vec![u32::MAX]);
        assert_eq!(sliced.log2_n_slices, 1.0);
        assert_eq!(sliced.per_slice.log10_flops, 0.0);
        assert_eq!(sliced.log10_flops_total, result.stats.log10_flops);
    }

    #[test]
    fn response_requires_valid_path_and_explicit_slicing() {
        let invalid_path =
            decode_response(r#"{"path":[[0,2]],"sliced_legs":null,"wall_s":0}"#).unwrap();
        assert!(validate_response(&network(), invalid_path, None).is_err());
        let missing_slices =
            decode_response(r#"{"path":[[0,1]],"sliced_legs":null,"wall_s":0}"#).unwrap();
        assert!(validate_response(&network(), missing_slices, Some(1)).is_err());
        assert!(decode_response(r#"{"path":[[0,1]],"wall_s":0}"#).is_err());
        assert!(decode_response(r#"{"path":[[0,1]],"sliced_legs":null,"wall_s":-1}"#).is_err());
    }

    #[test]
    fn response_rejects_duplicate_slices_and_exceeded_target() {
        let duplicate =
            decode_response(r#"{"path":[[0,1]],"sliced_legs":[4294967295,4294967295],"wall_s":0}"#)
                .unwrap();
        assert!(validate_response(&network(), duplicate, Some(1)).is_err());
        let net = TensorNetwork {
            name: "open-output".into(),
            inputs: vec![vec![3, 8], vec![8]],
            output: vec![3],
            size_dict: [(3, 3), (8, 2)].into_iter().collect(),
        };
        let response = decode_response(r#"{"path":[[0,1]],"sliced_legs":[],"wall_s":0}"#).unwrap();
        assert!(validate_response(&net, response, Some(2)).is_err());
    }

    #[test]
    fn unary_path_preserves_negative_infinite_log_metrics() {
        let net = TensorNetwork {
            name: "unary".into(),
            inputs: vec![vec![17]],
            output: vec![17],
            size_dict: [(17, 2)].into_iter().collect(),
        };
        let response = decode_response(r#"{"path":[],"sliced_legs":[],"wall_s":0}"#).unwrap();
        let result = validate_response(&net, response, Some(2)).unwrap();
        assert_eq!(result.stats.log10_flops, f64::NEG_INFINITY);
        assert_eq!(result.stats.log2_read_write, f64::NEG_INFINITY);
        assert_eq!(result.sliced.unwrap().log10_flops_total, f64::NEG_INFINITY);
    }
}
