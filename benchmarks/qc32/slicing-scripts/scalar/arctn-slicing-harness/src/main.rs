use arctn::slice::slice_and_reconf_to_size_with_objective;
use arctn::{
    find_slices_to_size, simulate_path, slice_result_fits_target_size, PathStats, PlannerObjective,
    SliceResult, SsaPath, TensorNetwork,
};
use cpu_time::ProcessTime;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

const DYNAMIC_ROUNDS: usize = 3;
const DYNAMIC_SUBTREE_SIZE: usize = 8;

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "lowercase")]
enum Mode {
    Fixed,
    Dynamic,
}

impl Mode {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "fixed" => Ok(Self::Fixed),
            "dynamic" => Ok(Self::Dynamic),
            _ => Err(format!(
                "--mode must be 'fixed' or 'dynamic', got {value:?}"
            )),
        }
    }

    fn method_id(self) -> &'static str {
        match self {
            Self::Fixed => "arctn-heavy-fixed-tree-post-slicing",
            Self::Dynamic => "arctn-heavy-dynamic-slicing",
        }
    }

    fn mechanism(self) -> &'static str {
        match self {
            Self::Fixed => "fixed-path-slicing",
            Self::Dynamic => "slice-and-reconfigure",
        }
    }

    fn entrypoint(self) -> &'static str {
        match self {
            Self::Fixed => "arctn::find_slices_to_size",
            Self::Dynamic => "arctn::slice::slice_and_reconf_to_size_with_objective",
        }
    }
}

#[derive(Debug)]
struct Args {
    mode: Mode,
    network: PathBuf,
    heavy_path: PathBuf,
    target_size: usize,
    output: PathBuf,
}

#[derive(Debug, Deserialize)]
struct FrozenNetwork {
    name: String,
    inputs: Vec<Vec<u32>>,
    output: Vec<u32>,
    size_dict: Vec<(u32, usize)>,
}

#[derive(Debug)]
struct HeavyPathInput {
    path: SsaPath,
    method_metadata: Option<String>,
    preset_metadata: Option<String>,
}

#[derive(Debug, Serialize)]
struct ObjectiveReport {
    name: &'static str,
    flops_weight: f64,
    read_write_weight: f64,
    application: &'static str,
}

#[derive(Debug, Serialize)]
struct TimingReport {
    wall_seconds: f64,
    process_cpu_seconds: f64,
}

#[derive(Debug, Serialize)]
struct PathStatsReport {
    log10_flops: f64,
    log2_max_size: f64,
    log2_max_contraction_size: f64,
    log2_total_size: f64,
    log2_read_write: f64,
    log2_peak_size: f64,
}

impl From<PathStats> for PathStatsReport {
    fn from(value: PathStats) -> Self {
        Self {
            log10_flops: value.log10_flops,
            log2_max_size: value.log2_max_size,
            log2_max_contraction_size: value.log2_max_contraction_size,
            log2_total_size: value.log2_total_size,
            log2_read_write: value.log2_read_write,
            log2_peak_size: value.log2_peak_size,
        }
    }
}

#[derive(Debug, Serialize)]
struct NativeSliceMetrics {
    log2_n_slices: f64,
    log10_total_sliced_flops: f64,
    per_slice: PathStatsReport,
}

#[derive(Debug, Serialize)]
struct Parameters {
    rounds: Option<usize>,
    subtree_size: Option<usize>,
}

#[derive(Debug, Serialize)]
struct ValidationReport {
    network_valid: bool,
    input_heavy_ssa_path_valid: bool,
    input_path_consumed_unchanged: bool,
    final_ssa_path_valid: bool,
    target_satisfied: bool,
    fixed_final_path_equals_input: Option<bool>,
    pure_flops_objective_verified: bool,
}

#[derive(Debug, Serialize)]
struct OutputReport {
    schema: &'static str,
    status: &'static str,
    mode: Mode,
    method: &'static str,
    mechanism: &'static str,
    entrypoint: &'static str,
    network_name: String,
    target_size: usize,
    input_path_file: String,
    input_method_metadata: Option<String>,
    input_preset_metadata: Option<String>,
    input_heavy_ssa_path: SsaPath,
    input_path_stats: PathStatsReport,
    objective: ObjectiveReport,
    parameters: Parameters,
    stage_timing: TimingReport,
    final_ssa_path: SsaPath,
    sliced_legs: Vec<u32>,
    native_slice_metrics: NativeSliceMetrics,
    final_path_stats_before_slicing: PathStatsReport,
    validation: ValidationReport,
}

fn usage() -> &'static str {
    "Usage: arctn-slicing-harness --mode fixed|dynamic --network NETWORK.json \
     --heavy-path HEAVY_PATH.json --target-size ELEMENTS --output RESULT.json"
}

fn parse_args() -> Result<Args, String> {
    let mut mode = None;
    let mut network = None;
    let mut heavy_path = None;
    let mut target_size = None;
    let mut output = None;
    let raw: Vec<String> = env::args().skip(1).collect();
    if raw.iter().any(|arg| arg == "-h" || arg == "--help") {
        println!("{}", usage());
        std::process::exit(0);
    }
    let mut index = 0;
    while index < raw.len() {
        let flag = &raw[index];
        let value = raw
            .get(index + 1)
            .ok_or_else(|| format!("{flag} requires a value"))?;
        match flag.as_str() {
            "--mode" => mode = Some(Mode::parse(value)?),
            "--network" => network = Some(PathBuf::from(value)),
            "--heavy-path" => heavy_path = Some(PathBuf::from(value)),
            "--target-size" => {
                let parsed = value.parse::<usize>().map_err(|_| {
                    format!("--target-size must be a positive integer, got {value:?}")
                })?;
                if parsed == 0 {
                    return Err("--target-size must be greater than zero".into());
                }
                target_size = Some(parsed);
            }
            "--output" => output = Some(PathBuf::from(value)),
            _ => return Err(format!("unknown argument {flag:?}\n{}", usage())),
        }
        index += 2;
    }
    Ok(Args {
        mode: mode.ok_or_else(|| format!("missing --mode\n{}", usage()))?,
        network: network.ok_or_else(|| format!("missing --network\n{}", usage()))?,
        heavy_path: heavy_path.ok_or_else(|| format!("missing --heavy-path\n{}", usage()))?,
        target_size: target_size.ok_or_else(|| format!("missing --target-size\n{}", usage()))?,
        output: output.ok_or_else(|| format!("missing --output\n{}", usage()))?,
    })
}

fn read_json(path: &Path) -> Result<Value, String> {
    let bytes = fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))?;
    serde_json::from_slice(&bytes)
        .map_err(|error| format!("parse JSON {}: {error}", path.display()))
}

fn load_network(path: &Path) -> Result<TensorNetwork, String> {
    let value = read_json(path)?;
    let frozen: FrozenNetwork = serde_json::from_value(value)
        .map_err(|error| format!("decode frozen network {}: {error}", path.display()))?;
    let mut size_dict = HashMap::with_capacity(frozen.size_dict.len());
    for (leg, dimension) in frozen.size_dict {
        if size_dict.insert(leg, dimension).is_some() {
            return Err(format!(
                "network {} repeats leg {leg} in size_dict",
                frozen.name
            ));
        }
    }
    let network = TensorNetwork {
        name: frozen.name,
        inputs: frozen.inputs,
        output: frozen.output,
        size_dict,
    };
    network.validate()?;
    Ok(network)
}

fn string_at(value: &Value, pointer: &str) -> Option<String> {
    value
        .pointer(pointer)
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
}

fn load_heavy_path(path: &Path) -> Result<HeavyPathInput, String> {
    let value = read_json(path)?;
    let method_metadata = string_at(&value, "/method");
    let preset_metadata =
        string_at(&value, "/planner/preset").or_else(|| string_at(&value, "/preset"));
    if let Some(method) = &method_metadata {
        if !method.to_ascii_lowercase().contains("heavy") {
            return Err(format!(
                "path metadata identifies a non-Heavy method: {method:?}"
            ));
        }
    }
    if let Some(preset) = &preset_metadata {
        if preset != "heavy" {
            return Err(format!(
                "path metadata identifies preset {preset:?}, expected 'heavy'"
            ));
        }
    }
    let path_value = if value.is_array() {
        value.clone()
    } else {
        value
            .get("ssa_path")
            .or_else(|| value.get("path"))
            .cloned()
            .ok_or_else(|| {
                format!(
                    "{} must be a raw SSA array or contain 'ssa_path'",
                    path.display()
                )
            })?
    };
    let ssa: SsaPath = serde_json::from_value(path_value)
        .map_err(|error| format!("decode Heavy SSA path {}: {error}", path.display()))?;
    Ok(HeavyPathInput {
        path: ssa,
        method_metadata,
        preset_metadata,
    })
}

fn pure_flops_objective() -> Result<PlannerObjective, String> {
    let objective = PlannerObjective::new(1.0, 0.0)?;
    if objective.as_str() != "total_flops"
        || objective.flops_weight() != 1.0
        || objective.read_write_weight() != 0.0
    {
        return Err("ArcTN did not resolve the requested pure-FLOPs objective".into());
    }
    Ok(objective)
}

fn run_mode(
    mode: Mode,
    network: &TensorNetwork,
    heavy_path: &SsaPath,
    target_size: usize,
    objective: PlannerObjective,
) -> Result<(SsaPath, SliceResult, TimingReport), String> {
    let wall_started = Instant::now();
    let cpu_started = ProcessTime::now();
    let result = match mode {
        Mode::Fixed => find_slices_to_size(network, heavy_path, target_size)
            .map(|slice| (heavy_path.clone(), slice)),
        Mode::Dynamic => slice_and_reconf_to_size_with_objective(
            network,
            heavy_path,
            target_size,
            DYNAMIC_ROUNDS,
            DYNAMIC_SUBTREE_SIZE,
            objective,
        ),
    };
    let timing = TimingReport {
        wall_seconds: wall_started.elapsed().as_secs_f64(),
        process_cpu_seconds: cpu_started.elapsed().as_secs_f64(),
    };
    let (path, slice) = result.ok_or_else(|| {
        format!(
            "{} did not produce a slicing plan satisfying target_size={target_size}",
            mode.mechanism()
        )
    })?;
    Ok((path, slice, timing))
}

fn build_report(args: &Args) -> Result<OutputReport, String> {
    let network = load_network(&args.network)?;
    let heavy = load_heavy_path(&args.heavy_path)?;
    if heavy.path.len() != network.n_tensors().saturating_sub(1) {
        return Err(format!(
            "Heavy SSA path has {} steps for {} input tensors; expected {}",
            heavy.path.len(),
            network.n_tensors(),
            network.n_tensors().saturating_sub(1)
        ));
    }
    let input_stats = simulate_path(&network, &heavy.path)
        .map_err(|error| format!("invalid input Heavy SSA path: {error}"))?;
    let objective = pure_flops_objective()?;
    let input_path_copy = heavy.path.clone();
    let (final_path, slice, stage_timing) = run_mode(
        args.mode,
        &network,
        &heavy.path,
        args.target_size,
        objective,
    )?;
    if heavy.path != input_path_copy {
        return Err("slicing mutated the caller-supplied Heavy SSA path".into());
    }
    let final_stats = simulate_path(&network, &final_path)
        .map_err(|error| format!("invalid final SSA path: {error}"))?;
    let target_satisfied =
        slice_result_fits_target_size(&network, &final_path, &slice, args.target_size)?;
    if !target_satisfied {
        return Err("final path and sliced legs do not satisfy target_size".into());
    }
    if matches!(args.mode, Mode::Fixed) && final_path != heavy.path {
        return Err("fixed-path mode changed the input Heavy SSA path".into());
    }
    let objective_application = match args.mode {
        Mode::Fixed => {
            "pure FLOPs is the frozen paired-comparison objective; find_slices_to_size has no objective argument"
        }
        Mode::Dynamic => {
            "passed directly to slice_and_reconf_to_size_with_objective"
        }
    };
    Ok(OutputReport {
        schema: "arctn-heavy-slicing-harness-v1",
        status: "success",
        mode: args.mode,
        method: args.mode.method_id(),
        mechanism: args.mode.mechanism(),
        entrypoint: args.mode.entrypoint(),
        network_name: network.name.clone(),
        target_size: args.target_size,
        input_path_file: args.heavy_path.display().to_string(),
        input_method_metadata: heavy.method_metadata,
        input_preset_metadata: heavy.preset_metadata,
        input_heavy_ssa_path: heavy.path,
        input_path_stats: input_stats.into(),
        objective: ObjectiveReport {
            name: objective.as_str(),
            flops_weight: objective.flops_weight(),
            read_write_weight: objective.read_write_weight(),
            application: objective_application,
        },
        parameters: Parameters {
            rounds: matches!(args.mode, Mode::Dynamic).then_some(DYNAMIC_ROUNDS),
            subtree_size: matches!(args.mode, Mode::Dynamic).then_some(DYNAMIC_SUBTREE_SIZE),
        },
        stage_timing,
        final_ssa_path: final_path,
        sliced_legs: slice.legs,
        native_slice_metrics: NativeSliceMetrics {
            log2_n_slices: slice.log2_n_slices,
            log10_total_sliced_flops: slice.log10_flops_total,
            per_slice: slice.per_slice.into(),
        },
        final_path_stats_before_slicing: final_stats.into(),
        validation: ValidationReport {
            network_valid: true,
            input_heavy_ssa_path_valid: true,
            input_path_consumed_unchanged: true,
            final_ssa_path_valid: true,
            target_satisfied: true,
            fixed_final_path_equals_input: matches!(args.mode, Mode::Fixed).then_some(true),
            pure_flops_objective_verified: true,
        },
    })
}

fn write_json(path: &Path, report: &OutputReport) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("create output directory {}: {error}", parent.display()))?;
    }
    let bytes =
        serde_json::to_vec_pretty(report).map_err(|error| format!("serialize result: {error}"))?;
    fs::write(path, bytes).map_err(|error| format!("write {}: {error}", path.display()))
}

fn main() {
    let result = parse_args()
        .and_then(|args| build_report(&args).and_then(|report| write_json(&args.output, &report)));
    if let Err(error) = result {
        eprintln!("arctn-slicing-harness: {error}");
        std::process::exit(1);
    }
}
