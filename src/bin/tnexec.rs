//! Single-process tensor-network executor.
//!
//! Supports live pathfinding, saved plans, and sliced contraction with `--target-size`.
//! Reported GFLOPS is scalar multiplications divided by wall time; a complex
//! multiplication counts as one operation.

mod cli;

use arctn::{
    contract_network, contract_network_sliced, find_slices, greedy, naive_einsum,
    parse_execution_plan_for_network as parse_saved_execution_plan, random_greedy, simulate_path,
    DenseTensor, LegId, Scalar, TensorNetwork,
};
use num_complex::{Complex32, Complex64};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::path::Path;
use std::time::Instant;

const FEASIBILITY_TOLERANCE_LOG2: f64 = 1e-9;
// This is an acceptance threshold with an absolute floor, not a general
// roundoff bound for every network or contraction order.
const CHECK_RELATIVE_TOLERANCE: f64 = 1e-6;

/// Evaluate references using at least double precision, preserving the exact
/// values supplied to the executor rather than generating a second input set.
trait OracleScalar: Scalar {
    type Reference: Scalar;
    const REFERENCE_DTYPE: &'static str;
    fn to_reference(self) -> Self::Reference;
}

impl OracleScalar for f32 {
    type Reference = f64;
    const REFERENCE_DTYPE: &'static str = "f64";
    fn to_reference(self) -> f64 {
        f64::from(self)
    }
}

impl OracleScalar for f64 {
    type Reference = f64;
    const REFERENCE_DTYPE: &'static str = "f64";
    fn to_reference(self) -> f64 {
        self
    }
}

impl OracleScalar for Complex32 {
    type Reference = Complex64;
    const REFERENCE_DTYPE: &'static str = "c64";
    fn to_reference(self) -> Complex64 {
        Complex64::new(f64::from(self.re), f64::from(self.im))
    }
}

impl OracleScalar for Complex64 {
    type Reference = Complex64;
    const REFERENCE_DTYPE: &'static str = "c64";
    fn to_reference(self) -> Complex64 {
        self
    }
}

fn reference_tensors<T: OracleScalar>(
    tensors: &[DenseTensor<T>],
) -> Vec<DenseTensor<T::Reference>> {
    tensors
        .iter()
        .map(|tensor| {
            DenseTensor::from_data(
                tensor.shape().to_vec(),
                tensor
                    .data()
                    .iter()
                    .map(|&value| value.to_reference())
                    .collect(),
            )
        })
        .collect()
}

/// Disables inner GEMM threading when multiple slice chunks run concurrently.
/// Explicit MATMUL_NUM_THREADS settings take precedence. Call before the first GEMM.
fn set_matmul_threads(n_chunks: usize) {
    if std::env::var_os("MATMUL_NUM_THREADS").is_some() {
        return;
    }
    if n_chunks > 1 {
        std::env::set_var("MATMUL_NUM_THREADS", "1");
    }
}

struct Args {
    net_file: String,
    method: String,
    method_explicit: bool,
    trials: usize,
    trials_explicit: bool,
    seed: u64,
    mem_target: Option<f64>,
    canonical_target_size: Option<usize>,
    complex: bool,
    single: bool,
    check: bool,
    data_file: Option<String>,
    load_path: Option<String>,
    allow_legacy_plan: bool,
}

fn parse_positive_target_size(raw: &str) -> usize {
    raw.parse::<usize>()
        .ok()
        .filter(|value| *value > 0)
        .unwrap_or_else(|| {
            eprintln!("--target-size 必须是大于 0 的整数（单切片最大中间张量元素数）");
            std::process::exit(2);
        })
}

fn parse_nonnegative_log2_target(raw: &str) -> f64 {
    raw.parse::<f64>()
        .ok()
        .filter(|value| value.is_finite() && *value >= 0.0)
        .unwrap_or_else(|| {
            eprintln!("--mem-target 必须是有限且不小于 0 的 log2(元素数)");
            std::process::exit(2);
        })
}

fn target_size_to_log2(target_size: usize) -> f64 {
    (target_size as f64).log2()
}

fn target_size_from_log2(target_log2: f64) -> Option<usize> {
    if !target_log2.is_finite() || target_log2 < 0.0 || target_log2 >= usize::BITS as f64 {
        return None;
    }
    let value = target_log2.exp2();
    let nearest = value.round();
    // Round exp2(log2(N)) near an integer; floor fractional element targets.
    let roundoff_tolerance = 8.0 * f64::EPSILON * nearest.abs().max(1.0);
    let elements = if (value - nearest).abs() <= roundoff_tolerance {
        nearest
    } else {
        value.floor()
    };
    let usize_upper_exclusive = 2.0_f64.powi(usize::BITS as i32);
    (elements >= 1.0 && elements < usize_upper_exclusive).then_some(elements as usize)
}

fn strictest_target_log2(cli: Option<f64>, artifact: Option<f64>) -> Option<f64> {
    match (cli, artifact) {
        (Some(cli), Some(artifact)) => Some(cli.min(artifact)),
        (Some(target), None) | (None, Some(target)) => Some(target),
        (None, None) => None,
    }
}

fn strictest_target_size(cli: Option<usize>, artifact: Option<usize>) -> Option<usize> {
    match (cli, artifact) {
        (Some(cli), Some(artifact)) => Some(cli.min(artifact)),
        (Some(target), None) | (None, Some(target)) => Some(target),
        (None, None) => None,
    }
}

fn effective_target_size(
    cli_target_size: Option<usize>,
    artifact_target_size: Option<usize>,
    cli_target_log2: Option<f64>,
    artifact_target_log2: Option<f64>,
) -> Option<usize> {
    let exact = strictest_target_size(cli_target_size, artifact_target_size);
    let legacy = strictest_target_log2(cli_target_log2, artifact_target_log2)
        .and_then(target_size_from_log2);
    match (exact, legacy) {
        (Some(exact), Some(legacy)) => Some(exact.min(legacy)),
        (Some(target), None) | (None, Some(target)) => Some(target),
        (None, None) => None,
    }
}

fn effective_target_log2(
    cli_target_size: Option<usize>,
    artifact_target_size: Option<usize>,
    cli_target_log2: Option<f64>,
    artifact_target_log2: Option<f64>,
) -> Option<f64> {
    [
        cli_target_size.map(target_size_to_log2),
        artifact_target_size.map(target_size_to_log2),
        cli_target_log2,
        artifact_target_log2,
    ]
    .into_iter()
    .flatten()
    .min_by(f64::total_cmp)
}

fn total_execution_log10_flops(per_slice_log10_flops: f64, log2_n_slices: f64) -> f64 {
    per_slice_log10_flops + log2_n_slices * std::f64::consts::LOG10_2
}

/// Convert the logical `|A| + |B| + |C|` element estimate into dtype-aware bytes.
/// This deliberately excludes backend workspace, allocator caches and parallel copies.
fn logical_contraction_byte_estimate(
    log2_elements: f64,
    element_bytes: usize,
) -> Option<(f64, Option<f64>)> {
    if !log2_elements.is_finite() || element_bytes == 0 {
        return None;
    }
    let log2_bytes = log2_elements + (element_bytes as f64).log2();
    let bytes = log2_bytes.exp2();
    Some((log2_bytes, bytes.is_finite().then_some(bytes)))
}

/// Conservatively gate the unsliced oracle used by `--check`.
///
/// Permutation, trace, and summation can allocate their result while the source
/// tensor remains live, so the internal upper bound is
/// `2 * 2^stats.log2_peak_size`. The caller's input tensors and sliced result
/// are also live. Accumulate all terms in log2 space to avoid overflow.
fn full_oracle_fits_byte_budget(
    stats: &arctn::PathStats,
    externally_live_elements: usize,
    element_bytes: usize,
    byte_budget: usize,
) -> bool {
    if !stats.log2_peak_size.is_finite()
        || stats.log2_peak_size < 0.0
        || element_bytes == 0
        || byte_budget == 0
    {
        return false;
    }
    let external_log2 = if externally_live_elements == 0 {
        f64::NEG_INFINITY
    } else {
        (externally_live_elements as f64).log2()
    };
    let total_elements_log2 = arctn::path::logaddexp2(stats.log2_peak_size + 1.0, external_log2);
    let total_bytes_log2 = total_elements_log2 + (element_bytes as f64).log2();
    total_bytes_log2 <= (byte_budget as f64).log2()
}

fn print_usage() {
    println!(
        "Usage: tnexec --net <net.json> [--method rgreedy|greedy] [--trials N] \
         [--seed S] [--load-path tnpath.json] [--target-size ELEMENTS] \
         [--allow-legacy-plan] \
         [--complex] [--single] [--check] \
         [--data <little-endian.bin>]\n\n\
         --target-size bounds the element count of one intermediate tensor per slice; \
         it does not bound concurrent live memory or process RSS.\n\
         Deprecated alias: --mem-target <LOG2_ELEMENTS>."
    );
}

fn parse() -> Args {
    let argv: Vec<String> = std::env::args().collect();
    if argv
        .iter()
        .skip(1)
        .any(|arg| matches!(arg.as_str(), "-h" | "--help"))
    {
        print_usage();
        std::process::exit(0);
    }
    let mut a = Args {
        net_file: String::new(),
        method: "rgreedy".into(),
        method_explicit: false,
        trials: 64,
        trials_explicit: false,
        seed: 1,
        mem_target: None,
        canonical_target_size: None,
        complex: false,
        single: false,
        check: false,
        data_file: None,
        load_path: None,
        allow_legacy_plan: false,
    };
    let mut target_flag: Option<String> = None;
    let val = |i| cli::value::<String>(&argv, i);
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--net" => {
                a.net_file = val(i);
                i += 2;
            }
            "--method" => {
                a.method = val(i);
                a.method_explicit = true;
                i += 2;
            }
            "--trials" => {
                a.trials = cli::value(&argv, i);
                a.trials_explicit = true;
                i += 2;
            }
            "--seed" => {
                a.seed = cli::value(&argv, i);
                i += 2;
            }
            "--target-size" | "--mem-target" => {
                let flag = argv[i].as_str();
                if let Some(previous) = &target_flag {
                    eprintln!(
                        "{previous} 与 {flag} 是同一选项的两种写法，不能同时或重复指定；\
                         请仅使用 --target-size"
                    );
                    std::process::exit(2);
                }
                target_flag = Some(flag.to_owned());
                let raw = val(i);
                if flag == "--target-size" {
                    let target_size = parse_positive_target_size(&raw);
                    a.canonical_target_size = Some(target_size);
                    a.mem_target = Some(target_size_to_log2(target_size));
                } else {
                    eprintln!(
                        "警告: --mem-target <LOG2_ELEMENTS> 已弃用；\
                         请改用 --target-size <ELEMENTS>"
                    );
                    let target_log2 = parse_nonnegative_log2_target(&raw);
                    a.mem_target = Some(target_log2);
                }
                i += 2;
            }
            "--complex" => {
                a.complex = true;
                i += 1;
            }
            "--single" => {
                a.single = true;
                i += 1;
            }
            "--check" => {
                a.check = true;
                i += 1;
            }
            "--data" => {
                a.data_file = Some(val(i));
                i += 2;
            }
            "--load-path" => {
                a.load_path = Some(val(i));
                i += 2;
            }
            "--allow-legacy-plan" => {
                a.allow_legacy_plan = true;
                i += 1;
            }
            other => {
                eprintln!("未知参数: {other}");
                std::process::exit(2);
            }
        }
    }
    if a.net_file.is_empty() {
        print_usage();
        std::process::exit(2);
    }
    if a.load_path.is_some() && (a.method_explicit || a.trials_explicit) {
        eprintln!("--load-path 已交付精确路径，不能再与 --method 或 --trials 并用");
        std::process::exit(2);
    }
    if a.allow_legacy_plan && a.load_path.is_none() {
        eprintln!("--allow-legacy-plan 只能与 --load-path 一起使用");
        std::process::exit(2);
    }
    a
}

fn sliced_path_stats(
    net: &TensorNetwork,
    path: &arctn::SsaPath,
    sliced: &[LegId],
) -> Result<arctn::PathStats, String> {
    arctn::slice::validate_slice_legs(net, sliced)?;
    let mut sliced_net = net.clone();
    for &leg in sliced {
        sliced_net.size_dict.insert(leg, 1);
    }
    simulate_path(&sliced_net, path)
}

fn slice_result_from_parts(
    net: &TensorNetwork,
    legs: Vec<LegId>,
    per_slice: arctn::PathStats,
) -> arctn::slice::SliceResult {
    let log2_n_slices = legs.iter().map(|&leg| net.log2_dim(leg)).sum::<f64>();
    arctn::slice::SliceResult {
        legs,
        log2_n_slices,
        per_slice,
        log10_flops_total: total_execution_log10_flops(per_slice.log10_flops, log2_n_slices),
    }
}

#[allow(clippy::too_many_arguments)]
fn resolve_execution_slices(
    net: &TensorNetwork,
    path: &arctn::SsaPath,
    path_stats: &arctn::PathStats,
    planned_slices: Option<&[LegId]>,
    declared_target_size: Option<usize>,
    declared_target_log2: Option<f64>,
    cli_target_size: Option<usize>,
    cli_target_log2: Option<f64>,
) -> Result<(Vec<LegId>, arctn::PathStats, &'static str), String> {
    let (result, source) = if let Some(exact) = planned_slices {
        let per_slice = sliced_path_stats(net, path, exact)?;
        (
            slice_result_from_parts(net, exact.to_vec(), per_slice),
            "artifact",
        )
    } else if let Some(target) = cli_target_size {
        if arctn::slice::path_fits_target_size(net, path, target)? {
            (
                slice_result_from_parts(net, Vec::new(), *path_stats),
                "none",
            )
        } else {
            let result = arctn::slice::find_slices_to_size(net, path, target).ok_or_else(|| {
                format!(
                    "最大单个中间张量目标 target_size={target} 不可达（裸路径 \
                     log2_max_size={:.6}）",
                    path_stats.log2_max_size
                )
            })?;
            (result, "target-recomputed")
        }
    } else if let Some(target) = cli_target_log2 {
        if path_stats.log2_max_size > target + FEASIBILITY_TOLERANCE_LOG2 {
            let result = find_slices(net, path, target).ok_or_else(|| {
                format!(
                    "最大单个中间张量目标 log2={target:.6} 不可达（裸路径 \
                     log2_max_size={:.6}）",
                    path_stats.log2_max_size
                )
            })?;
            (result, "target-recomputed")
        } else {
            (
                slice_result_from_parts(net, Vec::new(), *path_stats),
                "none",
            )
        }
    } else {
        (
            slice_result_from_parts(net, Vec::new(), *path_stats),
            "none",
        )
    };

    for (constraint_source, target) in
        [("artifact", declared_target_size), ("CLI", cli_target_size)]
    {
        if let Some(target) = target {
            if !arctn::slice::slice_result_fits_target_size(net, path, &result, target)? {
                return Err(format!(
                    "已保存/选定的精确切片计划不满足 {constraint_source} \
                     target_size={target}；执行阶段不会自动替换切片腿"
                ));
            }
        }
    }
    for (constraint_source, target) in
        [("artifact", declared_target_log2), ("CLI", cli_target_log2)]
    {
        if let Some(target) = target {
            if result.per_slice.log2_max_size > target + FEASIBILITY_TOLERANCE_LOG2 {
                return Err(format!(
                    "已保存/选定的精确切片计划不满足 {constraint_source} log2 target: \
                     重新计算得到 log2_max_size={:.6} > target={target:.6}；执行阶段不会自动替换切片腿",
                    result.per_slice.log2_max_size
                ));
            }
        }
    }

    Ok((result.legs, result.per_slice, source))
}

/// Build deterministic random inputs from the seed and tensor index.
fn make_tensors<T: Scalar>(net: &TensorNetwork, seed: u64) -> Vec<DenseTensor<T>> {
    net.inputs
        .iter()
        .enumerate()
        .map(|(k, legs)| {
            let shape: Vec<usize> = legs.iter().map(|&l| net.dim(l)).collect();
            let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0x5eed_0000 ^ (k as u64));
            DenseTensor::random(shape, &mut rng)
        })
        .collect()
}

fn compare_with_oracle<T: OracleScalar>(
    actual: &DenseTensor<T>,
    expected: &DenseTensor<T::Reference>,
    label: &str,
) -> (String, bool) {
    if actual.shape() != expected.shape() {
        return (format!("FAIL({label}_shape_mismatch)"), true);
    }
    // Subtract after promotion too, so the comparison does not introduce
    // another single-precision rounding or overflow in a squared magnitude.
    let diff = actual
        .data()
        .iter()
        .zip(expected.data())
        .map(|(&value, &reference)| (value.to_reference() - reference).abs())
        .fold(0.0f64, |largest, error| {
            if error.is_finite() {
                largest.max(error)
            } else {
                f64::INFINITY
            }
        });
    let scale = expected.data().iter().fold(1.0f64, |largest, &value| {
        let magnitude = value.abs();
        if magnitude.is_finite() {
            largest.max(magnitude)
        } else {
            f64::INFINITY
        }
    });
    let tolerance = CHECK_RELATIVE_TOLERANCE * scale;
    let passed = diff.is_finite() && scale.is_finite() && diff <= tolerance;
    (
        format!(
            "{}({label}_max_abs_diff={diff:.2e};scale={scale:.2e};tolerance={tolerance:.2e};oracle_dtype={})",
            if passed { "PASS" } else { "FAIL" },
            T::REFERENCE_DTYPE,
        ),
        !passed,
    )
}

#[allow(clippy::too_many_arguments)]
fn run<T: OracleScalar>(
    a: &Args,
    net: &TensorNetwork,
    path: &arctn::SsaPath,
    stats: &arctn::PathStats,
    planned_slices: Option<&[LegId]>,
    declared_target_size: Option<usize>,
    declared_target_log2: Option<f64>,
    execution_method: &str,
    plan_verification: &str,
) {
    // An artifact-provided slice set, including an empty set, is bound to its path.
    // A CLI target only validates that exact plan and never replaces its slice legs.
    let (sliced, per_slice_stats, slice_source) = resolve_execution_slices(
        net,
        path,
        stats,
        planned_slices,
        declared_target_size,
        declared_target_log2,
        a.canonical_target_size,
        a.mem_target,
    )
    .unwrap_or_else(|error| {
        eprintln!("切片计划不可执行: {error}");
        std::process::exit(3);
    });
    let n_slices = sliced
        .iter()
        .try_fold(1usize, |acc, &leg| acc.checked_mul(net.dim(leg)))
        .unwrap_or_else(|| {
            eprintln!("切片数乘积溢出 usize，拒绝执行");
            std::process::exit(3);
        });
    let log2_n_slices: f64 = sliced.iter().map(|&leg| net.log2_dim(leg)).sum();
    let log10_flops_total = total_execution_log10_flops(per_slice_stats.log10_flops, log2_n_slices);

    // Load supplied tensor data or generate deterministic inputs for timing.
    // Validate the plan before allocating potentially large input tensors.
    let tensors = match &a.data_file {
        Some(p) => arctn::network::load_tensors_bin::<T>(std::path::Path::new(p), net)
            .unwrap_or_else(|error| {
                eprintln!("读取 --data {p:?} 失败: {error}");
                std::process::exit(2);
            }),
        None => make_tensors::<T>(net, a.seed),
    };

    // Outer concurrency is bounded by both chunk count and the Rayon pool size.
    // Report it because each active slice clones the input tensors.
    let out_numel: usize = net
        .output
        .iter()
        .map(|&l| net.dim(l))
        .product::<usize>()
        .max(1);
    let n_chunks = arctn::slice::slice_parallel_chunks(n_slices, out_numel);
    let slice_par = n_chunks.min(rayon::current_num_threads());
    set_matmul_threads(n_chunks);

    let t0 = Instant::now();
    let result = if sliced.is_empty() {
        contract_network(net, tensors.clone(), path).unwrap_or_else(|error| {
            eprintln!("收缩失败: {error}");
            std::process::exit(3);
        })
    } else {
        contract_network_sliced::<T>(net, &tensors, path, &sliced).unwrap_or_else(|error| {
            eprintln!("切片收缩失败: {error}");
            std::process::exit(3);
        })
    };
    let secs = t0.elapsed().as_secs_f64();

    // Account for every slice when computing executed FLOPs.
    let flops = 10f64.powf(log10_flops_total);
    let gflops = flops / secs.max(1e-12) / 1e9;

    // Prefer the naive oracle for small networks; otherwise use a guarded full contraction.
    let mut check = String::from("skipped");
    let mut check_failed = false;
    if a.check {
        // A byte budget applies the same memory policy across all scalar widths.
        const FULL_ORACLE_BYTE_BUDGET: usize = 512 * 1024 * 1024;
        // Original inputs and result remain live while the promoted reference
        // inputs are allocated. Count both input copies at the wider dtype as
        // a conservative upper bound before allocating the reference inputs.
        let externally_live_elements = tensors.iter().try_fold(result.numel(), |acc, tensor| {
            tensor
                .numel()
                .checked_mul(2)
                .and_then(|input_elements| acc.checked_add(input_elements))
        });
        let full_oracle_fits = externally_live_elements.is_some_and(|external| {
            full_oracle_fits_byte_budget(
                stats,
                external,
                std::mem::size_of::<T::Reference>(),
                FULL_ORACLE_BYTE_BUDGET,
            )
        });
        if net.n_tensors() <= 8 && result.numel() <= (1 << 20) && full_oracle_fits {
            match naive_einsum(net, &reference_tensors(&tensors)) {
                Ok(reference) => {
                    (check, check_failed) = compare_with_oracle(&result, &reference, "naive");
                }
                Err(_) => check = "too_large_for_oracle".into(), // Skip oversized joint-leg spaces.
            }
        } else if !sliced.is_empty()
            && result.numel() <= (1 << 22)
            && n_slices <= (1 << 12)
            && full_oracle_fits
        {
            // Run the unsliced oracle only when its full live set fits the budget.
            let reference = contract_network(net, reference_tensors(&tensors), path)
                .unwrap_or_else(|error| {
                    eprintln!("无切片对照失败: {error}");
                    std::process::exit(3);
                });
            (check, check_failed) = compare_with_oracle(&result, &reference, "slice_oracle");
        } else {
            check = "too_large_for_oracle".into();
        }
    }

    let effective_target_log2 = effective_target_log2(
        a.canonical_target_size,
        declared_target_size,
        a.mem_target,
        declared_target_log2,
    );
    let target_size = effective_target_size(
        a.canonical_target_size,
        declared_target_size,
        a.mem_target,
        declared_target_log2,
    )
    .map(|value| value.to_string())
    .unwrap_or_else(|| "null".to_owned());
    let target_log2_size = effective_target_log2
        .map(|value| format!("{value:.4}"))
        .unwrap_or_else(|| "null".to_owned());
    let dtype_itemsize = std::mem::size_of::<T>();
    let byte_fields = (!path.is_empty())
        .then(|| logical_contraction_byte_estimate(stats.log2_max_contraction_size, dtype_itemsize))
        .flatten();
    let per_slice_byte_fields = (!path.is_empty())
        .then(|| {
            logical_contraction_byte_estimate(
                per_slice_stats.log2_max_contraction_size,
                dtype_itemsize,
            )
        })
        .flatten();
    let log2_max_contraction_bytes = byte_fields
        .map(|(log2_bytes, _)| format!("{log2_bytes:.4}"))
        .unwrap_or_else(|| "null".to_owned());
    let estimated_max_contraction_bytes = byte_fields
        .and_then(|(_, bytes)| bytes)
        .map(|bytes| format!("{bytes:.0}"))
        .unwrap_or_else(|| "null".to_owned());
    let per_slice_log2_max_contraction_bytes = per_slice_byte_fields
        .map(|(log2_bytes, _)| format!("{log2_bytes:.4}"))
        .unwrap_or_else(|| "null".to_owned());
    let per_slice_estimated_max_contraction_bytes = per_slice_byte_fields
        .and_then(|(_, bytes)| bytes)
        .map(|bytes| format!("{bytes:.0}"))
        .unwrap_or_else(|| "null".to_owned());
    println!(
        "EXEC,net={},dtype={},method={},log10_flops={:.4},log2_peak={:.4},\
         log2_max_size={:.4},log2_max_contraction_size={:.4},log2_peak_size={:.4},\
         dtype_itemsize={},contraction_memory_estimate_metric=logical_two_inputs_plus_output_bytes,\
         log2_max_contraction_bytes={},estimated_max_contraction_bytes={},\
         target_size={},target_log2_size={},execution_log10_flops_total={:.4},\
         per_slice_log2_max_size={:.4},per_slice_log2_max_contraction_size={:.4},\
         per_slice_log2_max_contraction_bytes={},per_slice_estimated_max_contraction_bytes={},\
         per_slice_log2_peak_size={:.4},slice_source={},n_sliced_legs={},n_slices={},\
         plan_verification={},slice_par={},out_numel={},t_exec_s={:.4},GFLOPS={:.3},check={}",
        a.net_file,
        match (a.single, a.complex) {
            (true, false) => "f32",
            (true, true) => "c32",
            (false, false) => "f64",
            (false, true) => "c64",
        },
        execution_method,
        // Legacy log10_flops records the unsliced SSA path cost.
        stats.log10_flops,
        // Legacy log2_peak historically stored log2_max_size.
        stats.log2_max_size,
        stats.log2_max_size,
        stats.log2_max_contraction_size,
        stats.log2_peak_size,
        dtype_itemsize,
        log2_max_contraction_bytes,
        estimated_max_contraction_bytes,
        target_size,
        target_log2_size,
        log10_flops_total,
        per_slice_stats.log2_max_size,
        per_slice_stats.log2_max_contraction_size,
        per_slice_log2_max_contraction_bytes,
        per_slice_estimated_max_contraction_bytes,
        per_slice_stats.log2_peak_size,
        slice_source,
        sliced.len(),
        n_slices,
        plan_verification,
        slice_par,
        result.numel(),
        secs,
        gflops,
        check,
    );
    if check_failed {
        eprintln!(
            "tnexec: oracle 校验失败（阈值 {CHECK_RELATIVE_TOLERANCE:.1e} × max(1, 参考结果最大绝对值)）"
        );
        std::process::exit(4);
    }
}

fn main() {
    let a = parse();
    let (net, leg_labels) = TensorNetwork::load_json_with_labels(Path::new(&a.net_file))
        .unwrap_or_else(|error| {
            eprintln!("载入网络 {:?} 失败: {error}", a.net_file);
            std::process::exit(2);
        });

    let (
        path,
        stats,
        planned_slices,
        declared_target_size,
        declared_target_log2,
        execution_method,
        plan_verification,
    ) = if let Some(plan_file) = &a.load_path {
        let text = std::fs::read_to_string(plan_file).unwrap_or_else(|error| {
            eprintln!("读取 --load-path {plan_file:?} 失败: {error}");
            std::process::exit(2);
        });
        let plan = parse_saved_execution_plan(&text, &net, &leg_labels, a.allow_legacy_plan)
            .unwrap_or_else(|error| {
                eprintln!("--load-path {plan_file:?} 不是可执行计划: {error}");
                std::process::exit(2);
            });
        let plan_verification = if plan.legacy_unverified {
            eprintln!(
                "警告: --allow-legacy-plan 正在执行未绑定完整网络身份的旧 artifact；\
                     EXEC 将标记 plan_verification=legacy-unverified"
            );
            "legacy-unverified"
        } else {
            "verified"
        };
        let stats = simulate_path(&net, &plan.path).unwrap_or_else(|error| {
            eprintln!("--load-path 的 ssa_path 与当前网络不匹配: {error}");
            std::process::exit(2);
        });
        (
            plan.path,
            stats,
            plan.sliced,
            plan.declared_target_size,
            plan.declared_target_log2,
            "load-path".to_owned(),
            plan_verification.to_owned(),
        )
    } else {
        // Find a path only when no saved plan was supplied.
        let (path, stats) = match a.method.as_str() {
            "greedy" => greedy(&net),
            "rgreedy" => random_greedy(&net, a.trials, a.seed),
            other => {
                eprintln!("--method 仅支持 greedy|rgreedy: {other}");
                std::process::exit(2);
            }
        }
        .unwrap_or_else(|error| {
            eprintln!("寻路失败: {error}");
            std::process::exit(2);
        });
        (
            path,
            stats,
            None,
            None,
            None,
            a.method.clone(),
            "live-search".to_owned(),
        )
    };

    match (a.single, a.complex) {
        (true, false) => run::<f32>(
            &a,
            &net,
            &path,
            &stats,
            planned_slices.as_deref(),
            declared_target_size,
            declared_target_log2,
            &execution_method,
            &plan_verification,
        ),
        (true, true) => run::<Complex32>(
            &a,
            &net,
            &path,
            &stats,
            planned_slices.as_deref(),
            declared_target_size,
            declared_target_log2,
            &execution_method,
            &plan_verification,
        ),
        (false, false) => run::<f64>(
            &a,
            &net,
            &path,
            &stats,
            planned_slices.as_deref(),
            declared_target_size,
            declared_target_log2,
            &execution_method,
            &plan_verification,
        ),
        (false, true) => run::<Complex64>(
            &a,
            &net,
            &path,
            &stats,
            planned_slices.as_deref(),
            declared_target_size,
            declared_target_log2,
            &execution_method,
            &plan_verification,
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        effective_target_log2, effective_target_size, full_oracle_fits_byte_budget,
        logical_contraction_byte_estimate, parse_saved_execution_plan, resolve_execution_slices,
        slice_result_from_parts, strictest_target_log2, target_size_from_log2, target_size_to_log2,
        total_execution_log10_flops,
    };
    use arctn::{simulate_path, PathStats, TensorNetwork};
    use std::collections::HashMap;

    #[test]
    fn single_precision_reference_error_exceeds_execution_error() {
        let net = TensorNetwork::load_json(std::path::Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/demo_tiny6.net.json"
        )))
        .unwrap();
        let tensors = super::make_tensors::<f32>(&net, 1);
        let wide: Vec<arctn::DenseTensor<f64>> = tensors
            .iter()
            .map(|tensor| {
                arctn::DenseTensor::from_data(
                    tensor.shape().to_vec(),
                    tensor
                        .data()
                        .iter()
                        .map(|&value| f64::from(value))
                        .collect(),
                )
            })
            .collect();
        let (path, _) = arctn::greedy(&net).unwrap();
        let actual = arctn::contract_network(&net, tensors.clone(), &path).unwrap();
        let naive_single = arctn::naive_einsum(&net, &tensors).unwrap();
        let naive_double = arctn::naive_einsum(&net, &wide).unwrap();
        let error = |values: &arctn::DenseTensor<f32>| {
            values
                .data()
                .iter()
                .zip(naive_double.data())
                .map(|(&value, &reference)| (f64::from(value) - reference).abs())
                .fold(0.0f64, f64::max)
        };
        let actual_error = error(&actual);
        let reference_error = error(&naive_single);
        let scale = naive_double
            .data()
            .iter()
            .fold(1.0f64, |maximum, value| maximum.max(value.abs()));
        eprintln!(
            "same f32 inputs: execution vs f64={actual_error:.12e}; naive f32 vs f64={reference_error:.12e}; old difference={:.12e}; tolerance={:.12e}",
            actual.max_abs_diff(&naive_single),
            super::CHECK_RELATIVE_TOLERANCE * scale,
        );
        assert!(actual_error <= super::CHECK_RELATIVE_TOLERANCE * scale);
        assert!(reference_error > super::CHECK_RELATIVE_TOLERANCE * scale);
        let (status, failed) = super::compare_with_oracle(&actual, &naive_double, "naive");
        assert!(!failed, "{status}");
        assert!(status.contains("oracle_dtype=f64"), "{status}");
        let promoted = super::reference_tensors(&tensors);
        for (actual_input, expected_input) in promoted.iter().zip(&wide) {
            assert_eq!(actual_input.data(), expected_input.data());
        }
    }

    #[test]
    fn complex_single_reference_preserves_both_components() {
        use num_complex::{Complex32, Complex64};
        let input = arctn::DenseTensor::from_data(
            vec![2],
            vec![Complex32::new(0.1, -0.2), Complex32::new(-0.3, 0.4)],
        );
        let expected = arctn::DenseTensor::from_data(
            vec![2],
            input
                .data()
                .iter()
                .map(|value| Complex64::new(f64::from(value.re), f64::from(value.im)))
                .collect(),
        );
        let promoted = super::reference_tensors(std::slice::from_ref(&input));
        assert_eq!(promoted[0].data(), expected.data());
        let (status, failed) = super::compare_with_oracle(&input, &expected, "naive");
        assert!(!failed, "{status}");
        assert!(status.contains("oracle_dtype=c64"), "{status}");

        let mut incorrect = input;
        incorrect.data_mut()[0].im += 0.001;
        assert!(super::compare_with_oracle(&incorrect, &expected, "naive").1);
    }

    #[test]
    fn oracle_rejects_corruption_nonfinite_values_and_shape_mismatch() {
        let expected = arctn::DenseTensor::from_data(vec![1], vec![1.0f64]);
        for value in [1.001f32, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let actual = arctn::DenseTensor::from_data(vec![1], vec![value]);
            let (status, failed) = super::compare_with_oracle(&actual, &expected, "naive");
            assert!(failed, "{status}");
        }
        let actual = arctn::DenseTensor::from_data(vec![1], vec![1.0f32]);
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let invalid = arctn::DenseTensor::from_data(vec![1], vec![value]);
            assert!(super::compare_with_oracle(&actual, &invalid, "naive").1);
        }
        let wrong_shape = arctn::DenseTensor::from_data(vec![1, 1], vec![1.0f32]);
        assert!(super::compare_with_oracle(&wrong_shape, &expected, "naive").1);

        let incorrect_double = arctn::DenseTensor::from_data(vec![1], vec![1.001f64]);
        assert!(super::compare_with_oracle(&incorrect_double, &expected, "naive").1);
        let incorrect_complex =
            arctn::DenseTensor::from_data(vec![1], vec![num_complex::Complex64::new(1.0, 0.001)]);
        let expected_complex =
            arctn::DenseTensor::from_data(vec![1], vec![num_complex::Complex64::new(1.0, 0.0)]);
        assert!(super::compare_with_oracle(&incorrect_complex, &expected_complex, "naive").1);
    }

    #[test]
    fn target_size_is_an_element_count() {
        assert_eq!(target_size_to_log2(1), 0.0);
        assert_eq!(target_size_to_log2(1 << 20), 20.0);
        for target_size in [5, 9, 10] {
            let target_log2 = target_size_to_log2(target_size);
            assert_eq!(target_size_from_log2(target_log2), Some(target_size));
            assert_eq!(
                effective_target_size(Some(target_size), None, Some(target_log2), None),
                Some(target_size),
                "canonical integer must remain authoritative"
            );
        }
        // A genuinely fractional legacy log2 target keeps floor(2**target).
        assert_eq!(target_size_from_log2(2.5), Some(5));
        assert_eq!(target_size_from_log2(usize::BITS as f64), None);
        assert!(
            (total_execution_log10_flops(3.0, 10.0) - (3.0 + 10.0 * 2f64.log10())).abs() < 1e-12
        );
        assert_eq!(strictest_target_log2(Some(10.0), Some(9.0)), Some(9.0));
        assert_eq!(target_size_from_log2(9.0), Some(512));
        assert_eq!(
            effective_target_size(Some(10), None, Some(target_size_to_log2(10)), Some(2.0)),
            Some(4),
            "a stricter artifact target must override the canonical CLI integer"
        );
        assert_eq!(
            effective_target_size(Some(10), Some(3), None, None),
            Some(3)
        );
        assert_eq!(
            effective_target_log2(Some(8), Some(4), Some(2.5), Some(1.0)),
            Some(1.0)
        );
    }

    #[test]
    fn contraction_byte_estimate_uses_dtype_width() {
        let (log2_bytes, bytes) = logical_contraction_byte_estimate(31.0_f64.log2(), 16).unwrap();
        assert!((log2_bytes - 496.0_f64.log2()).abs() < 1e-12);
        assert!((bytes.unwrap() - 496.0).abs() < 1e-9);
        assert!(logical_contraction_byte_estimate(1.0, 0).is_none());
        let (huge_log2, huge_bytes) = logical_contraction_byte_estimate(2048.0, 8).unwrap();
        assert_eq!(huge_log2, 2051.0);
        assert!(huge_bytes.is_none(), "绝对字节溢出时仍应保留 log2 bytes");
    }

    #[test]
    fn full_oracle_gate_uses_live_peak_dtype_width_and_external_tensors() {
        let stats = PathStats {
            log10_flops: 0.0,
            // Distinguish the live-peak gate from max-contraction-size accounting.
            log2_max_size: 1.0,
            log2_max_contraction_size: 2.0,
            log2_total_size: 0.0,
            log2_read_write: 0.0,
            log2_peak_size: 30.0,
        };
        assert!(!full_oracle_fits_byte_budget(
            &stats,
            0,
            std::mem::size_of::<f64>(),
            512 * 1024 * 1024,
        ));

        let one_million_live = PathStats {
            log2_peak_size: 20.0,
            ..stats
        };
        let ten_mib = 10 * 1024 * 1024;
        assert!(full_oracle_fits_byte_budget(
            &one_million_live,
            0,
            std::mem::size_of::<f32>(),
            ten_mib,
        ));
        assert!(!full_oracle_fits_byte_budget(
            &one_million_live,
            0,
            std::mem::size_of::<f64>(),
            ten_mib,
        ));
        assert!(!full_oracle_fits_byte_budget(
            &one_million_live,
            1 << 20,
            std::mem::size_of::<f32>(),
            ten_mib,
        ));
        // The source and result coexist, so a 4 MiB peak needs at least 8 MiB.
        assert!(!full_oracle_fits_byte_budget(
            &one_million_live,
            0,
            std::mem::size_of::<f32>(),
            6 * 1024 * 1024,
        ));
    }

    #[test]
    fn modern_identity_and_dual_slice_representation_are_strict() {
        let labels = vec!["b".to_owned(), "a".to_owned(), "c".to_owned()];
        let net = TensorNetwork {
            name: "net".to_owned(),
            inputs: vec![vec![0, 1, 2]],
            output: vec![2],
            size_dict: HashMap::from([(0, 2), (1, 2), (2, 2)]),
        };
        let modern = serde_json::json!({
            "schema": arctn::execution_plan::EXECUTION_PLAN_SCHEMA,
            "schema_version": arctn::execution_plan::EXECUTION_PLAN_V1_VERSION,
            "network_canon": arctn::PathCache::network_canon(&net),
            "network_leg_labels": &labels,
            "net": "net",
            "ssa_path": [],
            "sliced": {"legs": [0], "leg_labels": ["b"]},
            "memory_constraint_metric": "max_intermediate_elements_per_slice",
            "target_size": 127,
            "memory_target_log2_elements": target_size_to_log2(127),
        });
        let verified =
            parse_saved_execution_plan(&modern.to_string(), &net, &labels, false).unwrap();
        assert_eq!(verified.sliced, Some(vec![0]));
        assert_eq!(verified.declared_target_size, Some(127));
        assert_eq!(
            verified.declared_target_log2,
            Some(target_size_to_log2(127))
        );
        assert!(!verified.legacy_unverified);

        let mut empty = modern.clone();
        empty["sliced"]["legs"] = serde_json::json!([]);
        empty["sliced"]["leg_labels"] = serde_json::json!([]);
        assert_eq!(
            parse_saved_execution_plan(&empty.to_string(), &net, &labels, false)
                .unwrap()
                .sliced,
            Some(vec![])
        );

        let mut inconsistent_targets = empty.clone();
        inconsistent_targets["memory_target_log2_elements"] = serde_json::json!(7.0);
        assert!(parse_saved_execution_plan(
            &inconsistent_targets.to_string(),
            &net,
            &labels,
            false,
        )
        .unwrap_err()
        .contains("inconsistent"));

        let mut mismatched = modern;
        mismatched["sliced"]["legs"] = serde_json::json!([1]);
        assert!(
            parse_saved_execution_plan(&mismatched.to_string(), &net, &labels, true,)
                .unwrap_err()
                .contains("same legs")
        );

        let legacy_text = r#"{"ssa_path": [], "sliced": {"legs": [1]}}"#;
        assert!(parse_saved_execution_plan(legacy_text, &net, &labels, false).is_err());
        let legacy = parse_saved_execution_plan(legacy_text, &net, &labels, true).unwrap();
        assert_eq!(legacy.sliced, Some(vec![1]));
        assert!(legacy.legacy_unverified);

        let path_only =
            parse_saved_execution_plan(r#"{"ssa_path": []}"#, &net, &labels, true).unwrap();
        assert!(path_only.sliced.is_none());

        let mut log2_only = empty.clone();
        log2_only.as_object_mut().unwrap().remove("target_size");
        let log2_only =
            parse_saved_execution_plan(&log2_only.to_string(), &net, &labels, false).unwrap();
        assert_eq!(log2_only.declared_target_size, None);
        assert_eq!(
            log2_only.declared_target_log2,
            Some(target_size_to_log2(127))
        );

        let mut exact_only = empty;
        exact_only
            .as_object_mut()
            .unwrap()
            .remove("memory_target_log2_elements");
        let exact_only =
            parse_saved_execution_plan(&exact_only.to_string(), &net, &labels, false).unwrap();
        assert_eq!(exact_only.declared_target_size, Some(127));
        assert_eq!(exact_only.declared_target_log2, None);
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn saved_target_size_above_f64_integer_precision_round_trips_exactly() {
        let labels = vec!["edge".to_owned()];
        let net = TensorNetwork {
            name: "net".to_owned(),
            inputs: vec![vec![0]],
            output: vec![0],
            size_dict: HashMap::from([(0, 2)]),
        };
        let target_size = (1usize << 53) + 1;
        let plan = serde_json::json!({
            "schema": arctn::execution_plan::EXECUTION_PLAN_SCHEMA,
            "schema_version": arctn::execution_plan::EXECUTION_PLAN_V1_VERSION,
            "network_canon": arctn::PathCache::network_canon(&net),
            "network_leg_labels": &labels,
            "net": "net",
            "ssa_path": [],
            "sliced": {"legs": [], "leg_labels": []},
            "memory_constraint_metric": "max_intermediate_elements_per_slice",
            "target_size": target_size,
            "memory_target_log2_elements": (target_size as f64).log2(),
        });
        let parsed = parse_saved_execution_plan(&plan.to_string(), &net, &labels, false).unwrap();
        assert_eq!(parsed.declared_target_size, Some(target_size));
    }

    fn triangle_network() -> (TensorNetwork, arctn::SsaPath) {
        let net = TensorNetwork {
            name: "triangle".to_owned(),
            inputs: vec![vec![0, 1], vec![0, 2], vec![1, 2]],
            output: vec![],
            size_dict: HashMap::from([(0, 2), (1, 2), (2, 2)]),
        };
        // Contract (0, 1) into SSA node 3, then contract it with tensor 2.
        (net, vec![(0, 1), (2, 3)])
    }

    #[test]
    fn exact_saved_slice_set_is_validated_but_never_recomputed() {
        let (net, path) = triangle_network();
        let stats = simulate_path(&net, &path).unwrap();
        assert!(stats.log2_max_size > 1.0);

        // An explicit empty set is complete and must not gain implicit slice legs.
        let exact =
            resolve_execution_slices(&net, &path, &stats, Some(&[]), None, None, None, Some(1.0));
        assert!(exact.unwrap_err().contains("不会自动替换切片腿"));

        // A legacy path-only artifact may derive slices from the CLI target.
        let (legs, per_slice, source) =
            resolve_execution_slices(&net, &path, &stats, None, None, None, None, Some(1.0))
                .unwrap();
        assert!(!legs.is_empty());
        assert!(per_slice.log2_max_size <= 1.0);
        assert_eq!(source, "target-recomputed");
    }

    #[test]
    fn exact_saved_slice_set_is_checked_against_artifact_target() {
        let (net, path) = triangle_network();
        let stats = simulate_path(&net, &path).unwrap();
        let (legs, per_slice, source) =
            resolve_execution_slices(&net, &path, &stats, Some(&[1]), None, Some(1.0), None, None)
                .unwrap();
        assert_eq!(legs, vec![1]);
        assert!(per_slice.log2_max_size <= 1.0);
        assert_eq!(source, "artifact");

        let replay_target = per_slice.log2_max_size - 5e-10;
        assert!(resolve_execution_slices(
            &net,
            &path,
            &stats,
            Some(&[1]),
            None,
            Some(replay_target),
            None,
            None,
        )
        .is_ok());
    }

    #[test]
    fn path_only_plan_uses_exact_integer_recomputation() {
        let (net, path) = triangle_network();
        let stats = simulate_path(&net, &path).unwrap();
        let target_size = 2;
        let (legs, per_slice, source) = resolve_execution_slices(
            &net,
            &path,
            &stats,
            None,
            None,
            None,
            Some(target_size),
            None,
        )
        .unwrap();
        assert!(!legs.is_empty());
        assert_eq!(source, "target-recomputed");
        let result = slice_result_from_parts(&net, legs, per_slice);
        assert!(
            arctn::slice::slice_result_fits_target_size(&net, &path, &result, target_size).unwrap()
        );
    }

    #[test]
    fn exact_and_log2_constraints_are_audited_independently() {
        let (net, path) = triangle_network();
        let stats = simulate_path(&net, &path).unwrap();
        let error = resolve_execution_slices(
            &net,
            &path,
            &stats,
            Some(&[1]),
            Some(2),
            Some(0.5),
            None,
            None,
        )
        .unwrap_err();
        assert!(error.contains("log2 target"));
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn exact_target_rejects_n_plus_one_even_when_log2_tolerance_accepts_it() {
        let target_size = 1usize << 31;
        let produced_size = target_size + 1;
        let target_log2 = (target_size as f64).log2();
        assert!((produced_size as f64).log2() <= target_log2 + super::FEASIBILITY_TOLERANCE_LOG2);

        let net = TensorNetwork {
            name: "exact-boundary".to_owned(),
            inputs: vec![vec![0]],
            output: vec![0],
            size_dict: HashMap::from([(0, produced_size)]),
        };
        let path = vec![];
        let stats = simulate_path(&net, &path).unwrap();
        let error = resolve_execution_slices(
            &net,
            &path,
            &stats,
            Some(&[]),
            Some(target_size),
            Some(target_log2),
            None,
            None,
        )
        .unwrap_err();
        assert!(error.contains("target_size"));
    }
}
