//! Tensor-network pathfinding command-line tool.
//!
//! Supports multiple search methods, Auto presets, path caching, path import
//! and export, and slicing targets. Run with `--help` for the full interface.

mod cli;

use std::path::PathBuf;
use std::time::Instant;

use arctn::greedy;
use arctn::network::TensorNetwork;
use arctn::paths::optimal::{optimal_dp, DEFAULT_MAX_N};
use arctn::tree::reconfigure_path;

const DEFAULT_FLOPS_WEIGHT: f64 = 1.0;
const DEFAULT_READ_WRITE_WEIGHT: f64 = 64.0;

fn print_usage() {
    println!(
        "tnpath {version}\n\
Usage: tnpath <net.json> [OPTIONS]\n\n\
Core options:\n\
  --method <NAME>       auto|greedy|rgreedy|reconf|bisect|budget|optimal|\n\
                        temper|stemper|treesa|orderdp|import|all (default: all)\n\
  --preset <NAME>       light|heavy for --method auto (default: heavy)\n\
  --seed <N>            Random seed (default: 42)\n\
  --trials <N>          Trial count for non-auto methods\n\
  --max-time <SECONDS>  Add an auto wall-clock upper bound\n\
  --flops-weight <W>    Auto FLOPs weight (default: 1)\n\
  --read-write-weight <W>\n\
                        Auto read/write weight (default: 64)\n\
  --target-size <N>     Maximum single-intermediate elements per slice\n\
  --load-path <FILE>    Load an SSA path\n\
  --save-path <FILE>    Save the selected SSA path and metadata\n\
  --ready-file <FILE>   Atomically announce planner readiness\n\
  --go-file <FILE>      Wait up to 60 seconds for timing release after READY\n\
  --quiet               Suppress timed planner diagnostics (benchmark mode)\n\
  --build-info          Print compile-time build identity as JSON\n\
  -h, --help            Print this help\n\
  -V, --version         Print the version\n\n\
Auto accepts --preset light|heavy plus optional --max-time, --target-size, and objective weights.\n\
Light/Heavy requires a separately supplied engine library selected by ARCTN_ENGINE_LIBRARY.\n\
Both objective weights must be finite and non-negative, and cannot both be zero.\n\
The target controls one intermediate tensor, not concurrent live memory or process RSS.\n\
Time budgets are cooperative inside the library. Use an isolated worker process when a\n\
preemptive wall-clock kill is required.",
        version = env!("CARGO_PKG_VERSION")
    );
}

/// Parse a finite, positive floating-point budget.
/// The public Auto entry points validate the same invariant.
fn parse_positive_budget_arg(args: &[String], i: usize, flag: &str, unit: &str) -> f64 {
    match args
        .get(i + 1)
        .and_then(|raw| raw.parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value > 0.0)
    {
        Some(value) => value,
        None => {
            eprintln!("{flag} 必须是有限且大于 0 的数字（{unit}）");
            std::process::exit(2);
        }
    }
}

fn parse_positive_usize_arg(args: &[String], i: usize, flag: &str, unit: &str) -> usize {
    match args
        .get(i + 1)
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|value| *value > 0)
    {
        Some(value) => value,
        None => {
            eprintln!("{flag} 必须是大于 0 的整数（{unit}）");
            std::process::exit(2);
        }
    }
}

fn parse_nonnegative_weight_arg(args: &[String], i: usize, flag: &str) -> f64 {
    match args
        .get(i + 1)
        .and_then(|raw| raw.parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value >= 0.0)
    {
        Some(value) => value,
        None => {
            eprintln!("{flag} 必须是有限且不小于 0 的数字");
            std::process::exit(2);
        }
    }
}

fn auto_cache_config(
    preset: arctn::AutoPreset,
    seed: u64,
    target_size: Option<usize>,
    max_time: Option<f64>,
    objective: arctn::PlannerObjective,
) -> String {
    let rayon_threads = rayon::current_num_threads();
    format!(
        "auto|preset={}|seed={seed}|target_size={target_size:?}|max_time={max_time:?}|objective={}|flops_weight_bits={:016x}|read_write_weight_bits={:016x}|rayon_threads={rayon_threads}",
        preset.as_str(),
        objective.as_str(),
        objective.flops_weight().to_bits(),
        objective.read_write_weight().to_bits(),
    )
}

fn target_size_to_log2(target_size: usize) -> f64 {
    (target_size as f64).log2()
}

fn set_plan_target_fields(
    record: &mut serde_json::Value,
    target_size: Option<usize>,
    target_log2: Option<f64>,
) {
    if target_size.is_some() || target_log2.is_some() {
        record["memory_constraint_metric"] =
            serde_json::json!("max_intermediate_elements_per_slice");
        // Log2-only targets have no exact integer target_size.
        record["target_size"] = serde_json::json!(target_size);
        record["memory_target_log2_elements"] = serde_json::json!(target_log2);
    }
}

fn mark_unique_flag(seen: &mut Option<String>, flag: &str) {
    if let Some(previous) = seen {
        eprintln!("{previous} 与 {flag} 不能重复指定");
        std::process::exit(2);
    }
    *seen = Some(flag.to_owned());
}

fn post_slice_target(method: &str, target_log2: Option<f64>) -> Option<f64> {
    (method != "auto").then_some(target_log2).flatten()
}

fn validate_complete_plan_delivery(
    target_log2: Option<f64>,
    has_slice_handoff: bool,
) -> Result<(), &'static str> {
    if target_log2.is_some() && !has_slice_handoff {
        Err("请求了内存目标，但规划器没有交付与路径配对的 sliced 对象")
    } else {
        Ok(())
    }
}

/// Returns the source identity embedded when the executable was built.
fn build_info_json() -> String {
    let version = serde_json::to_string(env!("CARGO_PKG_VERSION"))
        .expect("package version must serialize as JSON");
    let commit = serde_json::to_string(env!("ARCTN_GIT_COMMIT"))
        .expect("embedded git commit must serialize as JSON");
    let profile = serde_json::to_string(env!("ARCTN_BUILD_PROFILE"))
        .expect("embedded profile must serialize as JSON");
    let source_state =
        serde_json::to_string(option_env!("ARCTN_BUILD_SOURCE_STATE").unwrap_or("unknown"))
            .expect("embedded source state must serialize as JSON");
    let mt = env!("ARCTN_BUILD_FEATURE_MT") == "1";
    let mpi = env!("ARCTN_BUILD_FEATURE_MPI") == "1";
    // JSON-escape the build metadata while preserving field order.
    format!(
        "{{\"version\":{version},\"git_commit\":{commit},\"source_state\":{source_state},\"profile\":{profile},\"features\":{{\"mt\":{mt},\"mpi\":{mpi}}}}}"
    )
}

/// Atomically publish a new `--save-path` artifact without ever replacing an
/// existing directory entry (including a symbolic link).
///
/// The payload is completed and synced under a unique name in the target
/// directory before a single `hard_link` makes it visible.  A hard link, unlike
/// `rename`, also gives us authoritative no-clobber semantics at publication
/// time, closing the race after the early `symlink_metadata` check.
fn atomic_publish_save_path(path: &std::path::Path, payload: &[u8]) -> std::io::Result<()> {
    use std::io::{Error, ErrorKind, Write};
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    match std::fs::symlink_metadata(path) {
        Ok(_) => {
            return Err(Error::new(
                ErrorKind::AlreadyExists,
                format!(
                    "refusing to replace --save-path artifact: {}",
                    path.display()
                ),
            ));
        }
        Err(error) if error.kind() == ErrorKind::NotFound => {}
        Err(error) => return Err(error),
    }

    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| std::path::Path::new("."));
    let name = path
        .file_name()
        .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "--save-path needs a file name"))?
        .to_string_lossy();

    let (temporary, mut file) = (0..128)
        .find_map(|_| {
            let serial = COUNTER.fetch_add(1, Ordering::Relaxed);
            let candidate =
                parent.join(format!(".{name}.save.{}.{serial}.tmp", std::process::id()));
            match std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&candidate)
            {
                Ok(file) => Some(Ok((candidate, file))),
                Err(error) if error.kind() == ErrorKind::AlreadyExists => None,
                Err(error) => Some(Err(error)),
            }
        })
        .transpose()?
        .ok_or_else(|| {
            Error::new(
                ErrorKind::AlreadyExists,
                "could not allocate a unique --save-path temporary file",
            )
        })?;

    let write_result = (|| {
        file.write_all(payload)?;
        file.sync_all()
    })();
    drop(file);
    if let Err(error) = write_result {
        let _ = std::fs::remove_file(&temporary);
        return Err(error);
    }

    if let Err(error) = std::fs::hard_link(&temporary, path) {
        let _ = std::fs::remove_file(&temporary);
        return Err(error);
    }

    // Once linked, never delete the published target: even a cleanup failure
    // must leave complete evidence rather than make an observed artifact vanish.
    let cleanup = std::fs::remove_file(&temporary);
    // Directory fsync is unavailable on some supported filesystems/platforms;
    // publication is already atomic, so make the durability strengthening best-effort.
    if let Ok(directory) = std::fs::File::open(parent) {
        let _ = directory.sync_all();
    }
    cleanup
}

/// Publish the benchmark READY marker only after its bytes are durable. The
/// planner then waits for a parent-created GO marker. This two-phase handshake
/// lets the parent record its monotonic start before releasing the planner;
/// delayed READY observation can therefore never grant extra search time.
/// Publishing with `hard_link` is atomic and never replaces prior evidence.
fn publish_ready_marker(path: &std::path::Path) -> std::io::Result<()> {
    use std::io::{Error, ErrorKind, Write};
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    match std::fs::symlink_metadata(path) {
        Ok(_) => {
            return Err(Error::new(
                ErrorKind::AlreadyExists,
                format!("READY marker already exists: {}", path.display()),
            ));
        }
        Err(error) if error.kind() == ErrorKind::NotFound => {}
        Err(error) => return Err(error),
    }

    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| std::path::Path::new("."));
    let name = path
        .file_name()
        .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "READY path needs a file name"))?
        .to_string_lossy();

    let mut temporary = None;
    let mut marker = None;
    for _ in 0..128 {
        let serial = COUNTER.fetch_add(1, Ordering::Relaxed);
        let candidate = parent.join(format!(".{name}.ready.{}.{serial}.tmp", std::process::id()));
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&candidate)
        {
            Ok(file) => {
                temporary = Some(candidate);
                marker = Some(file);
                break;
            }
            Err(error) if error.kind() == ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        }
    }
    let temporary = temporary.ok_or_else(|| {
        Error::new(
            ErrorKind::AlreadyExists,
            "could not allocate a unique READY temporary file",
        )
    })?;
    let mut marker = marker.expect("temporary path and file are created together");

    let publish = (|| {
        marker.write_all(b"ready\n")?;
        marker.sync_all()?;
        drop(marker);

        // Publish atomically without replacing an existing target.
        std::fs::hard_link(&temporary, path)?;
        if let Err(error) = std::fs::remove_file(&temporary) {
            let _ = std::fs::remove_file(path);
            return Err(error);
        }
        Ok(())
    })();
    if publish.is_err() {
        let _ = std::fs::remove_file(&temporary);
    }
    publish
}

fn wait_for_go_marker(path: &std::path::Path, timeout: std::time::Duration) -> std::io::Result<()> {
    let start = Instant::now();
    loop {
        match std::fs::metadata(path) {
            Ok(metadata) if metadata.len() > 0 => return Ok(()),
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error),
        }
        if start.elapsed() >= timeout {
            return Err(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                "timed out waiting for the GO marker",
            ));
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("missing <net.json>; run `tnpath --help` for usage");
        std::process::exit(2);
    }
    if matches!(args[0].as_str(), "-h" | "--help") {
        print_usage();
        return;
    }
    if matches!(args[0].as_str(), "-V" | "--version") {
        println!("tnpath {}", env!("CARGO_PKG_VERSION"));
        return;
    }
    if args[0] == "--build-info" {
        println!("{}", build_info_json());
        return;
    }
    let mut file: Option<PathBuf> = None;
    let mut method = "all".to_string();
    let mut auto_preset = arctn::AutoPreset::Heavy;
    let mut auto_preset_set = false;
    let mut trials = 64usize;
    // `--trials` remains available to explicitly selected non-Auto methods.
    // Auto uses the selected Light/Heavy preset and rejects this override.
    let mut trials_set = false;
    let mut budget_s: Option<f64> = None;
    let mut max_time_flag: Option<String> = None;
    let mut flops_weight = DEFAULT_FLOPS_WEIGHT;
    let mut read_write_weight = DEFAULT_READ_WRITE_WEIGHT;
    let mut cache_dir: Option<PathBuf> = None;
    // Non-Auto slicing also needs the log2 target.
    let mut mem_target: Option<f64> = None;
    let mut canonical_target_size: Option<usize> = None;
    let mut target_size_flag: Option<String> = None;
    let mut seed = 42u64;
    let mut quiet = false;
    let mut max_n = DEFAULT_MAX_N;
    let mut slice_mode = "reconf".to_string(); // reconf | temper
    let mut slice_mode_set = false;
    let mut reconf_size = 8usize;
    // Tempering controls for `--method temper`.
    let mut chains = 8usize;
    let mut rounds = 40usize;
    let mut moves = 25_000usize;
    let mut tmin = 1e-4f64;
    let mut tmax = 0.15f64;
    let mut reconf_every = 2_000usize;
    let mut patience = 0usize; // Opt-in tempering early stop; zero disables it.
    let mut load_path: Option<PathBuf> = None;
    let mut save_path: Option<PathBuf> = None;
    let mut ready_file: Option<PathBuf> = None;
    let mut go_file: Option<PathBuf> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--load-path" => {
                load_path = Some(cli::value(&args, i));
                i += 2;
            }
            "--save-path" => {
                save_path = Some(cli::value(&args, i));
                i += 2;
            }
            "--ready-file" => {
                ready_file = Some(cli::value(&args, i));
                i += 2;
            }
            "--go-file" => {
                go_file = Some(cli::value(&args, i));
                i += 2;
            }
            "--quiet" => {
                quiet = true;
                i += 1;
            }
            "--chains" => {
                chains = cli::value(&args, i);
                i += 2;
            }
            "--rounds" => {
                rounds = cli::value(&args, i);
                i += 2;
            }
            "--moves" => {
                moves = cli::value(&args, i);
                i += 2;
            }
            "--tmin" => {
                tmin = cli::value(&args, i);
                i += 2;
            }
            "--tmax" => {
                tmax = cli::value(&args, i);
                i += 2;
            }
            "--reconf-every" => {
                reconf_every = cli::value(&args, i);
                i += 2;
            }
            "--patience" => {
                patience = cli::value(&args, i);
                i += 2;
            }
            "--slice-mode" => {
                slice_mode = cli::value(&args, i);
                slice_mode_set = true;
                if slice_mode != "reconf" && slice_mode != "temper" {
                    eprintln!("--slice-mode must be reconf or temper");
                    std::process::exit(2);
                }
                i += 2;
            }
            "--method" => {
                method = cli::value(&args, i);
                i += 2;
            }
            "--preset" => {
                let raw = args.get(i + 1).map(String::as_str).unwrap_or_else(|| {
                    eprintln!("--preset 需要 light|heavy");
                    std::process::exit(2);
                });
                auto_preset = arctn::AutoPreset::from_label(raw).unwrap_or_else(|| {
                    eprintln!("--preset 只支持 light|heavy，收到 {raw:?}");
                    std::process::exit(2);
                });
                auto_preset_set = true;
                i += 2;
            }
            "--trials" => {
                trials = cli::value(&args, i);
                trials_set = true;
                i += 2;
            }
            "--max-time" => {
                mark_unique_flag(&mut max_time_flag, "--max-time");
                budget_s = Some(parse_positive_budget_arg(&args, i, "--max-time", "秒"));
                i += 2;
            }
            "--flops-weight" => {
                flops_weight = parse_nonnegative_weight_arg(&args, i, "--flops-weight");
                i += 2;
            }
            "--read-write-weight" => {
                read_write_weight = parse_nonnegative_weight_arg(&args, i, "--read-write-weight");
                i += 2;
            }
            "--target-size" => {
                mark_unique_flag(&mut target_size_flag, "--target-size");
                let target_size =
                    parse_positive_usize_arg(&args, i, "--target-size", "单切片最大中间张量元素数");
                canonical_target_size = Some(target_size);
                mem_target = Some(target_size_to_log2(target_size));
                i += 2;
            }
            "--cache" => {
                cache_dir = Some(PathBuf::from(".arctn_cache"));
                i += 1;
            }
            "--cache-dir" => {
                cache_dir = Some(cli::value(&args, i));
                i += 2;
            }
            "--seed" => {
                seed = cli::value(&args, i);
                i += 2;
            }
            "--max-n" => {
                max_n = cli::value(&args, i);
                i += 2;
            }
            "--reconf-size" => {
                reconf_size = cli::value(&args, i);
                i += 2;
            }
            other if other.starts_with('-') => {
                eprintln!("未知参数: {other}");
                std::process::exit(2);
            }
            other => {
                if file.is_some() {
                    eprintln!("只能指定一个 net.json 路径，额外参数: {other}");
                    std::process::exit(2);
                }
                file = Some(PathBuf::from(other));
                i += 1;
            }
        }
    }
    let objective =
        arctn::PlannerObjective::new(flops_weight, read_write_weight).unwrap_or_else(|error| {
            eprintln!("规划目标权重无效: {error}");
            std::process::exit(2);
        });
    let default_objective =
        arctn::PlannerObjective::new(DEFAULT_FLOPS_WEIGHT, DEFAULT_READ_WRITE_WEIGHT)
            .expect("default planner objective must be valid");
    if method != "auto" && objective != default_objective {
        eprintln!("--flops-weight/--read-write-weight 的非默认值只能与 --method auto 使用");
        std::process::exit(2);
    }
    if auto_preset_set && method != "auto" {
        eprintln!("--preset 只能与 --method auto 使用");
        std::process::exit(2);
    }
    if method == "auto" && trials_set {
        eprintln!("Auto 不再接受 --trials；请使用 --preset light|heavy");
        std::process::exit(2);
    }
    if method != "auto" && budget_s.is_some() {
        eprintln!("--max-time 只能与 --method auto 使用");
        std::process::exit(2);
    }
    if method != "auto" && cache_dir.is_some() {
        eprintln!("--cache/--cache-dir requires --method auto");
        std::process::exit(2);
    }
    if method == "import" && load_path.is_none() {
        eprintln!("--method import requires --load-path");
        std::process::exit(2);
    }
    if load_path.is_some()
        && !matches!(
            method.as_str(),
            "import" | "reconf" | "temper" | "treesa" | "orderdp" | "all"
        )
    {
        eprintln!("--load-path is not used by method {method}");
        std::process::exit(2);
    }
    if slice_mode_set && method == "auto" {
        eprintln!(
            "--slice-mode 是非 Auto 方法的后切片策略；Auto 的 target-size 入口使用固定路径切片"
        );
        std::process::exit(2);
    }
    if slice_mode_set && method == "budget" {
        eprintln!("--method budget 本身就是路径+切片联合搜索，不接受会被忽略的 --slice-mode");
        std::process::exit(2);
    }
    if method == "budget" && mem_target.is_none() {
        eprintln!("--method budget 需要 --target-size <单切片最大中间张量元素数>");
        std::process::exit(2);
    }
    if slice_mode_set && mem_target.is_none() {
        eprintln!("--slice-mode 需要同时给出 --target-size");
        std::process::exit(2);
    }
    // Auto returns a complete plan for target_size; other methods use the
    // generic post-slicer below.
    let slice_to = post_slice_target(&method, mem_target);
    let exact_slice_to = (method != "auto")
        .then_some(canonical_target_size)
        .flatten();
    if (ready_file.is_some() || go_file.is_some()) && method != "auto" {
        eprintln!("--ready-file/--go-file 只能与 --method auto 使用");
        std::process::exit(2);
    }
    if ready_file.is_some() != go_file.is_some() {
        eprintln!("--ready-file 与 --go-file 必须成对使用");
        std::process::exit(2);
    }
    if ready_file.is_some() && cache_dir.is_some() {
        eprintln!("--ready-file 不能与 --cache/--cache-dir 并用：正式计时必须是 fresh search");
        std::process::exit(2);
    }
    let file = file.unwrap_or_else(|| {
        eprintln!("missing <net.json>; run `tnpath --help` for usage");
        std::process::exit(2);
    });
    // Preserve original leg labels so external evaluators can identify sliced legs.
    let (net, leg_labels) = TensorNetwork::load_json_with_labels(&file).unwrap_or_else(|e| {
        eprintln!("加载失败: {e}");
        std::process::exit(1);
    });
    let network_canon = arctn::pathcache::PathCache::network_canon(&net);
    println!(
        "网络 {}: {} 个张量, {} 条腿, {} 条开放腿",
        net.name,
        net.n_tensors(),
        net.size_dict.len(),
        net.output.len()
    );

    // Imported paths use JSON field `ssa_path: [[a,b], ...]`.
    let loaded: Option<arctn::path::SsaPath> = load_path.as_ref().map(|p| {
        let load = || -> Result<arctn::SsaPath, String> {
            let text = std::fs::read_to_string(p).map_err(|e| e.to_string())?;
            let record: serde_json::Value =
                serde_json::from_str(&text).map_err(|e| e.to_string())?;
            serde_json::from_value(record["ssa_path"].clone()).map_err(|e| e.to_string())
        };
        load().unwrap_or_else(|error| {
            eprintln!("cannot load path {}: {error}", p.display());
            std::process::exit(2);
        })
    });

    // `--save-path` selects by this invocation's top-level objective.
    // Keep candidates in memory and publish once after all methods finish.
    let best_saved = std::cell::Cell::new(f64::INFINITY);
    let best_save_record = std::cell::RefCell::new(None::<String>);
    let delivered = std::cell::Cell::new(false);
    let save_sliced_candidate = |how: &str,
                                 path: &arctn::path::SsaPath,
                                 path_stats: &arctn::path::PathStats,
                                 legs: &[arctn::LegId],
                                 per_slice: &arctn::path::PathStats,
                                 log2_n_slices: f64,
                                 log10_flops_total: f64,
                                 exact_target: Option<usize>,
                                 target: f64| {
        delivered.set(true);
        let selection_score = objective.score_sliced_log2(per_slice, log2_n_slices);
        if save_path.is_none() || selection_score >= best_saved.get() {
            return;
        }
        let labeled: Vec<&str> = legs
            .iter()
            .map(|&leg| {
                leg_labels
                    .get(leg as usize)
                    .map(String::as_str)
                    .expect("切片腿必须有原始标签")
            })
            .collect();
        best_saved.set(selection_score);
        let mut rec = serde_json::json!({
                "schema": arctn::execution_plan::EXECUTION_PLAN_SCHEMA,
                "schema_version": arctn::execution_plan::EXECUTION_PLAN_VERSION,
                "network_canon": network_canon.as_str(),
                "network_leg_labels": &leg_labels,
                "net": net.name,
            "method": how,
            // Top-level metrics describe this path on the unsliced network.
            "log10_flops": path_stats.log10_flops,
            "log2_max_size": path_stats.log2_max_size,
            "log2_max_contraction_size": path_stats.log2_max_contraction_size,
            "log2_total_size": path_stats.log2_total_size,
            "log2_read_write": path_stats.log2_read_write,
            "log2_peak_size": path_stats.log2_peak_size,
            "ssa_path": path,
            "max_intermediate_log2_elements_per_slice": per_slice.log2_max_size,
            "planner_objective": objective.as_str(),
            "flops_weight": objective.flops_weight(),
            "read_write_weight": objective.read_write_weight(),
            "planner_objective_score_log2": selection_score,
            "planner_log2_read_write": per_slice.log2_read_write + log2_n_slices,
            "planner_log2_total_writes": per_slice.log2_total_size + log2_n_slices,
            "sliced": {
                "legs": legs,
                "leg_labels": labeled,
                "log2_n_slices": log2_n_slices,
                "per_slice_log10_flops": per_slice.log10_flops,
                "per_slice_log2_write": per_slice.log2_total_size,
                "per_slice_log2_read_write": per_slice.log2_read_write,
                "per_slice_log2_max_size": per_slice.log2_max_size,
                "per_slice_log2_max_contraction_size": per_slice.log2_max_contraction_size,
                "per_slice_log2_peak_size": per_slice.log2_peak_size,
                "log10_flops_total": log10_flops_total,
            },
        });
        set_plan_target_fields(&mut rec, exact_target, Some(target));
        let rec =
            arctn::complete_execution_plan_v2(&net, &leg_labels, rec).unwrap_or_else(|error| {
                eprintln!("--save-path refused invalid execution plan: {error}");
                std::process::exit(3);
            });
        *best_save_record.borrow_mut() = Some(rec.to_string());
    };
    let run = |name: &str| {
        let t0 = Instant::now();
        let random_search = || arctn::random_greedy(&net, trials, seed);
        let checked_search = |result: Result<_, String>| match result {
            Ok(candidate) => Some(candidate),
            Err(e) => {
                println!("  {name:<8}  失败: {e}");
                None
            }
        };
        let result = match name {
            "greedy" => checked_search(greedy(&net)),
            "rgreedy" => checked_search(random_search()),
            "import" => {
                let p = loaded.clone().expect("--method import 需要 --load-path");
                match arctn::path::simulate_path(&net, &p) {
                    Ok(s) => Some((p, s)),
                    Err(e) => {
                        println!("  import    路径非法: {e}");
                        None
                    }
                }
            }
            "reconf" => {
                let p = match loaded.clone() {
                    Some(p) => p,
                    None => {
                        let Some((p, _)) = checked_search(random_search()) else {
                            return;
                        };
                        p
                    }
                };
                match reconfigure_path(&net, &p, reconf_size, 20) {
                    Ok(r) => Some(r),
                    Err(e) => {
                        println!("  {name:<12}  失败: {e}");
                        None
                    }
                }
            }
            "treesa" => {
                // TreeSA-style annealing starts from `--load-path` or random greedy.
                // `--rounds` supplies sweeps per beta and is capped below.
                let p = match loaded.clone() {
                    Some(p) => p,
                    None => {
                        let Some((p, _)) = checked_search(random_search()) else {
                            return;
                        };
                        p
                    }
                };
                match arctn::tree::treesa_path(
                    &net,
                    &p,
                    chains,
                    0.01,
                    15.0,
                    64,
                    rounds.min(20),
                    reconf_every.min(200),
                    reconf_size,
                    seed,
                )
                .and_then(|(tp, _)| reconfigure_path(&net, &tp, reconf_size, 20))
                {
                    Ok(r) => Some(r),
                    Err(e) => {
                        println!("  treesa    失败: {e}");
                        None
                    }
                }
            }
            "stemper" => {
                // Diversified tempering on the simplified network; zero rounds is adaptive.
                match arctn::simplify::temper_simplified(
                    &net,
                    trials,
                    seed,
                    chains,
                    rounds,
                    moves,
                    tmin,
                    tmax,
                    reconf_every,
                    reconf_size,
                )
                .and_then(|(p, _)| reconfigure_path(&net, &p, reconf_size, 20))
                {
                    Ok(r) => Some(r),
                    Err(e) => {
                        println!("  stemper   失败: {e}");
                        None
                    }
                }
            }
            "orderdp" => {
                // Optimize a fixed leaf order from the imported path or an RCM order.
                use arctn::paths::ordertree::{leaf_order_of_path, order_dp, rcm_order};
                let order = match &loaded {
                    Some(p) => match leaf_order_of_path(&net, p) {
                        Ok(o) => o,
                        Err(e) => {
                            println!("  {name:<8}  叶序提取失败: {e}");
                            return;
                        }
                    },
                    None => rcm_order(&net),
                };
                match order_dp(&net, &order) {
                    Ok(r) => Some(r),
                    Err(e) => {
                        println!("  {name:<8}  失败: {e}");
                        None
                    }
                }
            }
            "temper" => {
                // Start from `--load-path` or the best random-greedy path, then temper and reconfigure.
                let p = match loaded.clone() {
                    Some(p) => p,
                    None => {
                        let Some((p, _)) = checked_search(random_search()) else {
                            return;
                        };
                        p
                    }
                };
                match arctn::tree::temper_paths(
                    &net,
                    std::slice::from_ref(&p),
                    chains,
                    rounds,
                    moves,
                    tmin,
                    tmax,
                    reconf_every,
                    reconf_size,
                    seed,
                    patience,
                )
                .and_then(|(tp, _)| reconfigure_path(&net, &tp, reconf_size, 20))
                {
                    Ok(r) => Some(r),
                    Err(e) => {
                        println!("  temper    失败: {e}");
                        None
                    }
                }
            }
            "bisect" => match arctn::paths::bisect::bisect(&net, trials, seed, 12) {
                Ok((p, _)) => reconfigure_path(&net, &p, reconf_size, 20).ok(),
                Err(e) => {
                    println!("  {name:<8}  跳过: {e}");
                    None
                }
            },
            "budget" => {
                let Some(budget) = slice_to else {
                    println!("  budget    需要 --target-size <元素数>");
                    return;
                };
                match arctn::paths::budgeted::budgeted_portfolio(&net, budget, trials, seed, None) {
                    Ok(r) => {
                        // Convert the element target for the budgeted search API.
                        // Canonical callers finish on the returned path with the exact
                        // integer slicer, so the delivered path+slices pair is exact.
                        let slice = if let Some(target_size) = exact_slice_to {
                            arctn::slice::find_slices_to_size(&net, &r.path, target_size)
                        } else {
                            Some(arctn::slice::SliceResult {
                                legs: r.sliced,
                                log2_n_slices: r.log2_n_slices,
                                per_slice: r.per_slice,
                                log10_flops_total: r.log10_flops_total,
                            })
                        };
                        let Some(sr) = slice else {
                            println!(
                                "  budget    精确 target_size={} 不可达",
                                exact_slice_to.expect("只有 exact 分支可能返回 None")
                            );
                            return;
                        };
                        println!(
                            "  budget    单片 log10(flops)={:>7.3}  最大张量 log2={:>6.2}  {} 条切片腿 → 总 log10(flops)={:.3}",
                            sr.per_slice.log10_flops,
                            sr.per_slice.log2_max_size,
                            sr.legs.len(),
                            sr.log10_flops_total
                        );
                        match arctn::simulate_path(&net, &r.path) {
                            Ok(path_stats) => save_sliced_candidate(
                                "budget",
                                &r.path,
                                &path_stats,
                                &sr.legs,
                                &sr.per_slice,
                                sr.log2_n_slices,
                                sr.log10_flops_total,
                                exact_slice_to,
                                budget,
                            ),
                            Err(error) => println!(
                                "  budget    内部返回的路径在原网络上非法，拒绝保存: {error}"
                            ),
                        }
                        None
                    }
                    Err(e) => {
                        println!("  budget    失败: {e}");
                        None
                    }
                }
            }
            "optimal" => match optimal_dp(&net, max_n) {
                Ok(r) => Some(r),
                Err(e) => {
                    println!("  {name:<8}  跳过: {e}");
                    None
                }
            },
            _ => unreachable!(),
        };
        if let Some((path, stats)) = result {
            println!(
                "  {name:<8}  log10(flops)={:>7.3}  log2(最大张量)={:>6.2}  用时 {:.3}s",
                stats.log10_flops,
                stats.log2_max_size,
                t0.elapsed().as_secs_f64()
            );
            if let Some(target) = slice_to {
                use arctn::slice::{
                    slice_and_reconf, slice_and_reconf_to_size, slice_temper, slice_temper_to_size,
                };
                // Tempering slice mode alternates one-leg slicing with sliced-network tempering.
                // Six rounds per leg is a separate local budget; chains and moves follow the CLI.
                let result = match (slice_mode.as_str(), exact_slice_to) {
                    ("temper", Some(target_size)) => slice_temper_to_size(
                        &net,
                        &path,
                        target_size,
                        seed,
                        chains,
                        6,
                        moves,
                        reconf_size,
                    ),
                    ("temper", None) => {
                        slice_temper(&net, &path, target, seed, chains, 6, moves, reconf_size)
                    }
                    (_, Some(target_size)) => {
                        slice_and_reconf_to_size(&net, &path, target_size, 3, reconf_size)
                    }
                    (_, None) => slice_and_reconf(&net, &path, target, 3, reconf_size),
                };
                match result {
                    Some((sliced_path, sr)) => {
                        println!(
                            "           切片(+{slice_mode}): {} 条腿 → 2^{:.1} 个切片, 单片最大张量 log2={:.2}, 总 log10(flops)={:.3}（开销 ×{:.2}）",
                            sr.legs.len(),
                            sr.log2_n_slices,
                            sr.per_slice.log2_max_size,
                            sr.log10_flops_total,
                            10f64.powf(sr.log10_flops_total - stats.log10_flops),
                        );
                        match arctn::simulate_path(&net, &sliced_path) {
                            Ok(path_stats) => save_sliced_candidate(
                                &format!("{name}+slice-{slice_mode}"),
                                &sliced_path,
                                &path_stats,
                                &sr.legs,
                                &sr.per_slice,
                                sr.log2_n_slices,
                                sr.log10_flops_total,
                                exact_slice_to,
                                target,
                            ),
                            Err(error) => {
                                println!("           切片后路径在原网络上非法，拒绝保存: {error}")
                            }
                        }
                    }
                    None => println!("           切片: 无法达到目标最大张量 2^{target}"),
                }
            } else {
                delivered.set(true);
                let save_objective = objective;
                let save_score = save_objective.score_path_log2(&stats);
                let selection_score = save_score;
                if save_path.is_none() || selection_score >= best_saved.get() {
                    return;
                }
                best_saved.set(selection_score);
                let rec = serde_json::json!({
                    "schema": arctn::execution_plan::EXECUTION_PLAN_SCHEMA,
                    "schema_version": arctn::execution_plan::EXECUTION_PLAN_VERSION,
                    "network_canon": network_canon.as_str(),
                    "network_leg_labels": &leg_labels,
                    "net": net.name, "method": name,
                    "log10_flops": stats.log10_flops,
                    "log2_max_size": stats.log2_max_size,
                    "log2_max_contraction_size": stats.log2_max_contraction_size,
                    "log2_total_size": stats.log2_total_size,
                    "log2_read_write": stats.log2_read_write,
                    "log2_peak_size": stats.log2_peak_size,
                    "ssa_path": path,
                    "planner_objective": save_objective.as_str(),
                    "flops_weight": save_objective.flops_weight(),
                    "read_write_weight": save_objective.read_write_weight(),
                    "planner_objective_score_log2": save_score,
                    "planner_log2_read_write": stats.log2_read_write,
                    "planner_log2_total_writes": stats.log2_total_size,
                    "sliced": {
                        "legs": [],
                        "leg_labels": [],
                        "log2_n_slices": 0.0,
                        "per_slice_log10_flops": stats.log10_flops,
                        "per_slice_log2_write": stats.log2_total_size,
                        "per_slice_log2_read_write": stats.log2_read_write,
                        "per_slice_log2_max_size": stats.log2_max_size,
                        "per_slice_log2_max_contraction_size": stats.log2_max_contraction_size,
                        "per_slice_log2_peak_size": stats.log2_peak_size,
                        "log10_flops_total": stats.log10_flops,
                    },
                });
                let rec = arctn::complete_execution_plan_v2(&net, &leg_labels, rec).unwrap_or_else(
                    |error| {
                        eprintln!("--save-path refused invalid execution plan: {error}");
                        std::process::exit(3);
                    },
                );
                *best_save_record.borrow_mut() = Some(rec.to_string());
            }
        }
    };

    match method.as_str() {
        "auto" => {
            if !quiet {
                eprintln!(
                    "Auto preset={}，seed={}，planner objective={} (flops_weight={}, read_write_weight={})，target-size={:?}，max-time={:?}",
                    auto_preset.as_str(),
                    seed,
                    objective.as_str(),
                    objective.flops_weight(),
                    objective.read_write_weight(),
                    canonical_target_size,
                    budget_s,
                );
            }
            run_auto(
                &net,
                AutoRunConfig {
                    preset: auto_preset,
                    seed,
                    objective,
                    target_size: canonical_target_size,
                    max_time: budget_s,
                    cache_dir,
                    save_path: save_path.as_ref(),
                    leg_labels: &leg_labels,
                    ready_file: ready_file.as_deref(),
                    go_file: go_file.as_deref(),
                    quiet,
                },
            );
        }
        "all" => {
            run("greedy");
            run("rgreedy");
            run("reconf");
            run("bisect");
            run("optimal");
        }
        m @ ("greedy" | "rgreedy" | "reconf" | "bisect" | "budget" | "optimal" | "temper"
        | "stemper" | "treesa" | "import" | "orderdp") => run(m),
        other => {
            eprintln!("unknown method: {other}");
            std::process::exit(2);
        }
    }
    if method != "auto" && !delivered.get() {
        eprintln!("no method produced a valid path satisfying the requested target");
        std::process::exit(3);
    }
    let record_to_publish = best_save_record.borrow().clone();
    if let (Some(sp), Some(record)) = (save_path.as_deref(), record_to_publish.as_deref()) {
        atomic_publish_save_path(sp, record.as_bytes()).unwrap_or_else(|error| {
            eprintln!("--save-path 原子发布失败 {}: {error}", sp.display());
            std::process::exit(2);
        });
        println!("           路径已存 {}", sp.display());
    } else if method != "auto" && save_path.is_some() && mem_target.is_some() {
        eprintln!("--save-path 未发布：没有任何方法交付达标的 path+sliced 完整计划");
        std::process::exit(3);
    }
}

/// Run the selected Light or Heavy Auto preset.
struct AutoRunConfig<'a> {
    preset: arctn::AutoPreset,
    seed: u64,
    objective: arctn::PlannerObjective,
    target_size: Option<usize>,
    max_time: Option<f64>,
    cache_dir: Option<PathBuf>,
    save_path: Option<&'a PathBuf>,
    // Original label indexed by LegId, used when saving sliced legs.
    leg_labels: &'a [String],
    ready_file: Option<&'a std::path::Path>,
    go_file: Option<&'a std::path::Path>,
    quiet: bool,
}

fn run_auto(net: &TensorNetwork, config: AutoRunConfig<'_>) {
    use arctn::pathcache::PathCache;
    use arctn::{auto_path_preset_to_size_with_objective, auto_path_preset_with_objective};
    let AutoRunConfig {
        preset,
        seed,
        objective,
        target_size: canonical_target_size,
        max_time,
        cache_dir,
        save_path,
        leg_labels,
        ready_file,
        go_file,
        quiet,
    } = config;
    let cfg = auto_cache_config(preset, seed, canonical_target_size, max_time, objective);
    let network_canon = PathCache::network_canon(net);
    // Auto save artifacts include sliced legs when a target is requested.
    //
    // Top-level metrics describe the SSA path on the unsliced network.
    // In `sliced`, `per_slice_*` describes one slice and `log10_flops_total`
    // includes all slices. Unsliced artifacts use an explicit empty slice set.
    let save = |path: &arctn::path::SsaPath,
                stats: &arctn::path::PathStats,
                how: &str,
                slice: Option<&arctn::slice::SliceResult>,
                wall_s: Option<f64>| {
        if let Some(sp) = save_path {
            validate_complete_plan_delivery(
                canonical_target_size.map(target_size_to_log2),
                slice.is_some(),
            )
            .unwrap_or_else(|error| {
                eprintln!("--save-path 未发布: {error}");
                std::process::exit(3);
            });
            let planner_objective_score_log2 = slice.map_or_else(
                || objective.score_path_log2(stats),
                |sr| objective.score_sliced_log2(&sr.per_slice, sr.log2_n_slices),
            );
            let planner_log2_total_writes = slice.map_or(stats.log2_total_size, |sr| {
                sr.per_slice.log2_total_size + sr.log2_n_slices
            });
            let planner_log2_read_write = slice.map_or(stats.log2_read_write, |sr| {
                sr.per_slice.log2_read_write + sr.log2_n_slices
            });
            let mut rec = serde_json::json!({
                "schema": arctn::execution_plan::EXECUTION_PLAN_SCHEMA,
                "schema_version": arctn::execution_plan::EXECUTION_PLAN_VERSION,
                "network_canon": network_canon.as_str(),
                "network_leg_labels": leg_labels,
                "net": net.name, "method": how,
                "log10_flops": stats.log10_flops,
                "log2_max_size": stats.log2_max_size,
                "log2_max_contraction_size": stats.log2_max_contraction_size,
                "log2_total_size": stats.log2_total_size,
                "log2_read_write": stats.log2_read_write,
                "log2_peak_size": stats.log2_peak_size,
                "ssa_path": path,
                "preset": preset.as_str(),
                "seed": seed,
                "max_time": max_time,
                "rayon_threads": rayon::current_num_threads(),
                "wall_s": wall_s,
                "planner_objective": objective.as_str(),
                "flops_weight": objective.flops_weight(),
                "read_write_weight": objective.read_write_weight(),
                "planner_objective_score_log2": planner_objective_score_log2,
                "planner_log2_read_write": planner_log2_read_write,
                "planner_log2_total_writes": planner_log2_total_writes,
            });
            if canonical_target_size.is_some() {
                set_plan_target_fields(
                    &mut rec,
                    canonical_target_size,
                    canonical_target_size.map(target_size_to_log2),
                );
                rec["max_intermediate_log2_elements_per_slice"] =
                    serde_json::json!(slice.map(|value| value.per_slice.log2_max_size));
            }
            if let Some(sr) = slice {
                // Store both internal LegIds and original labels for external interoperability.
                let labeled: Vec<&str> = sr
                    .legs
                    .iter()
                    .map(|&l| {
                        leg_labels
                            .get(l as usize)
                            .map(|s| s.as_str())
                            .expect("切片腿必须有原始标签")
                    })
                    .collect();
                rec["sliced"] = serde_json::json!({
                    "legs": sr.legs,
                    "leg_labels": labeled,
                    "log2_n_slices": sr.log2_n_slices,
                    "per_slice_log10_flops": sr.per_slice.log10_flops,
                    "per_slice_log2_write": sr.per_slice.log2_total_size,
                    "per_slice_log2_read_write": sr.per_slice.log2_read_write,
                    "per_slice_log2_max_size": sr.per_slice.log2_max_size,
                    "per_slice_log2_max_contraction_size": sr.per_slice.log2_max_contraction_size,
                    "per_slice_log2_peak_size": sr.per_slice.log2_peak_size,
                    "log10_flops_total": sr.log10_flops_total,
                    "log2_write_total": sr.per_slice.log2_total_size + sr.log2_n_slices,
                    "log2_read_write_total": sr.per_slice.log2_read_write + sr.log2_n_slices,
                    "objective_score_log2": objective
                        .score_sliced_log2(&sr.per_slice, sr.log2_n_slices),
                });
            } else {
                rec["sliced"] = serde_json::json!({
                    "legs": [],
                    "leg_labels": [],
                    "log2_n_slices": 0.0,
                    "per_slice_log10_flops": stats.log10_flops,
                    "per_slice_log2_write": stats.log2_total_size,
                    "per_slice_log2_read_write": stats.log2_read_write,
                    "per_slice_log2_max_size": stats.log2_max_size,
                    "per_slice_log2_max_contraction_size": stats.log2_max_contraction_size,
                    "per_slice_log2_peak_size": stats.log2_peak_size,
                    "log10_flops_total": stats.log10_flops,
                    "log2_write_total": stats.log2_total_size,
                    "log2_read_write_total": stats.log2_read_write,
                    "objective_score_log2": objective.score_path_log2(stats),
                });
            }
            let rec =
                arctn::complete_execution_plan_v2(net, leg_labels, rec).unwrap_or_else(|error| {
                    eprintln!("--save-path refused invalid execution plan: {error}");
                    std::process::exit(3);
                });
            let payload = rec.to_string();
            atomic_publish_save_path(sp, payload.as_bytes()).unwrap_or_else(|error| {
                eprintln!("--save-path 原子发布失败 {}: {error}", sp.display());
                std::process::exit(2);
            });
            if !quiet {
                println!("  路径已存 {}", sp.display());
            }
        }
    };
    let mut cache = match (&cache_dir, canonical_target_size) {
        (Some(_), Some(_)) => {
            // Path-only cache entries cannot satisfy a target that also needs sliced legs.
            if !quiet {
                println!("  （--cache 与 --target-size 并用暂不缓存：缓存中不包含切片计划）");
            }
            None
        }
        (Some(dir), None) => match PathCache::on_disk(dir) {
            Ok(c) => Some(c),
            Err(e) => {
                eprintln!("  缓存目录不可用（{e}），本次不缓存");
                None
            }
        },
        (None, _) => None,
    };
    if let Some(c) = cache.as_mut() {
        let t_hit = Instant::now();
        if let Some(hit) = c.lookup(net, &cfg) {
            if !quiet {
                println!(
                    "  缓存命中（{}）: log10(flops)={:.3}  log2(峰值)={:.2}  objective log2={:.3}  读取用时 {:.4}s",
                    if hit.from_disk { "磁盘" } else { "内存" },
                    hit.stats.log10_flops,
                    hit.stats.log2_peak_size,
                    objective.score_path_log2(&hit.stats),
                    t_hit.elapsed().as_secs_f64()
                );
            }
            // Cached paths and target-size are mutually exclusive. The cache
            // does not preserve the original planning time.
            save(&hit.path, &hit.stats, "auto(缓存命中)", None, None);
            return;
        }
    }
    if let Some(path) = ready_file {
        publish_ready_marker(path).unwrap_or_else(|e| {
            eprintln!("--ready-file 原子发布失败 {}: {e}", path.display());
            std::process::exit(2);
        });
        let go = go_file.expect("READY/GO pairing is validated before loading the network");
        wait_for_go_marker(go, std::time::Duration::from_secs(60)).unwrap_or_else(|e| {
            eprintln!("--go-file 等待失败 {}: {e}", go.display());
            std::process::exit(2);
        });
    }
    let res = match canonical_target_size {
        Some(target_size) => auto_path_preset_to_size_with_objective(
            net,
            preset,
            seed,
            max_time,
            target_size,
            objective,
            true,
        ),
        None => auto_path_preset_with_objective(net, preset, seed, max_time, objective),
    }
    .unwrap_or_else(|e| {
        eprintln!("auto 寻路失败: {e}");
        std::process::exit(2);
    });
    if let Some(target_size) = canonical_target_size {
        let slice = res.sliced.as_ref().unwrap_or_else(|| {
            eprintln!("auto 寻路失败: exact target_size={target_size} 没有交付切片计划");
            std::process::exit(3);
        });
        let feasible =
            arctn::slice::slice_result_fits_target_size(net, &res.path, slice, target_size)
                .unwrap_or_else(|error| {
                    eprintln!("auto exact 切片计划重新计算失败: {error}");
                    std::process::exit(3);
                });
        if !feasible {
            eprintln!("auto 寻路返回了不满足 exact target_size={target_size} 的计划，拒绝发布");
            std::process::exit(3);
        }
    }
    if !quiet {
        let score = res.sliced.as_ref().map_or_else(
            || objective.score_path_log2(&res.stats),
            |slice| objective.score_sliced_log2(&slice.per_slice, slice.log2_n_slices),
        );
        println!(
            "  Auto {}: log10(flops)={:.3}  log2(峰值)={:.2}  objective log2={:.3}  用时 {:.3}s",
            preset.as_str(),
            res.stats.log10_flops,
            res.stats.log2_peak_size,
            score,
            res.wall_s,
        );
        if let Some(sr) = &res.sliced {
            println!(
                "  切片到 target-size: {} 条腿 → 2^{:.1} 个切片, 单片最大中间张量 log2={:.2}, 最大单步 A+B+C log2={:.2}, 总 log10(flops)={:.3}",
                sr.legs.len(),
                sr.log2_n_slices,
                sr.per_slice.log2_max_size,
                sr.per_slice.log2_max_contraction_size,
                sr.log10_flops_total
            );
        }
    }
    if let Some(c) = cache.as_mut() {
        match c.store(net, &cfg, &res.path, preset.as_str()) {
            Ok(_) => {
                if !quiet {
                    println!("  已存入缓存 {}", c.dir().unwrap().display());
                    println!("  （同键命中将复用已保存的路径；测量寻路时间时请勿开启缓存）");
                }
            }
            Err(e) => eprintln!("  缓存写入失败（忽略）: {e}"),
        }
    }
    save(
        &res.path,
        &res.stats,
        "auto",
        res.sliced.as_ref(),
        Some(res.wall_s),
    );
}

#[cfg(test)]
mod build_info_tests {
    use super::{
        atomic_publish_save_path, auto_cache_config, build_info_json, post_slice_target,
        publish_ready_marker, set_plan_target_fields, target_size_to_log2,
        validate_complete_plan_delivery, wait_for_go_marker,
    };

    #[test]
    fn build_info_uses_embedded_build_values() {
        let info: serde_json::Value =
            serde_json::from_str(&build_info_json()).expect("--build-info must be JSON");
        assert_eq!(info["version"], env!("CARGO_PKG_VERSION"));
        assert_eq!(info["git_commit"], env!("ARCTN_GIT_COMMIT"));
        assert_eq!(
            info["source_state"],
            option_env!("ARCTN_BUILD_SOURCE_STATE").unwrap_or("unknown")
        );
        assert_eq!(info["profile"], env!("ARCTN_BUILD_PROFILE"));
        assert_eq!(
            info["features"]["mt"],
            env!("ARCTN_BUILD_FEATURE_MT") == "1"
        );
        assert_eq!(
            info["features"]["mpi"],
            env!("ARCTN_BUILD_FEATURE_MPI") == "1"
        );
    }

    #[test]
    fn one_memory_target_dispatches_to_the_right_slicing_stage() {
        assert_eq!(post_slice_target("auto", Some(12.0)), None);
        assert_eq!(post_slice_target("greedy", Some(12.0)), Some(12.0));
        assert_eq!(post_slice_target("budget", Some(12.0)), Some(12.0));
        assert_eq!(post_slice_target("greedy", None), None);
        assert!(validate_complete_plan_delivery(Some(12.0), false).is_err());
        assert!(validate_complete_plan_delivery(Some(12.0), true).is_ok());
        assert!(validate_complete_plan_delivery(None, false).is_ok());
    }

    #[test]
    fn target_size_uses_element_count_not_log2_input() {
        assert_eq!(target_size_to_log2(1), 0.0);
        assert_eq!(target_size_to_log2(1 << 20), 20.0);
    }

    #[test]
    fn auto_cache_key_uses_exact_objective_weight_bits() {
        let objective_a = arctn::PlannerObjective::new(1.0, 64.0).unwrap();
        let objective_b = arctn::PlannerObjective::new(1.0, 32.0).unwrap();
        let config_a = auto_cache_config(arctn::AutoPreset::Heavy, 42, None, None, objective_a);
        let config_b = auto_cache_config(arctn::AutoPreset::Heavy, 42, None, None, objective_b);

        assert!(config_a.contains(&format!("flops_weight_bits={:016x}", 1.0f64.to_bits())));
        assert!(config_a.contains(&format!(
            "read_write_weight_bits={:016x}",
            64.0f64.to_bits()
        )));
        assert_ne!(config_a, config_b);
    }

    #[test]
    fn auto_cache_key_includes_the_rayon_thread_count() {
        let objective = arctn::PlannerObjective::FIXED;
        let config_for_threads = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| auto_cache_config(arctn::AutoPreset::Heavy, 42, None, None, objective))
        };
        let config_8 = config_for_threads(8);
        let config_16 = config_for_threads(16);

        assert!(config_8.contains("rayon_threads=8"));
        assert!(config_16.contains("rayon_threads=16"));
        assert_ne!(config_8, config_16);
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn plan_target_fields_preserve_exact_integer_above_f64_precision() {
        let target_size = (1usize << 53) + 1;
        let mut record = serde_json::json!({});
        set_plan_target_fields(
            &mut record,
            Some(target_size),
            Some(target_size_to_log2(target_size)),
        );
        assert_eq!(record["target_size"].as_u64(), Some(target_size as u64));
        assert_eq!(
            record["memory_target_log2_elements"].as_f64(),
            Some(target_size_to_log2(target_size))
        );
    }

    #[test]
    fn ready_marker_is_nonempty_create_new_and_leaves_no_temporary_file() {
        use std::time::{SystemTime, UNIX_EPOCH};

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock must be after Unix epoch")
            .as_nanos();
        let directory = std::env::temp_dir().join(format!(
            "arctn-tnpath-ready-test-{}-{nonce}",
            std::process::id()
        ));
        std::fs::create_dir(&directory).expect("create isolated test directory");
        let ready = directory.join("planner.ready");

        publish_ready_marker(&ready).expect("first publication must succeed");
        assert_eq!(std::fs::read(&ready).unwrap(), b"ready\n");
        let error = publish_ready_marker(&ready).expect_err("marker must be create-new");
        assert_eq!(error.kind(), std::io::ErrorKind::AlreadyExists);
        let entries: Vec<_> = std::fs::read_dir(&directory)
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect();
        assert_eq!(entries, vec![std::ffi::OsString::from("planner.ready")]);

        std::fs::remove_file(&ready).unwrap();
        std::fs::remove_dir(&directory).unwrap();
    }

    #[test]
    fn save_path_publication_is_no_clobber_and_leaves_no_temporary_file() {
        use std::time::{SystemTime, UNIX_EPOCH};

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock must be after Unix epoch")
            .as_nanos();
        let directory = std::env::temp_dir().join(format!(
            "arctn-tnpath-save-test-{}-{nonce}",
            std::process::id()
        ));
        std::fs::create_dir(&directory).expect("create isolated test directory");
        let output = directory.join("path.json");

        atomic_publish_save_path(&output, br#"{"ssa_path":[]}"#)
            .expect("first publication must succeed");
        assert_eq!(std::fs::read(&output).unwrap(), br#"{"ssa_path":[]}"#);
        let error = atomic_publish_save_path(&output, b"replacement")
            .expect_err("an existing artifact must never be replaced");
        assert_eq!(error.kind(), std::io::ErrorKind::AlreadyExists);
        assert_eq!(std::fs::read(&output).unwrap(), br#"{"ssa_path":[]}"#);

        let entries: Vec<_> = std::fs::read_dir(&directory)
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect();
        assert_eq!(entries, vec![std::ffi::OsString::from("path.json")]);

        std::fs::remove_dir_all(directory).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn save_path_publication_rejects_even_a_dangling_symlink() {
        use std::os::unix::fs::symlink;
        use std::time::{SystemTime, UNIX_EPOCH};

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock must be after Unix epoch")
            .as_nanos();
        let directory = std::env::temp_dir().join(format!(
            "arctn-tnpath-save-symlink-test-{}-{nonce}",
            std::process::id()
        ));
        std::fs::create_dir(&directory).expect("create isolated test directory");
        let missing = directory.join("missing-target");
        let output = directory.join("path.json");
        symlink(&missing, &output).expect("create dangling output symlink");

        let error = atomic_publish_save_path(&output, b"must-not-follow")
            .expect_err("a symbolic-link target must be rejected");
        assert_eq!(error.kind(), std::io::ErrorKind::AlreadyExists);
        assert!(std::fs::symlink_metadata(&output)
            .unwrap()
            .file_type()
            .is_symlink());
        assert!(!missing.exists());

        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn go_marker_must_be_nonempty_before_release() {
        use std::time::{SystemTime, UNIX_EPOCH};

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock must be after Unix epoch")
            .as_nanos();
        let directory = std::env::temp_dir().join(format!(
            "arctn-tnpath-go-test-{}-{nonce}",
            std::process::id()
        ));
        std::fs::create_dir(&directory).expect("create isolated test directory");
        let go = directory.join("planner.go");
        std::fs::write(&go, []).expect("create an initially empty marker");
        let go_for_thread = go.clone();
        let publisher = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(5));
            std::fs::write(go_for_thread, b"go\n").expect("publish GO contents");
        });
        wait_for_go_marker(&go, std::time::Duration::from_secs(1)).expect("GO wait must complete");
        publisher.join().unwrap();
        assert_eq!(std::fs::read(&go).unwrap(), b"go\n");
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn missing_go_marker_times_out() {
        let path =
            std::env::temp_dir().join(format!("arctn-missing-go-marker-{}", std::process::id()));
        let error = wait_for_go_marker(&path, std::time::Duration::ZERO).unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::TimedOut);
    }
}
