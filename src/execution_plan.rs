//! Versioned, validated execution-plan artifacts.
//!
//! Version 2 embeds the normalized integer-labeled tensor network, an SSA
//! contraction path, and the exact slice-leg set.  It is therefore executable
//! without reconstructing planning decisions.  Version 1 remains readable
//! when the caller supplies the network that the artifact is bound to.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::network::{LegId, TensorNetwork};
use crate::path::{simulate_path, PathStats, SsaPath};
use crate::pathcache::PathCache;
use crate::slice::{slice_result_fits_target_size, validate_slice_legs, SliceResult};

/// Stable execution-plan schema name.
pub const EXECUTION_PLAN_SCHEMA: &str = "arctn-execution-plan";
/// Schema version written by current ArcTN tools.
pub const EXECUTION_PLAN_VERSION: u64 = 2;
/// First version, which binds to an external network by canonical identity.
pub const EXECUTION_PLAN_V1_VERSION: u64 = 1;
/// Path representation used by execution-plan version 2.
pub const EXECUTION_PLAN_PATH_FORMAT: &str = "ssa-v1";
/// Memory metric constrained by `target_size`.
pub const MEMORY_CONSTRAINT_METRIC: &str = "max_intermediate_elements_per_slice";

const FEASIBILITY_TOLERANCE_LOG2: f64 = 1e-9;
const TARGET_DECLARATION_TOLERANCE_LOG2: f64 = 1e-12;
const METRIC_ABSOLUTE_TOLERANCE: f64 = 1e-9;
const METRIC_RELATIVE_TOLERANCE: f64 = 1e-12;

/// Dense integer-labeled network stored in an execution-plan version 2 file.
///
/// `size_dict` is an ordered list rather than a JSON object so leg identifiers
/// remain numbers and duplicate entries can be rejected during validation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionPlanNetwork {
    pub inputs: Vec<Vec<LegId>>,
    pub output: Vec<LegId>,
    pub size_dict: Vec<(LegId, usize)>,
}

impl ExecutionPlanNetwork {
    /// Creates the deterministic serialized representation of a network.
    pub fn from_network(net: &TensorNetwork) -> Self {
        let mut size_dict: Vec<_> = net
            .size_dict
            .iter()
            .map(|(&leg, &dimension)| (leg, dimension))
            .collect();
        size_dict.sort_unstable();
        Self {
            inputs: net.inputs.clone(),
            output: net.output.clone(),
            size_dict,
        }
    }

    /// Restores and validates the normalized network.
    pub fn to_network(&self, name: impl Into<String>) -> Result<TensorNetwork, String> {
        let mut seen = HashSet::new();
        let mut next_leg = 0u32;
        for &leg in self.inputs.iter().flatten().chain(self.output.iter()) {
            if seen.insert(leg) {
                if leg != next_leg {
                    return Err(
                        "execution plan network legs must use dense first-occurrence order"
                            .to_owned(),
                    );
                }
                next_leg = next_leg
                    .checked_add(1)
                    .ok_or_else(|| "execution plan contains too many legs".to_owned())?;
            }
        }
        if self.size_dict.len() != seen.len()
            || self
                .size_dict
                .iter()
                .enumerate()
                .any(|(index, &(leg, _))| usize::try_from(leg).ok() != Some(index))
        {
            return Err(
                "execution plan network.size_dict must contain each dense leg once in order"
                    .to_owned(),
            );
        }
        let mut dimensions = HashMap::with_capacity(self.size_dict.len());
        for &(leg, dimension) in &self.size_dict {
            if dimensions.insert(leg, dimension).is_some() {
                return Err(format!(
                    "execution plan network.size_dict contains duplicate leg {leg}"
                ));
            }
        }
        let net = TensorNetwork {
            name: name.into(),
            inputs: self.inputs.clone(),
            output: self.output.clone(),
            size_dict: dimensions,
        };
        net.validate()
            .map_err(|error| format!("execution plan network is invalid: {error}"))?;
        Ok(net)
    }
}

/// Validated fields needed to execute a saved plan.
#[derive(Clone, Debug)]
pub struct LoadedExecutionPlan {
    pub path: SsaPath,
    /// `Some` means the artifact supplies an exact set, including an empty set.
    /// `None` is limited to schema-less legacy path-only artifacts.
    pub sliced: Option<Vec<LegId>>,
    pub declared_target_size: Option<usize>,
    pub declared_target_log2: Option<f64>,
    pub legacy_unverified: bool,
    pub schema_version: Option<u64>,
}

/// Self-contained validated version 2 plan.
#[derive(Clone, Debug)]
pub struct EmbeddedExecutionPlan {
    pub network: TensorNetwork,
    pub network_leg_labels: Option<Vec<String>>,
    pub execution: LoadedExecutionPlan,
}

/// Completes a `tnpath` metadata record as a validated version 2 artifact.
///
/// The record must already contain `ssa_path` and a `sliced` object with
/// numeric `legs`.  All other non-reserved fields are preserved.
pub fn complete_execution_plan_v2(
    net: &TensorNetwork,
    network_leg_labels: &[String],
    mut record: Value,
) -> Result<Value, String> {
    net.validate()
        .map_err(|error| format!("cannot write execution plan for invalid network: {error}"))?;
    let root = record
        .as_object_mut()
        .ok_or_else(|| "execution plan record root must be an object".to_owned())?;

    let path: SsaPath = serde_json::from_value(
        root.get("ssa_path")
            .cloned()
            .ok_or_else(|| "execution plan record is missing ssa_path".to_owned())?,
    )
    .map_err(|error| format!("execution plan ssa_path is invalid: {error}"))?;
    simulate_path(net, &path)
        .map_err(|error| format!("execution plan ssa_path does not match the network: {error}"))?;

    let sliced = root
        .get_mut("sliced")
        .and_then(Value::as_object_mut)
        .ok_or_else(|| "execution plan record must contain a sliced object".to_owned())?;
    let legs: Vec<LegId> = serde_json::from_value(
        sliced
            .get("legs")
            .cloned()
            .ok_or_else(|| "execution plan sliced object is missing legs".to_owned())?,
    )
    .map_err(|error| format!("execution plan sliced.legs is invalid: {error}"))?;
    validate_slice_legs(net, &legs)
        .map_err(|error| format!("execution plan slice set is invalid: {error}"))?;
    let sliced_labels = labels_for_legs(network_leg_labels, &legs)?;
    sliced.insert(
        "leg_labels".to_owned(),
        serde_json::to_value(sliced_labels).expect("Vec<String> is JSON serializable"),
    );

    root.insert(
        "schema".to_owned(),
        Value::String(EXECUTION_PLAN_SCHEMA.to_owned()),
    );
    root.insert(
        "schema_version".to_owned(),
        Value::Number(EXECUTION_PLAN_VERSION.into()),
    );
    root.insert(
        "path_format".to_owned(),
        Value::String(EXECUTION_PLAN_PATH_FORMAT.to_owned()),
    );
    root.insert(
        "network".to_owned(),
        serde_json::to_value(ExecutionPlanNetwork::from_network(net))
            .expect("ExecutionPlanNetwork is JSON serializable"),
    );
    root.insert(
        "network_canon".to_owned(),
        Value::String(PathCache::network_canon(net)),
    );
    root.insert(
        "network_leg_labels".to_owned(),
        serde_json::to_value(network_leg_labels).expect("string labels are JSON serializable"),
    );
    root.insert("net".to_owned(), Value::String(net.name.clone()));
    root.entry("arctn_version".to_owned())
        .or_insert_with(|| Value::String(env!("CARGO_PKG_VERSION").to_owned()));
    populate_canonical_metadata(root)?;

    let text = serde_json::to_string(&record)
        .map_err(|error| format!("cannot serialize execution plan: {error}"))?;
    parse_execution_plan_for_network(&text, net, network_leg_labels, false)?;
    Ok(record)
}

/// Reads and validates a version 1, version 2, or explicitly allowed legacy plan.
///
/// Version 1 and legacy artifacts need `current_net` because they do not embed
/// an executable network.  Version 2 verifies both its embedded network and its
/// exact equality with `current_net`.
pub fn parse_execution_plan_for_network(
    text: &str,
    current_net: &TensorNetwork,
    current_leg_labels: &[String],
    allow_legacy_plan: bool,
) -> Result<LoadedExecutionPlan, String> {
    current_net
        .validate()
        .map_err(|error| format!("current network is invalid: {error}"))?;
    let value: Value =
        serde_json::from_str(text).map_err(|error| format!("plan JSON is invalid: {error}"))?;
    let root = value
        .as_object()
        .ok_or_else(|| "plan JSON root must be an object".to_owned())?;

    match schema_version(root)? {
        Some(EXECUTION_PLAN_V1_VERSION) => parse_v1(root, current_net, current_leg_labels),
        Some(EXECUTION_PLAN_VERSION) => {
            let embedded = parse_v2(root)?;
            ensure_same_network(current_net, &embedded.network)?;
            if let Some(labels) = &embedded.network_leg_labels {
                if labels != current_leg_labels {
                    return Err(
                        "plan network_leg_labels does not match the current network".to_owned()
                    );
                }
            }
            if let Some(saved_name) = root.get("net") {
                let saved_name = saved_name
                    .as_str()
                    .ok_or_else(|| "plan net field must be a string".to_owned())?;
                if saved_name != current_net.name {
                    return Err(format!(
                        "plan net={saved_name:?} does not match current network {:?}",
                        current_net.name
                    ));
                }
            }
            Ok(embedded.execution)
        }
        Some(version) => Err(format!(
            "unsupported execution plan schema_version: {version}"
        )),
        None => parse_legacy(root, current_net, current_leg_labels, allow_legacy_plan),
    }
}

/// Reads a self-contained version 2 artifact and validates all execution fields.
pub fn parse_embedded_execution_plan(text: &str) -> Result<EmbeddedExecutionPlan, String> {
    let value: Value =
        serde_json::from_str(text).map_err(|error| format!("plan JSON is invalid: {error}"))?;
    let root = value
        .as_object()
        .ok_or_else(|| "plan JSON root must be an object".to_owned())?;
    match schema_version(root)? {
        Some(EXECUTION_PLAN_VERSION) => parse_v2(root),
        Some(version) => Err(format!(
            "execution plan version {version} is not self-contained; version 2 is required"
        )),
        None => Err("schema-less execution plans are not self-contained".to_owned()),
    }
}

fn schema_version(root: &Map<String, Value>) -> Result<Option<u64>, String> {
    match (root.get("schema"), root.get("schema_version")) {
        (None, None) => Ok(None),
        (Some(schema), Some(version)) => {
            if schema.as_str() != Some(EXECUTION_PLAN_SCHEMA) {
                return Err(format!("unsupported execution plan schema: {schema}"));
            }
            let version = version
                .as_u64()
                .ok_or_else(|| "execution plan schema_version must be an integer".to_owned())?;
            Ok(Some(version))
        }
        _ => Err("execution plan must contain both schema and schema_version".to_owned()),
    }
}

fn populate_canonical_metadata(root: &mut Map<String, Value>) -> Result<(), String> {
    let mut metrics = take_object_or_empty(root, "metrics")?;
    copy_present(
        root,
        &mut metrics,
        &[
            "log10_flops",
            "log2_max_size",
            "log2_max_contraction_size",
            "log2_total_size",
            "log2_read_write",
            "log2_peak_size",
            "max_intermediate_log2_elements_per_slice",
        ],
    );
    if let Some(sliced) = root.get("sliced").and_then(Value::as_object) {
        copy_as(sliced, &mut metrics, "log2_n_slices", "log2_n_slices");
        copy_as(
            sliced,
            &mut metrics,
            "log10_flops_total",
            "sliced_log10_flops_total",
        );
        copy_as(
            sliced,
            &mut metrics,
            "per_slice_log2_max_size",
            "sliced_log2_max_size",
        );
        copy_as(
            sliced,
            &mut metrics,
            "per_slice_log2_max_contraction_size",
            "sliced_log2_max_contraction_size",
        );
        copy_as(
            sliced,
            &mut metrics,
            "per_slice_log2_peak_size",
            "sliced_log2_peak_size",
        );
    }
    root.insert("metrics".to_owned(), Value::Object(metrics));

    let mut planning = take_object_or_empty(root, "planning")?;
    copy_present(
        root,
        &mut planning,
        &[
            "preset",
            "planner_objective",
            "flops_weight",
            "read_write_weight",
            "slicing_mode",
            "generator_best_source",
            "chosen_path_stage",
            "method",
            "seed",
        ],
    );
    if !planning.contains_key("seed") {
        if let Some(seed) = root
            .get("multi_start")
            .and_then(Value::as_object)
            .and_then(|multi_start| multi_start.get("base_seed"))
            .filter(|value| !value.is_null())
        {
            planning.insert("seed".to_owned(), seed.clone());
        }
    }
    root.insert("planning".to_owned(), Value::Object(planning));

    let mut provenance = take_object_or_empty(root, "provenance")?;
    copy_present(root, &mut provenance, &["arctn_version"]);
    provenance
        .entry("source".to_owned())
        .or_insert_with(|| Value::String("tnpath".to_owned()));
    root.insert("provenance".to_owned(), Value::Object(provenance));
    Ok(())
}

fn take_object_or_empty(
    root: &mut Map<String, Value>,
    key: &str,
) -> Result<Map<String, Value>, String> {
    match root.remove(key) {
        None | Some(Value::Null) => Ok(Map::new()),
        Some(Value::Object(object)) => Ok(object),
        Some(_) => Err(format!("execution plan {key} must be an object")),
    }
}

fn copy_present(source: &Map<String, Value>, target: &mut Map<String, Value>, keys: &[&str]) {
    for &key in keys {
        copy_as(source, target, key, key);
    }
}

fn copy_as(
    source: &Map<String, Value>,
    target: &mut Map<String, Value>,
    source_key: &str,
    target_key: &str,
) {
    if target.contains_key(target_key) {
        return;
    }
    if let Some(value) = source.get(source_key).filter(|value| !value.is_null()) {
        target.insert(target_key.to_owned(), value.clone());
    }
}

fn parse_v2(root: &Map<String, Value>) -> Result<EmbeddedExecutionPlan, String> {
    if root.get("path_format").and_then(Value::as_str) != Some(EXECUTION_PLAN_PATH_FORMAT) {
        return Err(format!(
            "unsupported execution plan path_format: {}",
            root.get("path_format").unwrap_or(&Value::Null)
        ));
    }
    let saved_name = match root.get("net") {
        Some(value) => value
            .as_str()
            .ok_or_else(|| "plan net field must be a string".to_owned())?
            .to_owned(),
        None => "embedded-plan".to_owned(),
    };
    let stored_network: ExecutionPlanNetwork = serde_json::from_value(
        root.get("network")
            .cloned()
            .ok_or_else(|| "execution plan version 2 is missing network".to_owned())?,
    )
    .map_err(|error| format!("execution plan network is invalid: {error}"))?;
    let network = stored_network.to_network(saved_name)?;
    let expected_canon = PathCache::network_canon(&network);
    let saved_canon = root
        .get("network_canon")
        .and_then(Value::as_str)
        .ok_or_else(|| "execution plan network_canon must be a string".to_owned())?;
    if saved_canon != expected_canon {
        return Err("execution plan network_canon does not match embedded network".to_owned());
    }

    let network_leg_labels: Option<Vec<String>> = root
        .get("network_leg_labels")
        .map(|raw| {
            serde_json::from_value(raw.clone())
                .map_err(|error| format!("network_leg_labels is invalid: {error}"))
        })
        .transpose()?;
    if let Some(labels) = &network_leg_labels {
        validate_label_table(&network, labels)?;
    }

    let path = parse_path(root)?;
    let path_stats = simulate_path(&network, &path)
        .map_err(|error| format!("execution plan ssa_path is invalid: {error}"))?;
    let sliced = parse_slice_set(root, &network, network_leg_labels.as_deref(), true)?
        .expect("version 2 requires an explicit sliced object");
    validate_reported_metrics(root, &network, &path, &path_stats, &sliced)?;
    let (declared_target_size, declared_target_log2) = parse_target_declaration(root, true)?;
    validate_target(
        &network,
        &path,
        &sliced,
        declared_target_size,
        declared_target_log2,
    )?;

    Ok(EmbeddedExecutionPlan {
        network,
        network_leg_labels,
        execution: LoadedExecutionPlan {
            path,
            sliced: Some(sliced),
            declared_target_size,
            declared_target_log2,
            legacy_unverified: false,
            schema_version: Some(EXECUTION_PLAN_VERSION),
        },
    })
}

fn validate_reported_metrics(
    root: &Map<String, Value>,
    net: &TensorNetwork,
    path: &SsaPath,
    path_stats: &PathStats,
    sliced: &[LegId],
) -> Result<(), String> {
    let path_metrics = [
        ("log10_flops", path_stats.log10_flops),
        ("log2_max_size", path_stats.log2_max_size),
        (
            "log2_max_contraction_size",
            path_stats.log2_max_contraction_size,
        ),
        ("log2_total_size", path_stats.log2_total_size),
        ("log2_read_write", path_stats.log2_read_write),
        ("log2_peak_size", path_stats.log2_peak_size),
    ];
    validate_metric_fields(root, "execution plan", &path_metrics)?;

    let metrics = match root.get("metrics") {
        None => None,
        Some(Value::Object(metrics)) => Some(metrics),
        Some(_) => return Err("execution plan metrics must be an object".to_owned()),
    };
    if let Some(metrics) = metrics {
        validate_metric_fields(metrics, "execution plan metrics", &path_metrics)?;
    }

    let per_slice = sliced_path_stats(net, path, sliced)?;
    let log2_n_slices: f64 = sliced.iter().map(|&leg| net.log2_dim(leg)).sum();
    let sliced_log10_flops_total =
        per_slice.log10_flops + log2_n_slices * std::f64::consts::LOG10_2;
    let canonical_sliced_metrics = [
        ("log2_n_slices", log2_n_slices),
        ("sliced_log10_flops_total", sliced_log10_flops_total),
        ("sliced_log2_max_size", per_slice.log2_max_size),
        (
            "sliced_log2_max_contraction_size",
            per_slice.log2_max_contraction_size,
        ),
        ("sliced_log2_peak_size", per_slice.log2_peak_size),
        (
            "max_intermediate_log2_elements_per_slice",
            per_slice.log2_max_size,
        ),
    ];
    validate_metric_fields(root, "execution plan", &canonical_sliced_metrics)?;
    if let Some(metrics) = metrics {
        validate_metric_fields(metrics, "execution plan metrics", &canonical_sliced_metrics)?;
    }

    let sliced_object = root
        .get("sliced")
        .and_then(Value::as_object)
        .expect("version 2 slice object was validated before metrics");
    let sliced_metrics = [
        ("log2_n_slices", log2_n_slices),
        ("per_slice_log10_flops", per_slice.log10_flops),
        ("per_slice_log2_write", per_slice.log2_total_size),
        ("per_slice_log2_read_write", per_slice.log2_read_write),
        ("per_slice_log2_max_size", per_slice.log2_max_size),
        (
            "per_slice_log2_max_contraction_size",
            per_slice.log2_max_contraction_size,
        ),
        ("per_slice_log2_peak_size", per_slice.log2_peak_size),
        ("log10_flops_total", sliced_log10_flops_total),
        (
            "log2_write_total",
            per_slice.log2_total_size + log2_n_slices,
        ),
        (
            "log2_read_write_total",
            per_slice.log2_read_write + log2_n_slices,
        ),
    ];
    validate_metric_fields(sliced_object, "execution plan sliced", &sliced_metrics)
}

fn validate_metric_fields(
    object: &Map<String, Value>,
    location: &str,
    expected: &[(&str, f64)],
) -> Result<(), String> {
    for &(name, expected_value) in expected {
        let Some(raw) = object.get(name).filter(|value| !value.is_null()) else {
            continue;
        };
        let actual = raw
            .as_f64()
            .filter(|value| value.is_finite())
            .ok_or_else(|| format!("{location}.{name} must be a finite number"))?;
        let tolerance = METRIC_ABSOLUTE_TOLERANCE
            .max(METRIC_RELATIVE_TOLERANCE * expected_value.abs().max(actual.abs()));
        if !expected_value.is_finite() || (actual - expected_value).abs() > tolerance {
            return Err(format!(
                "{location}.{name}={actual} does not match the recomputed value {expected_value}"
            ));
        }
    }
    Ok(())
}

fn parse_v1(
    root: &Map<String, Value>,
    current_net: &TensorNetwork,
    current_leg_labels: &[String],
) -> Result<LoadedExecutionPlan, String> {
    let saved_name = root
        .get("net")
        .and_then(Value::as_str)
        .ok_or_else(|| "execution plan version 1 is missing string field net".to_owned())?;
    if saved_name != current_net.name {
        return Err(format!(
            "plan net={saved_name:?} does not match current network {:?}",
            current_net.name
        ));
    }
    let saved_canon = root
        .get("network_canon")
        .and_then(Value::as_str)
        .ok_or_else(|| "execution plan version 1 network_canon must be a string".to_owned())?;
    if saved_canon != PathCache::network_canon(current_net) {
        return Err("plan network_canon does not match the current network".to_owned());
    }
    let labels: Vec<String> = serde_json::from_value(
        root.get("network_leg_labels")
            .cloned()
            .ok_or_else(|| "execution plan version 1 is missing network_leg_labels".to_owned())?,
    )
    .map_err(|error| format!("network_leg_labels is invalid: {error}"))?;
    if labels != current_leg_labels {
        return Err("plan network_leg_labels does not match the current network".to_owned());
    }

    let path = parse_path(root)?;
    simulate_path(current_net, &path)
        .map_err(|error| format!("execution plan ssa_path is invalid: {error}"))?;
    let sliced_object = root
        .get("sliced")
        .and_then(Value::as_object)
        .ok_or_else(|| "execution plan version 1 requires a sliced object".to_owned())?;
    if !sliced_object.contains_key("legs") || !sliced_object.contains_key("leg_labels") {
        return Err(
            "execution plan version 1 sliced must contain both legs and leg_labels".to_owned(),
        );
    }
    let sliced = parse_slice_set(root, current_net, Some(current_leg_labels), true)?
        .expect("version 1 requires an explicit sliced object");
    let (declared_target_size, declared_target_log2) = parse_target_declaration(root, false)?;
    validate_target(
        current_net,
        &path,
        &sliced,
        declared_target_size,
        declared_target_log2,
    )?;
    Ok(LoadedExecutionPlan {
        path,
        sliced: Some(sliced),
        declared_target_size,
        declared_target_log2,
        legacy_unverified: false,
        schema_version: Some(EXECUTION_PLAN_V1_VERSION),
    })
}

fn parse_legacy(
    root: &Map<String, Value>,
    current_net: &TensorNetwork,
    current_leg_labels: &[String],
    allow_legacy_plan: bool,
) -> Result<LoadedExecutionPlan, String> {
    if root.contains_key("network_canon") || root.contains_key("network_leg_labels") {
        return Err(
            "execution plan network identity is incomplete; legacy mode cannot bypass a partially written schema"
                .to_owned(),
        );
    }
    if !allow_legacy_plan {
        return Err(
            "legacy plan lacks schema and network identity; pass --allow-legacy-plan only for a trusted artifact"
                .to_owned(),
        );
    }
    if let Some(saved_name) = root.get("net") {
        let saved_name = saved_name
            .as_str()
            .ok_or_else(|| "legacy plan net field must be a string".to_owned())?;
        if saved_name != current_net.name {
            return Err(format!(
                "plan net={saved_name:?} does not match current network {:?}",
                current_net.name
            ));
        }
    }
    let path = parse_path(root)?;
    simulate_path(current_net, &path)
        .map_err(|error| format!("legacy plan ssa_path is invalid: {error}"))?;
    let sliced = parse_slice_set(root, current_net, Some(current_leg_labels), false)?;
    let (declared_target_size, declared_target_log2) = parse_target_declaration(root, false)?;
    if (declared_target_size.is_some() || declared_target_log2.is_some()) && sliced.is_none() {
        return Err(
            "plan declares a memory target but does not provide a sliced object".to_owned(),
        );
    }
    if let Some(legs) = &sliced {
        validate_target(
            current_net,
            &path,
            legs,
            declared_target_size,
            declared_target_log2,
        )?;
    }
    Ok(LoadedExecutionPlan {
        path,
        sliced,
        declared_target_size,
        declared_target_log2,
        legacy_unverified: true,
        schema_version: None,
    })
}

fn parse_path(root: &Map<String, Value>) -> Result<SsaPath, String> {
    serde_json::from_value(
        root.get("ssa_path")
            .cloned()
            .ok_or_else(|| "execution plan is missing ssa_path".to_owned())?,
    )
    .map_err(|error| format!("ssa_path is invalid: {error}"))
}

fn parse_slice_set(
    root: &Map<String, Value>,
    net: &TensorNetwork,
    network_leg_labels: Option<&[String]>,
    require_explicit: bool,
) -> Result<Option<Vec<LegId>>, String> {
    let Some(raw) = root.get("sliced").filter(|value| !value.is_null()) else {
        if require_explicit {
            return Err(
                "versioned execution plans must provide sliced; an unsliced plan uses empty legs"
                    .to_owned(),
            );
        }
        return Ok(None);
    };
    let object = raw
        .as_object()
        .ok_or_else(|| "sliced must be an object".to_owned())?;
    let numeric_legs: Option<Vec<LegId>> = object
        .get("legs")
        .map(|raw| {
            serde_json::from_value(raw.clone())
                .map_err(|error| format!("sliced.legs is invalid: {error}"))
        })
        .transpose()?;
    let labeled_legs: Option<Vec<String>> = object
        .get("leg_labels")
        .map(|raw| {
            serde_json::from_value(raw.clone())
                .map_err(|error| format!("sliced.leg_labels is invalid: {error}"))
        })
        .transpose()?;

    if require_explicit && numeric_legs.is_none() {
        return Err("versioned execution plans require sliced.legs".to_owned());
    }
    let legs = if let Some(ids) = numeric_legs {
        if let Some(labels) = labeled_legs {
            let table = network_leg_labels.ok_or_else(|| {
                "sliced.leg_labels cannot be checked without network_leg_labels".to_owned()
            })?;
            let expected = labels_for_legs(table, &ids)?;
            if expected != labels {
                return Err(
                    "sliced.legs and sliced.leg_labels do not identify the same legs".to_owned(),
                );
            }
        }
        ids
    } else if let Some(labels) = labeled_legs {
        let table = network_leg_labels.ok_or_else(|| {
            "sliced.leg_labels cannot be resolved without network_leg_labels".to_owned()
        })?;
        let label_to_leg = unique_label_map(table)?;
        labels
            .iter()
            .map(|label| {
                label_to_leg
                    .get(label.as_str())
                    .copied()
                    .ok_or_else(|| format!("sliced leg label {label:?} is not in the network"))
            })
            .collect::<Result<Vec<_>, _>>()?
    } else {
        return Err("sliced object contains neither legs nor leg_labels".to_owned());
    };
    validate_slice_legs(net, &legs).map_err(|error| format!("invalid slice set: {error}"))?;
    Ok(Some(legs))
}

fn parse_target_declaration(
    root: &Map<String, Value>,
    require_metric: bool,
) -> Result<(Option<usize>, Option<f64>), String> {
    let exact = match root.get("target_size") {
        None | Some(Value::Null) => None,
        Some(raw) => Some(
            raw.as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .filter(|value| *value > 0)
                .ok_or_else(|| "target_size must be a positive usize integer".to_owned())?,
        ),
    };
    let log2 = match root.get("memory_target_log2_elements") {
        None | Some(Value::Null) => None,
        Some(raw) => {
            let value = raw.as_f64().ok_or_else(|| {
                "memory_target_log2_elements must be a finite non-negative number".to_owned()
            })?;
            if !value.is_finite() || value < 0.0 {
                return Err(
                    "memory_target_log2_elements must be a finite non-negative number".to_owned(),
                );
            }
            Some(value)
        }
    };
    if let (Some(exact), Some(log2)) = (exact, log2) {
        let expected = (exact as f64).log2();
        if (log2 - expected).abs() > TARGET_DECLARATION_TOLERANCE_LOG2 {
            return Err(format!(
                "target_size={exact} is inconsistent with memory_target_log2_elements={log2}; expected {expected}"
            ));
        }
    }
    let metric = root.get("memory_constraint_metric");
    if require_metric && exact.is_none() && (log2.is_some() || metric.is_some()) {
        return Err(
            "execution plan version 2 requires exact target_size for every memory target declaration"
            .to_owned(),
        );
    }
    if require_metric && exact.is_some() && log2.is_none() {
        return Err(
            "execution plan version 2 target_size requires memory_target_log2_elements".to_owned(),
        );
    }
    if exact.is_some() || log2.is_some() {
        if (require_metric || metric.is_some())
            && metric.and_then(Value::as_str) != Some(MEMORY_CONSTRAINT_METRIC)
        {
            return Err(format!(
                "memory_constraint_metric must be {MEMORY_CONSTRAINT_METRIC:?} when a target is declared"
            ));
        }
    } else if require_metric && metric.is_some() {
        return Err("memory_constraint_metric requires target_size".to_owned());
    }
    Ok((exact, log2))
}

fn validate_target(
    net: &TensorNetwork,
    path: &SsaPath,
    sliced: &[LegId],
    exact: Option<usize>,
    log2: Option<f64>,
) -> Result<(), String> {
    if exact.is_none() && log2.is_none() {
        return Ok(());
    }
    let per_slice = sliced_path_stats(net, path, sliced)?;
    if let Some(target) = exact {
        let result = SliceResult {
            legs: sliced.to_vec(),
            log2_n_slices: sliced.iter().map(|&leg| net.log2_dim(leg)).sum(),
            per_slice,
            log10_flops_total: per_slice.log10_flops,
        };
        if !slice_result_fits_target_size(net, path, &result, target)? {
            return Err(format!(
                "execution plan exceeds declared target_size={target}"
            ));
        }
    } else if let Some(target) = log2 {
        if per_slice.log2_max_size > target + FEASIBILITY_TOLERANCE_LOG2 {
            return Err(format!(
                "execution plan per-slice maximum {} exceeds declared log2 target {target}",
                per_slice.log2_max_size
            ));
        }
    }
    Ok(())
}

fn sliced_path_stats(
    net: &TensorNetwork,
    path: &SsaPath,
    sliced: &[LegId],
) -> Result<PathStats, String> {
    validate_slice_legs(net, sliced)?;
    let mut sliced_net = net.clone();
    for &leg in sliced {
        sliced_net.size_dict.insert(leg, 1);
    }
    simulate_path(&sliced_net, path)
}

fn validate_label_table(net: &TensorNetwork, labels: &[String]) -> Result<(), String> {
    if labels.len() != net.size_dict.len() {
        return Err("network_leg_labels length must match the embedded dense network".to_owned());
    }
    let mut referenced = HashSet::new();
    referenced.extend(net.inputs.iter().flatten().copied());
    referenced.extend(net.output.iter().copied());
    for leg in referenced {
        if labels.get(leg as usize).is_none() {
            return Err(format!(
                "network_leg_labels has no entry for embedded LegId {leg}"
            ));
        }
    }
    unique_label_map(labels)?;
    Ok(())
}

fn labels_for_legs(labels: &[String], legs: &[LegId]) -> Result<Vec<String>, String> {
    legs.iter()
        .map(|&leg| {
            labels
                .get(leg as usize)
                .cloned()
                .ok_or_else(|| format!("network_leg_labels has no entry for LegId {leg}"))
        })
        .collect()
}

fn unique_label_map(labels: &[String]) -> Result<HashMap<&str, LegId>, String> {
    let mut map = HashMap::with_capacity(labels.len());
    for (index, label) in labels.iter().enumerate() {
        let leg = LegId::try_from(index)
            .map_err(|_| "network_leg_labels contains more than u32::MAX entries".to_owned())?;
        if map.insert(label.as_str(), leg).is_some() {
            return Err("network_leg_labels must be unique".to_owned());
        }
    }
    Ok(map)
}

fn ensure_same_network(current: &TensorNetwork, embedded: &TensorNetwork) -> Result<(), String> {
    if current.inputs != embedded.inputs
        || current.output != embedded.output
        || current.size_dict != embedded.size_dict
    {
        return Err(
            "execution plan embedded network does not match the current network".to_owned(),
        );
    }
    if PathCache::network_canon(current) != PathCache::network_canon(embedded) {
        return Err("execution plan network canonical identity does not match".to_owned());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn triangle() -> (TensorNetwork, Vec<String>, SsaPath) {
        (
            TensorNetwork {
                name: "triangle".to_owned(),
                inputs: vec![vec![0, 1], vec![0, 2], vec![1, 2]],
                output: vec![],
                size_dict: HashMap::from([(0, 2), (1, 2), (2, 2)]),
            },
            vec!["a".to_owned(), "b".to_owned(), "c".to_owned()],
            vec![(0, 1), (2, 3)],
        )
    }

    fn v2_record(net: &TensorNetwork, labels: &[String], path: &SsaPath) -> Value {
        complete_execution_plan_v2(
            net,
            labels,
            serde_json::json!({
                "ssa_path": path,
                "method": "test",
                "sliced": {"legs": []},
            }),
        )
        .unwrap()
    }

    #[test]
    fn version_two_round_trips_embedded_network() {
        let (net, labels, path) = triangle();
        let record = v2_record(&net, &labels, &path);
        assert_eq!(record["schema_version"], EXECUTION_PLAN_VERSION);
        assert_eq!(record["path_format"], EXECUTION_PLAN_PATH_FORMAT);
        assert_eq!(
            record["network"]["size_dict"],
            serde_json::json!([[0, 2], [1, 2], [2, 2]])
        );
        assert_eq!(record["sliced"]["leg_labels"], serde_json::json!([]));
        assert_eq!(record["planning"]["method"], "test");
        assert_eq!(record["provenance"]["source"], "tnpath");
        assert_eq!(
            record["provenance"]["arctn_version"],
            env!("CARGO_PKG_VERSION")
        );

        let loaded = parse_embedded_execution_plan(&record.to_string()).unwrap();
        assert_eq!(loaded.network.inputs, net.inputs);
        assert_eq!(loaded.network.size_dict, net.size_dict);
        assert_eq!(loaded.execution.path, path);
        assert_eq!(loaded.execution.sliced, Some(Vec::new()));
        assert_eq!(loaded.execution.schema_version, Some(2));
    }

    #[test]
    fn writer_adds_canonical_metadata_without_dropping_legacy_fields() {
        let (net, labels, path) = triangle();
        let stats = simulate_path(&net, &path).unwrap();
        let record = complete_execution_plan_v2(
            &net,
            &labels,
            serde_json::json!({
                "ssa_path": path,
                "method": "auto",
                "preset": "heavy",
                "planner_objective": "flops_read_write",
                "flops_weight": 1.0,
                "read_write_weight": 64.0,
                "log10_flops": stats.log10_flops,
                "log2_max_size": stats.log2_max_size,
                "sliced": {
                    "legs": [],
                    "log2_n_slices": 0.0,
                    "log10_flops_total": stats.log10_flops,
                    "per_slice_log2_max_size": stats.log2_max_size,
                    "per_slice_log2_max_contraction_size": stats.log2_max_contraction_size,
                    "per_slice_log2_peak_size": stats.log2_peak_size
                },
                "multi_start": {"base_seed": 42},
            }),
        )
        .unwrap();

        assert_eq!(record["log10_flops"], stats.log10_flops);
        assert_eq!(record["metrics"]["log10_flops"], stats.log10_flops);
        assert_eq!(
            record["metrics"]["sliced_log2_max_size"],
            stats.log2_max_size
        );
        assert_eq!(record["planning"]["preset"], "heavy");
        assert_eq!(record["planning"]["seed"], 42);
        assert_eq!(record["provenance"]["source"], "tnpath");
        assert!(record.get("multi_start").is_some());
    }

    #[test]
    fn version_two_rejects_tampered_path_metrics() {
        let (net, labels, path) = triangle();
        let mut record = v2_record(&net, &labels, &path);
        record["metrics"]["log10_flops"] = serde_json::json!(-999.0);

        let error = parse_embedded_execution_plan(&record.to_string()).unwrap_err();
        assert!(error.contains("metrics.log10_flops"));
        assert!(error.contains("recomputed value"));
    }

    #[test]
    fn version_two_validates_slices_network_and_path_format() {
        let (net, labels, path) = triangle();
        let mut sliced = v2_record(&net, &labels, &path);
        sliced["sliced"]["legs"] = serde_json::json!([0]);
        sliced["sliced"]["leg_labels"] = serde_json::json!(["a"]);
        parse_execution_plan_for_network(&sliced.to_string(), &net, &labels, false).unwrap();

        let mut bad_label = sliced.clone();
        bad_label["sliced"]["leg_labels"] = serde_json::json!(["b"]);
        assert!(parse_embedded_execution_plan(&bad_label.to_string()).is_err());

        let mut bad_format = sliced.clone();
        bad_format["path_format"] = serde_json::json!("linear");
        assert!(parse_embedded_execution_plan(&bad_format.to_string()).is_err());

        let mut bad_network = sliced.clone();
        bad_network["network"]["size_dict"][0][1] = serde_json::json!(3);
        assert!(parse_embedded_execution_plan(&bad_network.to_string()).is_err());

        let sparse_network = ExecutionPlanNetwork {
            inputs: vec![vec![4]],
            output: vec![4],
            size_dict: vec![(4, 2)],
        };
        assert!(sparse_network.to_network("sparse").is_err());
    }

    #[test]
    fn version_one_and_explicit_legacy_remain_readable() {
        let (net, labels, path) = triangle();
        let v1 = serde_json::json!({
            "schema": EXECUTION_PLAN_SCHEMA,
            "schema_version": EXECUTION_PLAN_V1_VERSION,
            "network_canon": PathCache::network_canon(&net),
            "network_leg_labels": labels,
            "net": net.name,
            "ssa_path": path,
            "sliced": {"legs": [0], "leg_labels": ["a"]},
        });
        let loaded =
            parse_execution_plan_for_network(&v1.to_string(), &net, &labels, false).unwrap();
        assert_eq!(loaded.sliced, Some(vec![0]));
        assert_eq!(loaded.schema_version, Some(1));

        let legacy = serde_json::json!({"ssa_path": path, "sliced": {"legs": [1]}});
        assert!(
            parse_execution_plan_for_network(&legacy.to_string(), &net, &labels, false).is_err()
        );
        let loaded =
            parse_execution_plan_for_network(&legacy.to_string(), &net, &labels, true).unwrap();
        assert!(loaded.legacy_unverified);
        assert_eq!(loaded.sliced, Some(vec![1]));
    }

    #[test]
    fn exact_target_is_checked_against_saved_slice_set() {
        let (net, labels, path) = triangle();
        let mut record = v2_record(&net, &labels, &path);

        let mut log2_only = record.clone();
        log2_only["memory_target_log2_elements"] = serde_json::json!(1.0);
        log2_only["memory_constraint_metric"] = serde_json::json!(MEMORY_CONSTRAINT_METRIC);
        assert!(parse_embedded_execution_plan(&log2_only.to_string())
            .unwrap_err()
            .contains("exact target_size"));

        let mut orphan_metric = record.clone();
        orphan_metric["memory_constraint_metric"] = serde_json::json!(MEMORY_CONSTRAINT_METRIC);
        assert!(parse_embedded_execution_plan(&orphan_metric.to_string()).is_err());

        let mut target_without_log2 = record.clone();
        target_without_log2["target_size"] = serde_json::json!(8);
        target_without_log2["memory_constraint_metric"] =
            serde_json::json!(MEMORY_CONSTRAINT_METRIC);
        assert!(parse_embedded_execution_plan(&target_without_log2.to_string()).is_err());

        record["target_size"] = serde_json::json!(2);
        record["memory_target_log2_elements"] = serde_json::json!(1.0);
        record["memory_constraint_metric"] = serde_json::json!(MEMORY_CONSTRAINT_METRIC);
        assert!(parse_embedded_execution_plan(&record.to_string()).is_err());

        record["sliced"]["legs"] = serde_json::json!([0, 1]);
        record["sliced"]["leg_labels"] = serde_json::json!(["a", "b"]);
        parse_embedded_execution_plan(&record.to_string()).unwrap();
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn exact_target_above_f64_integer_precision_is_preserved() {
        let (net, labels, path) = triangle();
        let mut record = v2_record(&net, &labels, &path);
        let target = (1usize << 53) + 1;
        record["target_size"] = serde_json::json!(target);
        record["memory_target_log2_elements"] = serde_json::json!((target as f64).log2());
        record["memory_constraint_metric"] = serde_json::json!(MEMORY_CONSTRAINT_METRIC);
        let loaded = parse_embedded_execution_plan(&record.to_string()).unwrap();
        assert_eq!(loaded.execution.declared_target_size, Some(target));
    }
}
