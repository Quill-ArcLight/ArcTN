//! Tensor-network slicing.
//!
//! Slicing fixes selected legs in explicit loops to reduce the peak memory of
//! each contraction. [`find_slices`] selects legs, [`SliceResult`] stores the
//! slicing plan, and [`contract_network_sliced`] contracts and sums the slices.
//!
//! A slicing plan must remain paired with the path that produced it.
//! `crate::paths::budgeted` and `crate::paths::greedy_path_with_slicing` can
//! select slice legs while constructing a path.

use std::collections::{HashMap, HashSet};

use rayon::prelude::*;

use crate::network::{LegId, TensorNetwork};
use crate::objective::PlannerObjective;
use crate::path::{simulate_path_full, PathStats, SsaPath};
use crate::tensor::{DenseTensor, Scalar};

const FEASIBILITY_TOLERANCE_LOG2: f64 = 1e-9;

/// Memory target used by the slicing internals.
///
/// The public `target_size` API uses [`Self::Elements`] and is checked with
/// integer arithmetic. [`Self::LegacyLog2`] allows a floating-point tolerance
/// for the log2 target API.
#[derive(Clone, Copy, Debug)]
pub(crate) enum SliceTarget {
    Elements(std::num::NonZeroUsize),
    LegacyLog2(f64),
}

impl SliceTarget {
    pub(crate) fn from_elements(target_size: usize) -> Option<Self> {
        std::num::NonZeroUsize::new(target_size).map(Self::Elements)
    }

    pub(crate) fn log2_hint(self) -> f64 {
        match self {
            Self::Elements(value) => (value.get() as f64).log2(),
            Self::LegacyLog2(value) => value,
        }
    }

    pub(crate) fn legs_are_feasible(self, net: &TensorNetwork, legs: &[LegId]) -> bool {
        self.dimensions_are_feasible(legs.iter().map(|&leg| net.dim(leg)))
    }

    pub(crate) fn dimensions_are_feasible(
        self,
        dimensions: impl IntoIterator<Item = usize>,
    ) -> bool {
        match self {
            Self::LegacyLog2(target) => {
                let size: f64 = dimensions
                    .into_iter()
                    .map(|dimension| (dimension as f64).log2())
                    .sum();
                size <= target + FEASIBILITY_TOLERANCE_LOG2
            }
            Self::Elements(target) => {
                // Divide the remaining allowance instead of multiplying the
                // dimensions, so an over-target intermediate cannot overflow
                // `usize` before it is rejected.
                let mut remaining = target.get();
                for dim in dimensions {
                    if dim > remaining {
                        return false;
                    }
                    remaining /= dim;
                }
                true
            }
        }
    }

    fn path_is_feasible(
        self,
        net: &TensorNetwork,
        stats: &PathStats,
        step_legs: &[Vec<LegId>],
    ) -> bool {
        match self {
            Self::LegacyLog2(target) => stats.log2_max_size <= target + FEASIBILITY_TOLERANCE_LOG2,
            Self::Elements(_) if net.n_tensors() == 1 => {
                // An empty SSA path still performs unary trace/sum processing.
                // `target_size` constrains its produced output root, not the
                // larger resident input tensor.
                self.legs_are_feasible(net, &net.output)
            }
            Self::Elements(_) => step_legs
                .iter()
                .all(|legs| self.legs_are_feasible(net, legs)),
        }
    }
}

/// Return whether every produced intermediate is at most `target_size`
/// elements. For a single-tensor network this checks the unary-processed
/// output root rather than vacuously accepting the empty SSA path.
pub fn path_fits_target_size(
    net: &TensorNetwork,
    path: &SsaPath,
    target_size: usize,
) -> Result<bool, String> {
    let target = SliceTarget::from_elements(target_size)
        .ok_or_else(|| "target_size 必须大于 0".to_owned())?;
    let (stats, step_legs) = simulate_path_full(net, path)?;
    Ok(target.path_is_feasible(net, &stats, &step_legs))
}

/// Exact feasibility audit for a path paired with a slicing result.
pub fn slice_result_fits_target_size(
    net: &TensorNetwork,
    path: &SsaPath,
    result: &SliceResult,
    target_size: usize,
) -> Result<bool, String> {
    validate_slice_legs(net, &result.legs)?;
    let mut sliced_net = net.clone();
    for &leg in &result.legs {
        let dim = sliced_net
            .size_dict
            .get_mut(&leg)
            .expect("validate_slice_legs checked size_dict");
        *dim = 1;
    }
    path_fits_target_size(&sliced_net, path, target_size)
}

pub(crate) fn slice_leg_is_executable(net: &TensorNetwork, leg: LegId) -> bool {
    net.size_dict.get(&leg).is_some_and(|&dim| dim > 0)
        && !net.output.contains(&leg)
        && net.inputs.iter().any(|legs| legs.contains(&leg))
        && !net
            .inputs
            .iter()
            .any(|legs| legs.iter().filter(|&&candidate| candidate == leg).count() > 1)
}

pub fn validate_slice_legs(net: &TensorNetwork, sliced: &[LegId]) -> Result<(), String> {
    let mut seen = HashSet::with_capacity(sliced.len());
    for &leg in sliced {
        if !seen.insert(leg) {
            return Err(format!("切片腿 {leg} 重复出现"));
        }
        let dim = net
            .size_dict
            .get(&leg)
            .copied()
            .ok_or_else(|| format!("切片腿 {leg} 不在 size_dict 中"))?;
        if dim == 0 {
            return Err(format!("切片腿 {leg} 的维度为 0"));
        }
        if net.output.contains(&leg) {
            return Err(format!("切片腿 {leg} 在 output 中"));
        }
        if !net.inputs.iter().any(|legs| legs.contains(&leg)) {
            return Err(format!("切片腿 {leg} 未被任何输入张量持有"));
        }
        if net
            .inputs
            .iter()
            .any(|legs| legs.iter().filter(|&&candidate| candidate == leg).count() > 1)
        {
            return Err(format!("切片腿 {leg} 在同一输入张量中重复出现，暂不支持"));
        }
    }
    Ok(())
}

#[derive(Clone, Debug)]
pub struct SliceResult {
    /// Sliced legs.
    pub legs: Vec<LegId>,
    /// Base-two logarithm of the slice count.
    pub log2_n_slices: f64,
    /// Cost of one slice.
    pub per_slice: PathStats,
    /// Base-ten logarithm of total FLOPs across all slices.
    pub log10_flops_total: f64,
}

/// Clone a network with selected leg dimensions set to one.
fn with_dims_one(net: &TensorNetwork, legs: &[LegId]) -> TensorNetwork {
    let mut n = net.clone();
    for l in legs {
        n.size_dict.insert(*l, 1);
    }
    n
}

/// Select slice legs greedily until each intermediate meets the size target.
///
/// Returns `None` when no executable leg can make further progress.
pub fn find_slices(
    net: &TensorNetwork,
    path: &SsaPath,
    target_log2_size: f64,
) -> Option<SliceResult> {
    find_slices_for_target(net, path, SliceTarget::LegacyLog2(target_log2_size))
}

/// Exact element-count variant of [`find_slices`]. A zero target is invalid
/// and returns `None`; all positive targets are checked without a log2
/// tolerance.
pub fn find_slices_to_size(
    net: &TensorNetwork,
    path: &SsaPath,
    target_size: usize,
) -> Option<SliceResult> {
    find_slices_for_target(net, path, SliceTarget::from_elements(target_size)?)
}

pub(crate) fn find_slices_for_target(
    net: &TensorNetwork,
    path: &SsaPath,
    target: SliceTarget,
) -> Option<SliceResult> {
    let mut sliced: Vec<LegId> = Vec::new();
    loop {
        let cur = with_dims_one(net, &sliced);
        let (stats, step_legs) = simulate_path_full(&cur, path).ok()?;
        if target.path_is_feasible(&cur, &stats, &step_legs) {
            let log2_n: f64 = sliced.iter().map(|&l| net.log2_dim(l)).sum();
            let log10_total = stats.log10_flops + log2_n * std::f64::consts::LOG10_2;
            return Some(SliceResult {
                legs: sliced,
                log2_n_slices: log2_n,
                per_slice: stats,
                log10_flops_total: log10_total,
            });
        }
        // Use the same over-target-intermediate score as `slice_temper`.
        let best = pick_slice_leg(net, &cur, &step_legs, target)?;
        sliced.push(best);
        sliced.sort_unstable();
    }
}

/// Deadline-aware [`find_slices`].
///
/// Returns `None` at the deadline because a partial slicing plan has no
/// authoritative score. A `None` deadline uses the unrestricted path.
pub fn find_slices_until(
    net: &TensorNetwork,
    path: &SsaPath,
    target_log2_size: f64,
    deadline: Option<std::time::Instant>,
) -> Option<SliceResult> {
    find_slices_until_for_target(
        net,
        path,
        SliceTarget::LegacyLog2(target_log2_size),
        deadline,
    )
}

pub(crate) fn find_slices_until_for_target(
    net: &TensorNetwork,
    path: &SsaPath,
    target: SliceTarget,
    deadline: Option<std::time::Instant>,
) -> Option<SliceResult> {
    if deadline.is_none() {
        return find_slices_for_target(net, path, target);
    }
    let mut sliced: Vec<LegId> = Vec::new();
    loop {
        if deadline_reached(deadline) {
            return None;
        }
        let cur = with_dims_one(net, &sliced);
        let (stats, step_legs) = simulate_path_full(&cur, path).ok()?;
        if target.path_is_feasible(&cur, &stats, &step_legs) {
            let log2_n: f64 = sliced.iter().map(|&l| net.log2_dim(l)).sum();
            let log10_total = stats.log10_flops + log2_n * std::f64::consts::LOG10_2;
            return Some(SliceResult {
                legs: sliced,
                log2_n_slices: log2_n,
                per_slice: stats,
                log10_flops_total: log10_total,
            });
        }
        let best = pick_slice_leg(net, &cur, &step_legs, target)?;
        sliced.push(best);
        sliced.sort_unstable();
    }
}

/// Alternate slicing and subtree reconfiguration, retaining the best round.
///
/// Each round selects slice legs, sets their dimensions to one, and
/// reconfigures the path to reduce per-slice cost.
pub fn slice_and_reconf(
    net: &TensorNetwork,
    path: &SsaPath,
    target_log2_size: f64,
    rounds: usize,
    subtree_size: usize,
) -> Option<(SsaPath, SliceResult)> {
    slice_and_reconf_with_objective(
        net,
        path,
        target_log2_size,
        rounds,
        subtree_size,
        PlannerObjective::FIXED,
    )
}

/// [`slice_and_reconf`] with a per-call planner objective.
pub fn slice_and_reconf_with_objective(
    net: &TensorNetwork,
    path: &SsaPath,
    target_log2_size: f64,
    rounds: usize,
    subtree_size: usize,
    objective: PlannerObjective,
) -> Option<(SsaPath, SliceResult)> {
    slice_and_reconf_until_for_target_with_objective(
        net,
        path,
        SliceTarget::LegacyLog2(target_log2_size),
        rounds,
        subtree_size,
        None,
        objective,
    )
}

/// Exact element-count variant of [`slice_and_reconf`].
pub fn slice_and_reconf_to_size(
    net: &TensorNetwork,
    path: &SsaPath,
    target_size: usize,
    rounds: usize,
    subtree_size: usize,
) -> Option<(SsaPath, SliceResult)> {
    slice_and_reconf_to_size_with_objective(
        net,
        path,
        target_size,
        rounds,
        subtree_size,
        PlannerObjective::FIXED,
    )
}

/// [`slice_and_reconf_to_size`] with a per-call planner objective.
pub fn slice_and_reconf_to_size_with_objective(
    net: &TensorNetwork,
    path: &SsaPath,
    target_size: usize,
    rounds: usize,
    subtree_size: usize,
    objective: PlannerObjective,
) -> Option<(SsaPath, SliceResult)> {
    slice_and_reconf_until_for_target_with_objective(
        net,
        path,
        SliceTarget::from_elements(target_size)?,
        rounds,
        subtree_size,
        None,
        objective,
    )
}

/// Deadline-aware [`slice_and_reconf`].
///
/// Returns the best complete snapshot at the deadline. A `None` deadline uses
/// the unrestricted path.
pub fn slice_and_reconf_until(
    net: &TensorNetwork,
    path: &SsaPath,
    target_log2_size: f64,
    rounds: usize,
    subtree_size: usize,
    deadline: Option<std::time::Instant>,
) -> Option<(SsaPath, SliceResult)> {
    slice_and_reconf_until_with_objective(
        net,
        path,
        target_log2_size,
        rounds,
        subtree_size,
        deadline,
        PlannerObjective::FIXED,
    )
}

/// [`slice_and_reconf_until`] with a per-call planner objective.
#[allow(clippy::too_many_arguments)]
pub fn slice_and_reconf_until_with_objective(
    net: &TensorNetwork,
    path: &SsaPath,
    target_log2_size: f64,
    rounds: usize,
    subtree_size: usize,
    deadline: Option<std::time::Instant>,
    objective: PlannerObjective,
) -> Option<(SsaPath, SliceResult)> {
    slice_and_reconf_until_with_finder(
        net,
        path,
        SliceTarget::LegacyLog2(target_log2_size),
        rounds,
        subtree_size,
        deadline,
        find_slices_until_for_target,
        objective,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn slice_and_reconf_until_for_target_with_objective(
    net: &TensorNetwork,
    path: &SsaPath,
    target: SliceTarget,
    rounds: usize,
    subtree_size: usize,
    deadline: Option<std::time::Instant>,
    objective: PlannerObjective,
) -> Option<(SsaPath, SliceResult)> {
    slice_and_reconf_until_with_finder(
        net,
        path,
        target,
        rounds,
        subtree_size,
        deadline,
        find_slices_until_for_target,
        objective,
    )
}

#[allow(clippy::too_many_arguments)]
fn slice_and_reconf_until_with_finder<F>(
    net: &TensorNetwork,
    path: &SsaPath,
    target: SliceTarget,
    rounds: usize,
    subtree_size: usize,
    deadline: Option<std::time::Instant>,
    mut finder: F,
    objective: PlannerObjective,
) -> Option<(SsaPath, SliceResult)>
where
    F: FnMut(
        &TensorNetwork,
        &SsaPath,
        SliceTarget,
        Option<std::time::Instant>,
    ) -> Option<SliceResult>,
{
    if deadline_reached(deadline) {
        return None;
    }
    let mut path = path.clone();
    let mut best: Option<(SsaPath, SliceResult)> = None;
    for _ in 0..rounds.max(1) {
        if deadline_reached(deadline) {
            return best;
        }
        let Some(sr) = finder(net, &path, target, deadline) else {
            return best;
        };
        let better = best
            .as_ref()
            .map(|(_, incumbent)| {
                objective.score_sliced_log2(&sr.per_slice, sr.log2_n_slices)
                    < objective.score_sliced_log2(&incumbent.per_slice, incumbent.log2_n_slices)
            })
            .unwrap_or(true);
        if better {
            best = Some((path.clone(), sr.clone()));
        }
        if deadline_reached(deadline) {
            return best;
        }
        let sliced_net = with_dims_one(net, &sr.legs);
        let Ok((candidate, stats)) = crate::tree::reconfigure_path_until_with_objective(
            &sliced_net,
            &path,
            subtree_size,
            20,
            deadline,
            objective,
        ) else {
            return best;
        };
        let current = simulate_path_full(&sliced_net, &path).ok()?.0;
        if objective.score_path_log2(&stats) <= objective.score_path_log2(&current) {
            path = candidate;
        }
    }
    if !deadline_reached(deadline) {
        if let Some(sr) = finder(net, &path, target, deadline) {
            let better = best
                .as_ref()
                .map(|(_, incumbent)| {
                    objective.score_sliced_log2(&sr.per_slice, sr.log2_n_slices)
                        < objective.score_sliced_log2(&incumbent.per_slice, incumbent.log2_n_slices)
                })
                .unwrap_or(true);
            if better {
                best = Some((path, sr));
            }
        }
    }
    best
}

/// Pick the largest leg shared by the most over-target intermediates.
///
/// Ties are resolved by the smallest leg ID.
fn pick_slice_leg(
    net: &TensorNetwork,
    cur: &TensorNetwork,
    step_legs: &[Vec<LegId>],
    target: SliceTarget,
) -> Option<LegId> {
    let mut score: HashMap<LegId, (usize, f64)> = HashMap::new();
    for legs in step_legs {
        if target.legs_are_feasible(cur, legs) {
            continue;
        }
        for &l in legs {
            let d = cur.log2_dim(l);
            if d <= 0.0 || !slice_leg_is_executable(net, l) {
                continue;
            }
            let e = score.entry(l).or_insert((0, d));
            e.0 += 1;
        }
    }
    score
        .into_iter()
        .max_by(|a, b| {
            (a.1 .0)
                .cmp(&b.1 .0)
                .then(a.1 .1.total_cmp(&b.1 .1))
                .then(b.0.cmp(&a.0))
        })
        .map(|(l, _)| l)
}

#[inline]
fn deadline_reached(deadline: Option<std::time::Instant>) -> bool {
    deadline
        .map(|d| std::time::Instant::now() >= d)
        .unwrap_or(false)
}

/// Add slice legs incrementally and anneal the path after each addition.
///
/// Retains the best feasible snapshot and stops after two rounds without an
/// improvement or after reaching the slice limit.
#[allow(clippy::too_many_arguments)]
pub fn slice_temper(
    net: &TensorNetwork,
    path: &SsaPath,
    target_log2_size: f64,
    seed: u64,
    chains: usize,
    rounds_per_leg: usize,
    moves: usize,
    reconf_size: usize,
) -> Option<(SsaPath, SliceResult)> {
    slice_temper_with_objective(
        net,
        path,
        target_log2_size,
        seed,
        chains,
        rounds_per_leg,
        moves,
        reconf_size,
        PlannerObjective::FIXED,
    )
}

/// [`slice_temper`] with a per-call planner objective.
#[allow(clippy::too_many_arguments)]
pub fn slice_temper_with_objective(
    net: &TensorNetwork,
    path: &SsaPath,
    target_log2_size: f64,
    seed: u64,
    chains: usize,
    rounds_per_leg: usize,
    moves: usize,
    reconf_size: usize,
    objective: PlannerObjective,
) -> Option<(SsaPath, SliceResult)> {
    slice_temper_until_with_objective(
        net,
        path,
        target_log2_size,
        seed,
        chains,
        rounds_per_leg,
        moves,
        reconf_size,
        None,
        objective,
    )
}

/// Exact element-count variant of [`slice_temper`].
#[allow(clippy::too_many_arguments)]
pub fn slice_temper_to_size(
    net: &TensorNetwork,
    path: &SsaPath,
    target_size: usize,
    seed: u64,
    chains: usize,
    rounds_per_leg: usize,
    moves: usize,
    reconf_size: usize,
) -> Option<(SsaPath, SliceResult)> {
    slice_temper_to_size_with_objective(
        net,
        path,
        target_size,
        seed,
        chains,
        rounds_per_leg,
        moves,
        reconf_size,
        PlannerObjective::FIXED,
    )
}

/// [`slice_temper_to_size`] with a per-call planner objective.
#[allow(clippy::too_many_arguments)]
pub fn slice_temper_to_size_with_objective(
    net: &TensorNetwork,
    path: &SsaPath,
    target_size: usize,
    seed: u64,
    chains: usize,
    rounds_per_leg: usize,
    moves: usize,
    reconf_size: usize,
    objective: PlannerObjective,
) -> Option<(SsaPath, SliceResult)> {
    slice_temper_until_for_target_with_objective(
        net,
        path,
        SliceTarget::from_elements(target_size)?,
        seed,
        chains,
        rounds_per_leg,
        moves,
        reconf_size,
        None,
        objective,
    )
}

/// Deadline-aware [`slice_temper`] that returns the best feasible snapshot.
#[allow(clippy::too_many_arguments)]
pub fn slice_temper_until(
    net: &TensorNetwork,
    path: &SsaPath,
    target_log2_size: f64,
    seed: u64,
    chains: usize,
    rounds_per_leg: usize,
    moves: usize,
    reconf_size: usize,
    deadline: Option<std::time::Instant>,
) -> Option<(SsaPath, SliceResult)> {
    slice_temper_until_with_objective(
        net,
        path,
        target_log2_size,
        seed,
        chains,
        rounds_per_leg,
        moves,
        reconf_size,
        deadline,
        PlannerObjective::FIXED,
    )
}

/// [`slice_temper_until`] with a per-call planner objective.
#[allow(clippy::too_many_arguments)]
pub fn slice_temper_until_with_objective(
    net: &TensorNetwork,
    path: &SsaPath,
    target_log2_size: f64,
    seed: u64,
    chains: usize,
    rounds_per_leg: usize,
    moves: usize,
    reconf_size: usize,
    deadline: Option<std::time::Instant>,
    objective: PlannerObjective,
) -> Option<(SsaPath, SliceResult)> {
    slice_temper_until_for_target_with_objective(
        net,
        path,
        SliceTarget::LegacyLog2(target_log2_size),
        seed,
        chains,
        rounds_per_leg,
        moves,
        reconf_size,
        deadline,
        objective,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn slice_temper_until_for_target_with_objective(
    net: &TensorNetwork,
    path: &SsaPath,
    target: SliceTarget,
    seed: u64,
    chains: usize,
    rounds_per_leg: usize,
    moves: usize,
    reconf_size: usize,
    deadline: Option<std::time::Instant>,
    objective: PlannerObjective,
) -> Option<(SsaPath, SliceResult)> {
    const MAX_SLICED: usize = 64;
    const DRY_STOP: usize = 2;
    let mut sliced: Vec<LegId> = Vec::new();
    let mut path = path.clone();
    let mut best: Option<(SsaPath, SliceResult)> = None;
    let mut dry = 0usize;
    for step in 0.. {
        if deadline_reached(deadline) {
            return best;
        }
        let cur = with_dims_one(net, &sliced);
        let Ok((stats, step_legs)) = simulate_path_full(&cur, &path) else {
            return best;
        };
        if target.path_is_feasible(&cur, &stats, &step_legs) {
            let log2_n: f64 = sliced.iter().map(|&l| net.log2_dim(l)).sum();
            let sr = SliceResult {
                legs: sliced.clone(),
                log2_n_slices: log2_n,
                per_slice: stats,
                log10_flops_total: stats.log10_flops + log2_n * std::f64::consts::LOG10_2,
            };
            let improved = best
                .as_ref()
                .map(|(_, incumbent)| {
                    objective.score_sliced_log2(&sr.per_slice, sr.log2_n_slices)
                        < objective.score_sliced_log2(&incumbent.per_slice, incumbent.log2_n_slices)
                            - 1e-12
                })
                .unwrap_or(true);
            if improved {
                best = Some((path.clone(), sr));
                dry = 0;
            } else {
                dry += 1;
                if dry >= DRY_STOP {
                    return best;
                }
            }
        }
        if sliced.len() >= MAX_SLICED {
            return best;
        }
        let scoring_target = SliceTarget::LegacyLog2(
            target
                .log2_hint()
                .min(stats.log2_max_size - 2.0 * FEASIBILITY_TOLERANCE_LOG2),
        );
        let Some(leg) = pick_slice_leg(net, &cur, &step_legs, scoring_target) else {
            return best;
        };
        sliced.push(leg);
        sliced.sort_unstable();
        let sliced_net = with_dims_one(net, &sliced);
        match crate::tree::temper_paths_until_with_objective(
            &sliced_net,
            std::slice::from_ref(&path),
            chains,
            rounds_per_leg,
            moves,
            1e-4,
            0.15,
            1_000,
            reconf_size,
            seed.wrapping_add(step as u64 * 131),
            0,
            deadline,
            objective,
        ) {
            Ok((candidate, candidate_stats)) => {
                let current_stats = simulate_path_full(&sliced_net, &path).ok()?.0;
                if objective.score_path_log2(&candidate_stats)
                    <= objective.score_path_log2(&current_stats)
                {
                    path = candidate;
                }
            }
            Err(_) => {
                if deadline_reached(deadline) {
                    return best;
                }
                if let Ok((candidate, candidate_stats)) =
                    crate::tree::reconfigure_path_until_with_objective(
                        &sliced_net,
                        &path,
                        reconf_size,
                        20,
                        deadline,
                        objective,
                    )
                {
                    let current_stats = simulate_path_full(&sliced_net, &path).ok()?.0;
                    if objective.score_path_log2(&candidate_stats)
                        <= objective.score_path_log2(&current_stats)
                    {
                        path = candidate;
                    }
                }
            }
        }
    }
    unreachable!()
}

/// Decode a slice number into mixed-radix leg values.
///
/// The final leg changes fastest.
#[inline]
fn decode_slice_index(mut s: usize, dims: &[usize], n_slices: usize, idx: &mut [usize]) {
    // Reject out-of-range indices before modulo arithmetic can wrap them.
    assert!(s < n_slices, "切片编号 {s} 越界（n_slices={n_slices}）");
    for d in (0..dims.len()).rev() {
        idx[d] = s % dims[d];
        s /= dims[d];
    }
}

/// Return the number of execution chunks and the outer parallelism limit.
///
/// The result depends only on `(n_slices, out_numel)`, not the thread count.
pub fn slice_parallel_chunks(n_slices: usize, out_numel: usize) -> usize {
    if n_slices <= 1 {
        return 1;
    }
    let budget_chunks = (SLICE_PARTIAL_BUDGET / out_numel.max(1)).max(1);
    let n0 = n_slices.min(SLICE_MAX_CHUNKS).min(budget_chunks).max(1);
    if n0 <= 1 {
        return 1;
    }
    // Recompute the count from the chunk length to keep the tail non-empty.
    let chunk = n_slices.div_ceil(n0);
    n_slices.div_ceil(chunk)
}

/// Number of output elements, which is identical for every slice.
fn output_numel(net: &TensorNetwork) -> usize {
    net.output
        .iter()
        .try_fold(1usize, |acc, &l| acc.checked_mul(net.dim(l)))
        .unwrap_or(usize::MAX) // Overflow safely forces the chunk count to one.
        .max(1)
}

/// Maximum slice chunk count.
///
/// This must remain independent of the thread count because chunking fixes the
/// summation order and therefore floating-point rounding. The fixed limit also
/// leaves enough tasks for Rayon work stealing.
const SLICE_MAX_CHUNKS: usize = 256;

/// Element budget for parallel chunk partial sums.
const SLICE_PARTIAL_BUDGET: usize = 1 << 22;

/// Validate inputs for sliced contraction.
///
/// In addition to the ordinary tensor-count and shape checks, every sliced leg
/// must be:
///
/// - unique;
/// - present in `size_dict` with nonzero dimension;
/// - present in at least one input tensor;
/// - absent from the output;
/// - present at most once in each input tensor.
///
/// The final constraint excludes trace semantics that axis selection cannot
/// represent. This public validator lets ranged `tnmpi` execution enforce the
/// same contract as the single-process entry point.
pub fn validate_sliced_contraction_inputs<T: Scalar>(
    net: &TensorNetwork,
    tensors: &[DenseTensor<T>],
    sliced: &[LegId],
) -> Result<(), String> {
    crate::contract::validate_shapes(net, tensors)?;
    validate_slice_legs(net, sliced)
}

/// Contract every assignment of the sliced legs and sum the results.
///
/// Slice numbers are partitioned into contiguous chunks. Rayon processes the
/// chunks in parallel, each chunk accumulates in slice-number order, and the
/// partial sums are merged in chunk order. Thread-count-independent chunking
/// preserves the summation order across parallelism levels.
///
/// Parallel execution increases resident inputs, intermediates, and partial
/// sums. `target_size` constrains one slice, not these concurrent replicas.
pub fn contract_network_sliced<T: Scalar>(
    net: &TensorNetwork,
    tensors: &[DenseTensor<T>],
    path: &SsaPath,
    sliced: &[LegId],
) -> Result<DenseTensor<T>, String> {
    validate_sliced_contraction_inputs(net, tensors, sliced)?;
    if sliced.is_empty() {
        return crate::contract::contract_network(net, tensors.to_vec(), path);
    }

    let sliced_pos: HashMap<LegId, usize> = sliced
        .iter()
        .copied()
        .enumerate()
        .map(|(index, leg)| (leg, index))
        .collect();
    // Remove sliced legs from every input in the per-slice network.
    let mut sub = net.clone();
    for t in &mut sub.inputs {
        t.retain(|l| !sliced_pos.contains_key(l));
    }
    for &l in sliced {
        sub.size_dict.remove(&l);
    }

    let dims: Vec<usize> = sliced.iter().map(|&l| net.dim(l)).collect();
    let n_slices = dims.iter().try_fold(1usize, |total, &dim| {
        total
            .checked_mul(dim)
            .ok_or_else(|| "切片总数溢出 usize".to_string())
    })?;

    // Decode one slice, select its input axes, and reuse the same path.
    let run_one = |s: usize| -> Result<DenseTensor<T>, String> {
        let mut idx = vec![0usize; sliced.len()];
        decode_slice_index(s, &dims, n_slices, &mut idx);
        let sliced_tensors: Vec<DenseTensor<T>> = net
            .inputs
            .iter()
            .zip(tensors)
            .map(|(legs, t)| {
                let mut t = t.clone();
                // Remove axes in reverse order so positions remain valid.
                for ax in (0..legs.len()).rev() {
                    if let Some(&k) = sliced_pos.get(&legs[ax]) {
                        t = t.select_axis(ax, idx[k]);
                    }
                }
                t
            })
            .collect();
        crate::contract::contract_network(&sub, sliced_tensors, path)
    };

    let n_chunks = slice_parallel_chunks(n_slices, output_numel(net));

    // Accumulate each chunk in slice-number order.
    if n_chunks == 1 {
        let mut acc = run_one(0)?;
        for s in 1..n_slices {
            let part = run_one(s)?;
            for (a, b) in acc.data.iter_mut().zip(part.data) {
                *a += b;
            }
        }
        return Ok(acc);
    }

    // Chunk `c` owns the contiguous range `[lo, hi)`.
    let chunk = n_slices.div_ceil(n_chunks);
    let partials: Vec<Result<DenseTensor<T>, String>> = (0..n_chunks)
        .into_par_iter()
        .map(|c| {
            let lo = c * chunk;
            let hi = ((c + 1) * chunk).min(n_slices);
            // Reject empty chunks in every build mode.
            assert!(
                lo < hi,
                "块 {c} 为空：n_slices={n_slices} n_chunks={n_chunks} chunk={chunk}"
            );
            let mut acc = run_one(lo)?;
            for s in (lo + 1)..hi {
                let part = run_one(s)?;
                for (a, b) in acc.data.iter_mut().zip(part.data) {
                    *a += b;
                }
            }
            Ok(acc)
        })
        .collect(); // Indexed parallel collection preserves chunk order.

    // Merge serially in chunk order for thread-count-independent rounding.
    let mut iter = partials.into_iter();
    let mut acc = iter.next().expect("n_chunks >= 1")?;
    for p in iter {
        let part = p?;
        for (a, b) in acc.data.iter_mut().zip(part.data) {
            *a += b;
        }
    }
    Ok(acc)
}

#[cfg(test)]
mod deadline_incumbent_tests {
    use std::cell::Cell;
    use std::time::{Duration, Instant};

    use super::{
        find_slices, find_slices_to_size, find_slices_until, path_fits_target_size,
        slice_and_reconf_until_with_finder, slice_result_fits_target_size, slice_temper,
        SliceTarget, FEASIBILITY_TOLERANCE_LOG2,
    };
    use crate::network::TensorNetwork;

    fn fixture() -> (TensorNetwork, Vec<(usize, usize)>, super::SliceResult) {
        let net = TensorNetwork {
            name: "deadline-incumbent".into(),
            inputs: vec![vec![0, 1], vec![1, 2]],
            output: vec![0, 2],
            size_dict: [(0, 2), (1, 2), (2, 2)].into_iter().collect(),
        };
        let path = vec![(0, 1)];
        let slice = find_slices(&net, &path, 2.0).expect("fixture must have a complete slice");
        (net, path, slice)
    }

    fn assert_followup_none_keeps_incumbent() {
        let (net, path, first) = fixture();
        let expected_total = first.log10_flops_total.to_bits();
        let calls = Cell::new(0usize);
        let result = slice_and_reconf_until_with_finder(
            &net,
            &path,
            SliceTarget::LegacyLog2(2.0),
            2,
            2,
            Some(Instant::now() + Duration::from_secs(5)),
            |_, _, _, _| {
                let call = calls.get();
                calls.set(call + 1);
                match call {
                    0 => Some(first.clone()),
                    1 => None,
                    _ => panic!("search must return immediately after the incomplete follow-up"),
                }
            },
            crate::PlannerObjective::FIXED,
        )
        .expect("the first completed incumbent must survive the follow-up None");

        assert_eq!(calls.get(), 2, "fixture must exercise the follow-up finder");
        assert_eq!(result.0, path);
        assert_eq!(result.1.log10_flops_total.to_bits(), expected_total);
    }

    #[test]
    fn fixed_objective_keeps_incumbent_after_followup_deadline_none() {
        assert_followup_none_keeps_incumbent();
    }

    #[test]
    fn planner_does_not_emit_a_slice_that_execution_rejects() {
        let net = TensorNetwork {
            name: "repeated-leg-slice".into(),
            inputs: vec![vec![0, 0, 1], vec![0, 2], vec![0, 3]],
            output: vec![1, 2, 3],
            size_dict: [(0, 8), (1, 2), (2, 2), (3, 2)].into_iter().collect(),
        };
        let path = vec![(0, 1), (3, 2)];
        let stats = crate::simulate_path(&net, &path).unwrap();
        let unsupported = super::SliceResult {
            legs: vec![0],
            log2_n_slices: 3.0,
            per_slice: stats,
            log10_flops_total: stats.log10_flops + 3.0 * std::f64::consts::LOG10_2,
        };

        assert!(find_slices_to_size(&net, &path, 16).is_none());
        assert!(slice_result_fits_target_size(&net, &path, &unsupported, 16).is_err());
    }

    #[test]
    fn exact_product_target_tolerates_log_sum_rounding_for_uncuttable_output() {
        // Mathematically the only contraction result has exactly 499 * 465 elements.
        // On f64, however, log2(499) + log2(465) rounds a few ulps above
        // log2(499 * 465). Both result legs are outputs, so a false over-target
        // classification cannot be repaired by slicing and used to return None.
        let net = TensorNetwork {
            name: "exact-product-boundary".into(),
            inputs: vec![vec![0, 2], vec![2, 1]],
            output: vec![0, 1],
            size_dict: [(0, 499), (1, 465), (2, 2)].into_iter().collect(),
        };
        let path = vec![(0, 1)];
        let target = ((499usize * 465) as f64).log2();
        let stats = crate::simulate_path(&net, &path).expect("fixture path must be valid");
        assert!(stats.log2_max_size > target);
        assert!(stats.log2_max_size <= target + FEASIBILITY_TOLERANCE_LOG2);

        let greedy = find_slices(&net, &path, target).expect("exact target must be feasible");
        assert!(greedy.legs.is_empty());
        let exact = find_slices_to_size(&net, &path, 499 * 465)
            .expect("integer target must not reject its exact product");
        assert!(exact.legs.is_empty());
        assert!(slice_result_fits_target_size(&net, &path, &exact, 499 * 465).unwrap());
        assert!(find_slices_to_size(&net, &path, 499 * 465 - 1).is_none());

        let deadline = Some(Instant::now() + Duration::from_secs(1));
        let bounded = find_slices_until(&net, &path, target, deadline)
            .expect("deadline-aware finder must use the same boundary");
        assert!(bounded.legs.is_empty());

        let (_, tempered) = slice_temper(&net, &path, target, 0, 1, 1, 1, 2)
            .expect("temper finder must use the same boundary");
        assert!(tempered.legs.is_empty());
    }

    #[test]
    fn canonical_target_never_accepts_the_next_large_integer() {
        // Adjacent large integers differ by less than the log2 tolerance.
        let target_size = 1usize << 40;
        let net = TensorNetwork {
            name: "large-adjacent-integers".into(),
            inputs: vec![vec![0], vec![]],
            output: vec![0],
            size_dict: [(0, target_size + 1)].into_iter().collect(),
        };
        let path = vec![(0, 1)];
        let legacy_target = (target_size as f64).log2();
        assert!(
            find_slices(&net, &path, legacy_target).is_some(),
            "legacy f64 compatibility intentionally retains its tolerance"
        );
        assert!(find_slices_to_size(&net, &path, target_size).is_none());
        assert!(!path_fits_target_size(&net, &path, target_size).unwrap());
        assert!(path_fits_target_size(&net, &path, target_size + 1).unwrap());
    }

    #[test]
    fn single_tensor_exact_target_checks_unary_output_root() {
        let target_size = 1usize << 40;
        let net = TensorNetwork {
            name: "single-unary-root".into(),
            // Leg 1 is summed away by the unary finalization. The input is
            // larger than the output, but target_size constrains the root.
            inputs: vec![vec![0, 1]],
            output: vec![0],
            size_dict: [(0, target_size + 1), (1, 2)].into_iter().collect(),
        };
        let path = Vec::new();
        assert!(!path_fits_target_size(&net, &path, target_size).unwrap());
        assert!(path_fits_target_size(&net, &path, target_size + 1).unwrap());
        assert!(find_slices_to_size(&net, &path, target_size).is_none());
        let result = find_slices_to_size(&net, &path, target_size + 1)
            .expect("an exact unary root should need no sliced legs");
        assert!(result.legs.is_empty());
    }
}
