//! Greedy path construction with an intermediate-size target.
//!
//! Slice choices change candidate costs, so the lazy heap revalidates entries
//! when popped. Search supports restarts, optional seed paths, and deadlines.

use std::collections::{BinaryHeap, HashMap, HashSet};

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::network::{LegId, TensorNetwork};
use crate::objective::PlannerObjective;
use crate::path::{simulate_path, sorted_dedup, PathStats, SsaPath};
use crate::paths::greedy::CostFn;
use crate::slice::SliceTarget;

#[derive(Clone, Debug)]
pub struct BudgetedResult {
    pub path: SsaPath,
    pub sliced: Vec<LegId>,
    /// Metrics for one slice with sliced dimensions set to one.
    pub per_slice: PathStats,
    pub log2_n_slices: f64,
    /// Total FLOPs across all slices.
    pub log10_flops_total: f64,
}

fn objective_score(objective: PlannerObjective, result: &BudgetedResult) -> f64 {
    objective.score_sliced_log2(&result.per_slice, result.log2_n_slices)
}

// Candidate ordering matches the greedy min-heap with `(a, b)` tie-breaking.
#[derive(Clone, Copy, Debug)]
struct Cand {
    cost: f64, // Slice changes can invalidate this cached cost.
    a: usize,
    b: usize,
}
impl PartialEq for Cand {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost && self.a == other.a && self.b == other.b
    }
}
impl Eq for Cand {}
impl Ord for Cand {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other
            .cost
            .total_cmp(&self.cost)
            .then(other.a.cmp(&self.a))
            .then(other.b.cmp(&self.b))
    }
}
impl PartialOrd for Cand {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

struct State<'a> {
    net: &'a TensorNetwork,
    target: SliceTarget, // exact elements or legacy log2 budget
    node_legs: Vec<Option<Vec<LegId>>>,
    refcount: HashMap<LegId, usize>,
    in_output: HashSet<LegId>,
    holders: HashMap<LegId, HashSet<usize>>,
    heap: BinaryHeap<Cand>,
    sliced: HashSet<LegId>,
    costmod: f64,
    cost_fn: CostFn,
}

impl<'a> State<'a> {
    fn new(
        net: &'a TensorNetwork,
        target: SliceTarget,
        costmod: f64,
        cost_fn: CostFn,
        preslice: &[LegId],
    ) -> Self {
        let in_output: HashSet<LegId> = net.output.iter().copied().collect();
        let mut refcount: HashMap<LegId, usize> = HashMap::new();
        let mut holders: HashMap<LegId, HashSet<usize>> = HashMap::new();
        let mut node_legs = Vec::new();
        for (i, t) in net.inputs.iter().enumerate() {
            let d = sorted_dedup(t);
            for &l in &d {
                *refcount.entry(l).or_insert(0) += 1;
                holders.entry(l).or_default().insert(i);
            }
            node_legs.push(Some(d));
        }
        let mut st = State {
            net,
            target,
            node_legs,
            refcount,
            in_output,
            holders,
            heap: BinaryHeap::new(),
            sliced: preslice.iter().copied().collect(),
            costmod,
            cost_fn,
        };
        let mut seen: HashSet<(usize, usize)> = HashSet::new();
        for hs in st.holders.clone().values() {
            let mut v: Vec<usize> = hs.iter().copied().collect();
            v.sort_unstable();
            for i in 0..v.len() {
                for j in (i + 1)..v.len() {
                    if seen.insert((v[i], v[j])) {
                        let cost = st.cand_cost(v[i], v[j]);
                        st.heap.push(Cand {
                            cost,
                            a: v[i],
                            b: v[j],
                        });
                    }
                }
            }
        }
        st
    }

    /// Return a leg's effective log2 dimension; sliced legs have dimension one.
    fn log2dim(&self, l: LegId) -> f64 {
        if self.sliced.contains(&l) {
            0.0
        } else {
            self.net.log2_dim(l)
        }
    }

    fn legs_size(&self, legs: &[LegId]) -> f64 {
        legs.iter().map(|&l| self.log2dim(l)).sum::<f64>().exp2()
    }

    /// Return retained result legs using the canonical contraction rule.
    fn result_legs(&self, a: usize, b: usize) -> Vec<LegId> {
        crate::path::result_legs(
            self.node_legs[a].as_ref().unwrap(),
            self.node_legs[b].as_ref().unwrap(),
            &self.refcount,
            &self.in_output,
        )
    }

    /// Score a candidate with effective sliced dimensions.
    fn cand_cost(&self, a: usize, b: usize) -> f64 {
        let result = self.result_legs(a, b);
        let rsize = self.legs_size(&result);
        let sa = self.legs_size(self.node_legs[a].as_ref().unwrap());
        let sb = self.legs_size(self.node_legs[b].as_ref().unwrap());
        let base = rsize - self.costmod * (sa + sb);
        match self.cost_fn {
            CostFn::MemRemoved => base,
            CostFn::Skew(sk) => base + sk * (sa - sb).abs(),
            CostFn::Ratio => base / (rsize + 1.0),
            CostFn::LogRatio => (rsize + 2.0).log2() / (0.65 * (sa + sb) + 2.0).log2(),
            CostFn::HyperDeg(hd) => {
                let deg: usize = result.iter().map(|l| self.refcount[l]).sum();
                base + hd * deg as f64
            }
        }
    }

    fn alive(&self, i: usize) -> bool {
        self.node_legs.get(i).map(|x| x.is_some()).unwrap_or(false)
    }

    /// Pop candidates whose endpoints and cached costs remain valid.
    /// Reinsert candidates whose costs changed after slicing.
    fn pop_valid(&mut self, want: usize) -> Vec<Cand> {
        let mut out = Vec::new();
        while out.len() < want {
            let Some(c) = self.heap.pop() else { break };
            if !self.alive(c.a) || !self.alive(c.b) {
                continue;
            }
            let now = self.cand_cost(c.a, c.b);
            if (now - c.cost).abs() > 1e-9 * (1.0 + c.cost.abs()) {
                self.heap.push(Cand {
                    cost: now,
                    a: c.a,
                    b: c.b,
                });
                continue;
            }
            if out.iter().any(|o: &Cand| o.a == c.a && o.b == c.b) {
                continue;
            }
            out.push(c);
        }
        out
    }

    /// Slice result legs until the target is met.
    /// Prefer holder count, effective dimension, then leg id.
    fn enforce_budget(&mut self, result: &[LegId], exclude: (usize, usize)) -> Result<(), String> {
        loop {
            // Canonical element targets are checked with integer arithmetic;
            // sliced legs contribute dimension one. Legacy log2 callers keep
            // the historical tolerance through `SliceTarget`.
            if self
                .target
                .dimensions_are_feasible(result.iter().map(|&leg| {
                    if self.sliced.contains(&leg) {
                        1
                    } else {
                        self.net.dim(leg)
                    }
                }))
            {
                return Ok(());
            }
            let pick = result
                .iter()
                .copied()
                .filter(|&leg| {
                    self.log2dim(leg) > 0.0 && crate::slice::slice_leg_is_executable(self.net, leg)
                })
                .max_by(|&x, &y| {
                    self.holders[&x]
                        .len()
                        .cmp(&self.holders[&y].len())
                        .then_with(|| self.log2dim(x).total_cmp(&self.log2dim(y)))
                        .then(y.cmp(&x))
                });
            match pick {
                Some(l) => {
                    self.sliced.insert(l);
                    self.repush_after_slice(l, exclude);
                }
                None => return Err("结果腿全是开放腿，无法切片到预算内".into()),
            }
        }
    }

    /// Push updated costs for live candidates affected by a sliced leg.
    fn repush_after_slice(&mut self, l: LegId, exclude: (usize, usize)) {
        // Sort and deduplicate pairs for deterministic insertion order.
        let mut ps: Vec<usize> = self.holders[&l]
            .iter()
            .copied()
            .filter(|&p| p != exclude.0 && p != exclude.1 && self.alive(p))
            .collect();
        ps.sort_unstable();
        let mut pairs: Vec<(usize, usize)> = Vec::new();
        for &p in &ps {
            for &leg in self.node_legs[p].as_ref().unwrap() {
                for &q in &self.holders[&leg] {
                    if q != p && q != exclude.0 && q != exclude.1 && self.alive(q) {
                        pairs.push((p.min(q), p.max(q)));
                    }
                }
            }
        }
        pairs.sort_unstable();
        pairs.dedup();
        for (x, y) in pairs {
            let cost = self.cand_cost(x, y);
            self.heap.push(Cand { cost, a: x, b: y });
        }
    }

    fn contract(&mut self, a: usize, b: usize) -> Result<usize, String> {
        let result = self.result_legs(a, b);
        // Enforce the target before committing the contraction.
        self.enforce_budget(&result, (a, b))?;
        let la = self.node_legs[a].take().unwrap();
        let lb = self.node_legs[b].take().unwrap();
        for &l in &la {
            *self.refcount.get_mut(&l).unwrap() -= 1;
            self.holders.get_mut(&l).unwrap().remove(&a);
        }
        for &l in &lb {
            *self.refcount.get_mut(&l).unwrap() -= 1;
            self.holders.get_mut(&l).unwrap().remove(&b);
        }
        let new_id = self.node_legs.len();
        for &l in &result {
            *self.refcount.get_mut(&l).unwrap() += 1;
            self.holders.get_mut(&l).unwrap().insert(new_id);
        }
        self.node_legs.push(Some(result));
        let mut nbrs: HashSet<usize> = HashSet::new();
        for l in self.node_legs[new_id].as_ref().unwrap().clone() {
            for &h in &self.holders[&l] {
                if h != new_id {
                    nbrs.insert(h);
                }
            }
        }
        for nb in nbrs {
            let (x, y) = if nb < new_id {
                (nb, new_id)
            } else {
                (new_id, nb)
            };
            let cost = self.cand_cost(x, y);
            self.heap.push(Cand { cost, a: x, b: y });
        }
        Ok(new_id)
    }

    fn choose<R: Rng>(
        &mut self,
        temperature: f64,
        nbranch: usize,
        rng: &mut R,
    ) -> Option<(usize, usize)> {
        let mut choices = self.pop_valid(nbranch.max(1));
        if choices.is_empty() {
            return None;
        }
        // Reinsertion can disturb heap pop order, so sort the branch window.
        choices.sort_by(|a, b| a.cost.total_cmp(&b.cost));
        let idx = if temperature <= 0.0 || choices.len() == 1 {
            0
        } else {
            let cmin = choices[0].cost;
            let t = temperature * cmin.abs().max(1.0);
            let weights: Vec<f64> = choices
                .iter()
                .map(|c| (-(c.cost - cmin) / t).exp())
                .collect();
            let total: f64 = weights.iter().sum();
            if !total.is_finite() || total <= 0.0 {
                0
            } else {
                let mut r = rng.gen_range(0.0..total);
                let mut chosen = 0;
                for (i, w) in weights.iter().enumerate() {
                    if r < *w {
                        chosen = i;
                        break;
                    }
                    r -= w;
                }
                chosen
            }
        };
        let c = choices.swap_remove(idx);
        for other in choices {
            self.heap.push(other);
        }
        Some((c.a, c.b))
    }
}

/// Run one greedy path-and-slice trial, optionally from a preset slice set.
#[allow(clippy::too_many_arguments)]
fn budgeted_trial<R: Rng>(
    net: &TensorNetwork,
    target: SliceTarget,
    costmod: f64,
    temperature: f64,
    cost_fn: CostFn,
    preslice: &[LegId],
    rng: &mut R,
) -> Result<(SsaPath, Vec<LegId>), String> {
    let mut st = State::new(net, target, costmod, cost_fn, preslice);
    let mut path = SsaPath::new();
    while let Some((a, b)) = st.choose(temperature, 8, rng) {
        st.contract(a, b)?;
        path.push((a, b));
    }
    // Finish disconnected components with outer products.
    let mut rest: Vec<usize> = (0..st.node_legs.len()).filter(|&i| st.alive(i)).collect();
    while rest.len() > 1 {
        rest.sort_by(|&x, &y| {
            st.legs_size(st.node_legs[x].as_ref().unwrap())
                .total_cmp(&st.legs_size(st.node_legs[y].as_ref().unwrap()))
        });
        let (a, b) = (rest[0].min(rest[1]), rest[0].max(rest[1]));
        let new_id = st.contract(a, b)?;
        path.push((a, b));
        rest.remove(0);
        rest.remove(0);
        rest.push(new_id);
    }
    let mut sliced: Vec<LegId> = st.sliced.into_iter().collect();
    sliced.sort_unstable();
    Ok((path, sliced))
}

/// Deadline-aware version of [`budgeted_trial`].
/// Expiry returns `Ok(None)` because a partial SSA path is not valid.
#[allow(clippy::too_many_arguments)]
fn budgeted_trial_until<R: Rng>(
    net: &TensorNetwork,
    target: SliceTarget,
    costmod: f64,
    temperature: f64,
    cost_fn: CostFn,
    preslice: &[LegId],
    rng: &mut R,
    deadline: Option<std::time::Instant>,
) -> Result<Option<(SsaPath, Vec<LegId>)>, String> {
    if deadline.is_none() {
        return budgeted_trial(net, target, costmod, temperature, cost_fn, preslice, rng).map(Some);
    }
    if deadline_reached(deadline) {
        return Ok(None);
    }
    let mut st = State::new(net, target, costmod, cost_fn, preslice);
    let mut path = SsaPath::new();
    loop {
        if deadline_reached(deadline) {
            return Ok(None);
        }
        let Some((a, b)) = st.choose(temperature, 8, rng) else {
            break;
        };
        st.contract(a, b)?;
        path.push((a, b));
    }
    // Finish disconnected components with outer products.
    let mut rest: Vec<usize> = (0..st.node_legs.len()).filter(|&i| st.alive(i)).collect();
    while rest.len() > 1 {
        if deadline_reached(deadline) {
            return Ok(None);
        }
        rest.sort_by(|&x, &y| {
            st.legs_size(st.node_legs[x].as_ref().unwrap())
                .total_cmp(&st.legs_size(st.node_legs[y].as_ref().unwrap()))
        });
        let (a, b) = (rest[0].min(rest[1]), rest[0].max(rest[1]));
        let new_id = st.contract(a, b)?;
        path.push((a, b));
        rest.remove(0);
        rest.remove(0);
        rest.push(new_id);
    }
    let mut sliced: Vec<LegId> = st.sliced.into_iter().collect();
    sliced.sort_unstable();
    Ok(Some((path, sliced)))
}

/// Evaluate a path and slice set exactly across all slices.
fn evaluate(
    net: &TensorNetwork,
    path: &SsaPath,
    sliced: &[LegId],
) -> Result<BudgetedResult, String> {
    let mut cut = net.clone();
    for &l in sliced {
        cut.size_dict.insert(l, 1);
    }
    let per_slice = simulate_path(&cut, path)?;
    let log2_n: f64 = sliced.iter().map(|&l| net.log2_dim(l)).sum();
    Ok(BudgetedResult {
        path: path.clone(),
        sliced: sliced.to_vec(),
        per_slice,
        log2_n_slices: log2_n,
        // Total FLOPs equal per-slice FLOPs times the slice count.
        log10_flops_total: per_slice.log10_flops + log2_n * std::f64::consts::LOG10_2,
    })
}

/// Joint path-and-slice search under an intermediate-size target.
///
/// Trials either start without slices or reuse slices derived from a seed path.
pub fn budgeted_random_greedy(
    net: &TensorNetwork,
    budget_log2: f64,
    ntrials: usize,
    seed: u64,
) -> Result<BudgetedResult, String> {
    budgeted_random_greedy_with_objective(net, budget_log2, ntrials, seed, PlannerObjective::FIXED)
}

/// [`budgeted_random_greedy`] with a per-call planner objective.
pub fn budgeted_random_greedy_with_objective(
    net: &TensorNetwork,
    budget_log2: f64,
    ntrials: usize,
    seed: u64,
    objective: PlannerObjective,
) -> Result<BudgetedResult, String> {
    budgeted_random_greedy_seeded_with_objective(net, budget_log2, ntrials, seed, None, objective)
}

/// Joint search with an optional external path for deriving seed slices.
pub fn budgeted_random_greedy_seeded(
    net: &TensorNetwork,
    budget_log2: f64,
    ntrials: usize,
    seed: u64,
    seed_path: Option<&SsaPath>,
) -> Result<BudgetedResult, String> {
    budgeted_random_greedy_seeded_with_objective(
        net,
        budget_log2,
        ntrials,
        seed,
        seed_path,
        PlannerObjective::FIXED,
    )
}

/// [`budgeted_random_greedy_seeded`] with a per-call planner objective.
pub fn budgeted_random_greedy_seeded_with_objective(
    net: &TensorNetwork,
    budget_log2: f64,
    ntrials: usize,
    seed: u64,
    seed_path: Option<&SsaPath>,
    objective: PlannerObjective,
) -> Result<BudgetedResult, String> {
    budgeted_random_greedy_seeded_until_with_objective(
        net,
        budget_log2,
        ntrials,
        seed,
        seed_path,
        None,
        objective,
    )
}

/// Deadline-aware version of [`budgeted_random_greedy_seeded`].
///
/// Checks occur before each trial and between contractions. An active
/// contraction completes before control returns.
pub fn budgeted_random_greedy_seeded_until(
    net: &TensorNetwork,
    budget_log2: f64,
    ntrials: usize,
    seed: u64,
    seed_path: Option<&SsaPath>,
    deadline: Option<std::time::Instant>,
) -> Result<BudgetedResult, String> {
    budgeted_random_greedy_seeded_until_with_objective(
        net,
        budget_log2,
        ntrials,
        seed,
        seed_path,
        deadline,
        PlannerObjective::FIXED,
    )
}

/// [`budgeted_random_greedy_seeded_until`] with a per-call planner objective.
#[allow(clippy::too_many_arguments)]
pub fn budgeted_random_greedy_seeded_until_with_objective(
    net: &TensorNetwork,
    budget_log2: f64,
    ntrials: usize,
    seed: u64,
    seed_path: Option<&SsaPath>,
    deadline: Option<std::time::Instant>,
    objective: PlannerObjective,
) -> Result<BudgetedResult, String> {
    budgeted_random_greedy_seeded_until_for_target_with_objective(
        net,
        SliceTarget::LegacyLog2(budget_log2),
        ntrials,
        seed,
        seed_path,
        deadline,
        objective,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn budgeted_random_greedy_seeded_until_for_target_with_objective(
    net: &TensorNetwork,
    target: SliceTarget,
    ntrials: usize,
    seed: u64,
    seed_path: Option<&SsaPath>,
    deadline: Option<std::time::Instant>,
    objective: PlannerObjective,
) -> Result<BudgetedResult, String> {
    use rayon::prelude::*;
    if deadline_reached(deadline) {
        return Err("budgeted deadline exceeded".into());
    }
    // Downstream greedy and slicing kernels read leg dimensions directly.
    net.validate()?;
    // Derive slice seeds from both the external and greedy baselines.
    let slices_of = |p: &SsaPath| -> Vec<LegId> {
        crate::slice::find_slices_until_for_target(net, p, target, deadline)
            .map(|sr| sr.legs)
            .unwrap_or_default()
    };
    let gpath = crate::paths::greedy::greedy_unchecked(net).0;
    if deadline_reached(deadline) {
        return Err("budgeted deadline exceeded".into());
    }
    let seed_a: Vec<LegId> = seed_path.map(slices_of).unwrap_or_default();
    if deadline_reached(deadline) {
        return Err("budgeted deadline exceeded".into());
    }
    let seed_b: Vec<LegId> = slices_of(&gpath);
    if deadline_reached(deadline) {
        return Err("budgeted deadline exceeded".into());
    }
    // Adapt the fraction of unsliced starts to the baseline target gap.
    let base_for_gap: &SsaPath = seed_path.unwrap_or(&gpath);
    let base_peak = crate::path::simulate_path(net, base_for_gap)
        .map(|s| s.log2_max_size)
        .unwrap_or(target.log2_hint());
    if deadline_reached(deadline) {
        return Err("budgeted deadline exceeded".into());
    }
    let gap = (base_peak - target.log2_hint()).max(0.0);
    let pure_frac = (gap / 20.0).clamp(0.15, 0.65);
    let run_trial = |trial| {
        // Expired trials yield no candidate.
        if deadline
            .map(|d| std::time::Instant::now() >= d)
            .unwrap_or(false)
        {
            return None;
        }
        let mut rng = ChaCha8Rng::seed_from_u64(seed.wrapping_add(trial as u64));
        // Match random-greedy anchors and log-uniform sampling ranges.
        let (costmod, temperature) = match trial {
            0 => (1.0, 0.0),
            1 => (4.0, 0.0),
            2 => (8.0, 0.0),
            _ => {
                let cm = 10f64.powf(rng.gen_range(0.1f64.log10()..50f64.log10()));
                let t = 10f64.powf(rng.gen_range(-3.0..0.0));
                (cm, t)
            }
        };
        // Match random-greedy cost-function variants.
        let cost_fn = if trial <= 2 {
            CostFn::MemRemoved
        } else {
            match trial % 5 {
                3 => {
                    let mag = 10f64.powf(rng.gen_range(0.05f64.log10()..0.5f64.log10()));
                    CostFn::Skew(if rng.gen_bool(0.5) { mag } else { -mag })
                }
                4 => match trial % 3 {
                    0 => CostFn::Ratio,
                    1 => CostFn::LogRatio,
                    _ => {
                        let mag = 10f64.powf(rng.gen_range(0.02f64.log10()..0.5f64.log10()));
                        CostFn::HyperDeg(if rng.gen_bool(0.5) { mag } else { -mag })
                    }
                },
                _ => CostFn::MemRemoved,
            }
        };
        // Allocate remaining starts alternately across available slice seeds.
        let r = (trial as f64 + 0.5) / (ntrials.max(1) as f64);
        let pre: &[LegId] = if r < pure_frac {
            &[]
        } else if !seed_a.is_empty() && (trial % 2 == 0 || seed_b.is_empty()) {
            &seed_a
        } else if !seed_b.is_empty() {
            &seed_b
        } else {
            &[]
        };
        let (path, sliced) = budgeted_trial_until(
            net,
            target,
            costmod,
            temperature,
            cost_fn,
            pre,
            &mut rng,
            deadline,
        )
        .ok()??;
        if deadline_reached(deadline) {
            return None;
        }
        evaluate(net, &path, &sliced).ok()
    };
    // With a deadline, launch trials sequentially to avoid queued work after expiry.
    let best = if deadline.is_none() {
        (0..ntrials.max(1))
            .into_par_iter()
            .filter_map(&run_trial)
            .min_by(|a, b| objective_score(objective, a).total_cmp(&objective_score(objective, b)))
    } else {
        (0..ntrials.max(1))
            .filter_map(run_trial)
            .min_by(|a, b| objective_score(objective, a).total_cmp(&objective_score(objective, b)))
    }
    .ok_or("所有 budgeted trial 都失败（预算太紧或开放腿太大？）")?;

    // Reconfigure with the selected slice set fixed.
    let mut cut = net.clone();
    for &l in &best.sliced {
        cut.size_dict.insert(l, 1);
    }
    let mut cands: Vec<BudgetedResult> = vec![best.clone()];
    let polished_path = if deadline_reached(deadline) {
        best.path.clone()
    } else {
        match crate::tree::reconfigure_path_until_with_objective(
            &cut, &best.path, 8, 20, deadline, objective,
        ) {
            Ok((p, _)) if !deadline_reached(deadline) => {
                if let Ok(r) = evaluate(net, &p, &best.sliced) {
                    cands.push(r);
                }
                p
            }
            _ => best.path.clone(),
        }
    };
    // Reselect slices while alternating slicing and path reconfiguration.
    if !deadline_reached(deadline) {
        if let Some((p2, sr)) = crate::slice::slice_and_reconf_until_for_target_with_objective(
            net,
            &polished_path,
            target,
            3,
            8,
            deadline,
            objective,
        ) {
            cands.push(BudgetedResult {
                path: p2,
                sliced: sr.legs.clone(),
                per_slice: sr.per_slice,
                log2_n_slices: sr.log2_n_slices,
                log10_flops_total: sr.log10_flops_total,
            });
        }
    }
    // Every candidate must satisfy the target independently of its score.
    let valid: Vec<BudgetedResult> = cands
        .into_iter()
        .filter(|result| match target {
            SliceTarget::LegacyLog2(limit) => result.per_slice.log2_max_size <= limit + 1e-9,
            SliceTarget::Elements(limit) => crate::slice::slice_result_fits_target_size(
                net,
                &result.path,
                &crate::slice::SliceResult {
                    legs: result.sliced.clone(),
                    log2_n_slices: result.log2_n_slices,
                    per_slice: result.per_slice,
                    log10_flops_total: result.log10_flops_total,
                },
                limit.get(),
            )
            .unwrap_or(false),
        })
        .collect();
    valid
        .into_iter()
        .min_by(|a, b| objective_score(objective, a).total_cmp(&objective_score(objective, b)))
        .ok_or_else(|| "所有候选都超预算（不应发生：trial 解构造性守预算）".to_string())
}

/// Select the better result from joint search and a two-stage baseline.
pub fn budgeted_portfolio(
    net: &TensorNetwork,
    budget_log2: f64,
    ntrials: usize,
    seed: u64,
    seed_path: Option<&SsaPath>,
) -> Result<BudgetedResult, String> {
    budgeted_portfolio_with_objective(
        net,
        budget_log2,
        ntrials,
        seed,
        seed_path,
        PlannerObjective::FIXED,
    )
}

/// [`budgeted_portfolio`] with a per-call planner objective.
pub fn budgeted_portfolio_with_objective(
    net: &TensorNetwork,
    budget_log2: f64,
    ntrials: usize,
    seed: u64,
    seed_path: Option<&SsaPath>,
    objective: PlannerObjective,
) -> Result<BudgetedResult, String> {
    budgeted_portfolio_until_with_objective(
        net,
        budget_log2,
        ntrials,
        seed,
        seed_path,
        None,
        objective,
    )
}

/// Deadline-aware version of [`budgeted_portfolio`].
/// Expiry prevents new finishing stages and preserves complete candidates.
pub fn budgeted_portfolio_until(
    net: &TensorNetwork,
    budget_log2: f64,
    ntrials: usize,
    seed: u64,
    seed_path: Option<&SsaPath>,
    deadline: Option<std::time::Instant>,
) -> Result<BudgetedResult, String> {
    budgeted_portfolio_until_with_objective(
        net,
        budget_log2,
        ntrials,
        seed,
        seed_path,
        deadline,
        PlannerObjective::FIXED,
    )
}

/// [`budgeted_portfolio_until`] with a per-call planner objective.
#[allow(clippy::too_many_arguments)]
pub fn budgeted_portfolio_until_with_objective(
    net: &TensorNetwork,
    budget_log2: f64,
    ntrials: usize,
    seed: u64,
    seed_path: Option<&SsaPath>,
    deadline: Option<std::time::Instant>,
    objective: PlannerObjective,
) -> Result<BudgetedResult, String> {
    budgeted_portfolio_until_for_target_with_objective(
        net,
        SliceTarget::LegacyLog2(budget_log2),
        ntrials,
        seed,
        seed_path,
        deadline,
        objective,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn budgeted_portfolio_until_for_target_with_objective(
    net: &TensorNetwork,
    target: SliceTarget,
    ntrials: usize,
    seed: u64,
    seed_path: Option<&SsaPath>,
    deadline: Option<std::time::Instant>,
    objective: PlannerObjective,
) -> Result<BudgetedResult, String> {
    // Validate before either the joint or two-stage branch runs.
    if deadline_reached(deadline) {
        return Err("budgeted deadline exceeded".into());
    }
    net.validate()?;
    let joint = budgeted_random_greedy_seeded_until_for_target_with_objective(
        net, target, ntrials, seed, seed_path, deadline, objective,
    );
    let past =
        |d: Option<std::time::Instant>| d.map(|x| std::time::Instant::now() >= x).unwrap_or(false);
    // Two-stage candidate: baseline path followed by slicing.
    let owned;
    let base: &SsaPath = match seed_path {
        Some(p) => p,
        None => {
            if past(deadline) {
                // Without a seed path, an expired deadline cannot start the baseline.
                return joint;
            }
            let (rp, rs) = crate::paths::greedy::random_greedy_unchecked_with_objective(
                net, ntrials, seed, objective,
            );
            if past(deadline) {
                return joint;
            }
            // Reconfiguration is a proposal generator. Commit it only when the
            // complete-path score does not increase.
            owned = match crate::tree::reconfigure_path_until_with_objective(
                net, &rp, 8, 20, deadline, objective,
            ) {
                Ok((p, s)) if objective.score_path_log2(&s) <= objective.score_path_log2(&rs) => p,
                _ => rp,
            };
            &owned
        }
    };
    let two = if past(deadline) {
        None
    } else {
        crate::slice::slice_and_reconf_until_for_target_with_objective(
            net, base, target, 3, 8, deadline, objective,
        )
        .map(|(p, sr)| BudgetedResult {
            path: p,
            sliced: sr.legs.clone(),
            per_slice: sr.per_slice,
            log2_n_slices: sr.log2_n_slices,
            log10_flops_total: sr.log10_flops_total,
        })
    };
    match (joint, two) {
        (Ok(a), Some(b)) => Ok(
            if objective_score(objective, &a) <= objective_score(objective, &b) {
                a
            } else {
                b
            },
        ),
        (Ok(a), None) => Ok(a),
        (Err(_), Some(b)) => Ok(b),
        (Err(e), None) => Err(e),
    }
}

#[inline]
fn deadline_reached(deadline: Option<std::time::Instant>) -> bool {
    deadline
        .map(|d| std::time::Instant::now() >= d)
        .unwrap_or(false)
}
