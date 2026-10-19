//! Tensor-network simplification before path search.
//!
//! The pass applies path-independent low-rank contractions:
//! - rank 0: absorb into the smallest live tensor;
//! - rank 1: absorb into a tensor sharing its only leg;
//! - rank 2: absorb when the neighbor's rank does not increase.
//!
//! Simplification emits valid pairwise contractions as an SSA prefix. [`stitch`]
//! maps a path on the reduced network back to the original network.

use std::collections::{BTreeSet, HashMap, HashSet};

use crate::network::{LegId, TensorNetwork};
use crate::path::{simulate_path, sorted_dedup, PathStats, SsaPath};

pub struct Simplified {
    /// Simplification steps in original-network SSA numbering.
    pub prefix: SsaPath,
    /// Reduced network with original leg IDs and output order.
    pub reduced: TensorNetwork,
    /// Original SSA node represented by each reduced tensor.
    pub map: Vec<usize>,
}

/// Lightweight simplification state with leg slots, reference counts, and holders.
struct State {
    legs: Vec<Option<Vec<LegId>>>,
    refcount: HashMap<LegId, usize>,
    /// Live holders of each leg, ordered so the lowest-ID neighbor is cheap to read.
    holders: HashMap<LegId, BTreeSet<usize>>,
    /// Live holders of rank at most two, the only tensors whose decisions are queued.
    low_holders: HashMap<LegId, BTreeSet<usize>>,
    in_output: HashSet<LegId>,
    prefix: SsaPath,
    live_by_rank: BTreeSet<(usize, usize)>,
    live_count: usize,
}

/// A leg touched by one contraction.
struct ChangedLeg {
    leg: LegId,
    /// Lower contracted operand ID that held this leg.
    removed_min: usize,
}

struct ContractEffect {
    new_id: usize,
    changed: Vec<ChangedLeg>,
}

impl State {
    fn new(net: &TensorNetwork) -> Self {
        let in_output: HashSet<LegId> = net.output.iter().copied().collect();
        let mut refcount: HashMap<LegId, usize> = HashMap::new();
        let mut holders: HashMap<LegId, BTreeSet<usize>> = HashMap::new();
        let mut low_holders: HashMap<LegId, BTreeSet<usize>> = HashMap::new();
        let mut legs = Vec::new();
        for (i, t) in net.inputs.iter().enumerate() {
            let d = sorted_dedup(t);
            for &l in &d {
                *refcount.entry(l).or_insert(0) += 1;
                holders.entry(l).or_default().insert(i);
                if d.len() <= 2 {
                    low_holders.entry(l).or_default().insert(i);
                }
            }
            legs.push(Some(d));
        }
        let live_by_rank = legs
            .iter()
            .enumerate()
            .map(|(id, legs)| (legs.as_ref().unwrap().len(), id))
            .collect();
        let live_count = legs.len();
        State {
            legs,
            refcount,
            holders,
            low_holders,
            in_output,
            prefix: SsaPath::new(),
            live_by_rank,
            live_count,
        }
    }

    fn alive(&self, i: usize) -> bool {
        self.legs.get(i).map(|x| x.is_some()).unwrap_or(false)
    }

    fn rank(&self, i: usize) -> usize {
        self.legs[i].as_ref().map(|l| l.len()).unwrap_or(usize::MAX)
    }

    /// Returns result legs using the shared contraction rule.
    fn result_legs(&self, a: usize, b: usize) -> Vec<LegId> {
        crate::path::result_legs(
            self.legs[a].as_ref().unwrap(),
            self.legs[b].as_ref().unwrap(),
            &self.refcount,
            &self.in_output,
        )
    }

    /// Returns a live holder of `l` other than `me`.
    fn other_holder(&self, l: LegId, me: usize) -> Option<usize> {
        // Holder sets contain only live tensors; the minimum keeps output deterministic.
        self.holders.get(&l)?.iter().copied().find(|&h| h != me)
    }

    /// Returns the contraction tensor `i` would perform now, if any.
    fn decide(&self, i: usize) -> Option<(usize, usize)> {
        if !self.alive(i) {
            return None;
        }
        match self.rank(i) {
            0 => self.scalar_target(i).map(|j| (i, j)),
            1 => {
                let leg = self.legs[i].as_ref().unwrap()[0];
                self.other_holder(leg, i).map(|j| (i, j))
            }
            2 => {
                for &leg in self.legs[i].as_ref().unwrap() {
                    if let Some(j) = self.other_holder(leg, i) {
                        let result = self.result_legs(i.min(j), i.max(j));
                        if result.len() <= self.rank(j) {
                            return Some((i, j));
                        }
                    }
                }
                None
            }
            _ => None,
        }
    }

    /// Applies a contraction and records the legs whose neighborhood changed.
    fn contract(&mut self, a: usize, b: usize) -> ContractEffect {
        let result = self.result_legs(a, b);
        let la = self.legs[a].take().unwrap();
        let lb = self.legs[b].take().unwrap();
        let changed = crate::path::legs_union(&la, &lb)
            .into_iter()
            .map(|leg| {
                let in_a = la.binary_search(&leg).is_ok();
                let in_b = lb.binary_search(&leg).is_ok();
                let removed_min = match (in_a, in_b) {
                    (true, true) => a.min(b),
                    (true, false) => a,
                    _ => b,
                };
                ChangedLeg { leg, removed_min }
            })
            .collect();
        self.live_by_rank.remove(&(la.len(), a));
        self.live_by_rank.remove(&(lb.len(), b));
        for (id, ls) in [(a, &la), (b, &lb)] {
            for &l in ls {
                *self.refcount.get_mut(&l).unwrap() -= 1;
                self.holders.get_mut(&l).unwrap().remove(&id);
                if ls.len() <= 2 {
                    self.low_holders.get_mut(&l).unwrap().remove(&id);
                }
            }
        }
        let new_id = self.legs.len();
        for &l in &result {
            *self.refcount.get_mut(&l).unwrap() += 1;
            self.holders.get_mut(&l).unwrap().insert(new_id);
            if result.len() <= 2 {
                self.low_holders.entry(l).or_default().insert(new_id);
            }
        }
        self.legs.push(Some(result));
        self.live_by_rank
            .insert((self.legs[new_id].as_ref().unwrap().len(), new_id));
        self.live_count -= 1;
        let (x, y) = if a < b { (a, b) } else { (b, a) };
        self.prefix.push((x, y));
        ContractEffect { new_id, changed }
    }

    /// Preserves the scan's lowest-rank, then lowest-ID scalar target.
    fn scalar_target(&self, me: usize) -> Option<usize> {
        self.live_by_rank
            .iter()
            .map(|&(_, id)| id)
            .find(|&id| id != me)
    }

    /// Queues every low-rank tensor whose decision may have changed.
    ///
    /// Tensors outside `pending` keep a decision of "no contraction", so it is
    /// enough to queue tensors whose decision inputs changed. A decision reads,
    /// per leg, the lowest-ID other holder `j`, then `j`'s legs and rank and the
    /// retained-leg test for the pair. Only the lowest other holder can change:
    ///
    /// - `j`'s legs and rank are fixed while `j` lives.
    /// - The retained-leg test keeps a leg when a holder other than the pair
    ///   exists. If both members of the pair survive and hold a changed leg,
    ///   that leg had a contracted operand as another holder before and has the
    ///   new tensor after, because a leg with surviving holders is retained.
    ///
    /// Holders only lose the operands and gain the new tensor, whose ID exceeds
    /// every live ID, so the lowest other holder changes only when a removed
    /// operand was lower than every survivor seen by that tensor.
    fn requeue_affected(&self, effect: &ContractEffect, pending: &mut BTreeSet<usize>) {
        if self.rank(effect.new_id) <= 2 {
            pending.insert(effect.new_id);
        }
        for changed in &effect.changed {
            let leg = changed.leg;
            let mut survivors = self
                .holders
                .get(&leg)
                .into_iter()
                .flatten()
                .copied()
                .filter(|&id| id != effect.new_id);
            let Some(first) = survivors.next() else {
                continue;
            };
            if first > changed.removed_min {
                // Every survivor previously saw a removed operand as its neighbor.
                if let Some(low) = self.low_holders.get(&leg) {
                    pending.extend(low.iter().copied().filter(|&id| id != effect.new_id));
                }
            } else if survivors
                .next()
                .is_none_or(|second| second > changed.removed_min)
                && self.rank(first) <= 2
            {
                // Only the lowest survivor saw a removed operand as its neighbor.
                pending.insert(first);
            }
        }
    }
}

/// Simplifies a network to a fixed point; every contraction removes one tensor.
pub fn simplify(net: &TensorNetwork) -> Simplified {
    let mut st = State::new(net);
    let mut pending: BTreeSet<usize> = st
        .live_by_rank
        .iter()
        .filter_map(|&(rank, id)| (rank <= 2).then_some(id))
        .collect();
    while st.live_count > 1 {
        let Some(i) = pending.pop_first() else {
            break;
        };
        let pair = st.decide(i);
        if let Some((a, b)) = pair {
            let effect = st.contract(a, b);
            st.requeue_affected(&effect, &mut pending);
        }
    }

    // Collect live tensors into the reduced network.
    let mut map = Vec::new();
    let mut inputs = Vec::new();
    for (id, l) in st.legs.iter().enumerate() {
        if let Some(l) = l {
            map.push(id);
            inputs.push(l.clone());
        }
    }
    let mut size_dict = HashMap::new();
    for legs in &inputs {
        for &l in legs {
            size_dict.insert(l, net.dim(l));
        }
    }
    for &l in &net.output {
        size_dict.insert(l, net.dim(l));
    }
    Simplified {
        prefix: st.prefix,
        reduced: TensorNetwork {
            name: format!("{}-s", net.name),
            inputs,
            output: net.output.clone(),
            size_dict,
        },
        map,
    }
}

/// Stitches the simplification prefix and a reduced-network path into a full SSA path.
pub fn stitch(n_orig: usize, prefix: &SsaPath, map: &[usize], reduced_path: &SsaPath) -> SsaPath {
    let m = map.len();
    let base = n_orig + prefix.len();
    let trans = |x: usize| -> usize {
        if x < m {
            map[x]
        } else {
            base + (x - m)
        }
    };
    let mut full = prefix.clone();
    for &(a, b) in reduced_path {
        let (x, y) = (trans(a), trans(b));
        full.push((x.min(y), x.max(y)));
    }
    full
}

/// Simplifies, runs random-greedy, and stitches the full path.
///
/// Invalid networks return an error before search.
pub fn random_greedy_simplified(
    net: &TensorNetwork,
    ntrials: usize,
    seed: u64,
) -> Result<(SsaPath, PathStats), String> {
    random_greedy_simplified_with_objective(net, ntrials, seed, crate::PlannerObjective::FIXED)
}

/// [`random_greedy_simplified`] with a per-call planner objective.
pub fn random_greedy_simplified_with_objective(
    net: &TensorNetwork,
    ntrials: usize,
    seed: u64,
    objective: crate::PlannerObjective,
) -> Result<(SsaPath, PathStats), String> {
    net.validate()?;
    Ok(random_greedy_simplified_unchecked_with_objective(
        net, ntrials, seed, objective,
    ))
}

/// Simplified random-greedy for an already validated network.
pub(crate) fn random_greedy_simplified_unchecked_with_objective(
    net: &TensorNetwork,
    ntrials: usize,
    seed: u64,
    objective: crate::PlannerObjective,
) -> (SsaPath, PathStats) {
    let s = simplify(net);
    random_greedy_simplified_prepared_with_objective(net, &s, ntrials, seed, objective)
}

/// Reuses a prepared [`Simplified`] network for random-greedy.
///
/// `prepared` must come from the same validated `net`.
pub(crate) fn random_greedy_simplified_prepared_with_objective(
    net: &TensorNetwork,
    prepared: &Simplified,
    ntrials: usize,
    seed: u64,
    objective: crate::PlannerObjective,
) -> (SsaPath, PathStats) {
    let full = if prepared.reduced.n_tensors() <= 1 {
        prepared.prefix.clone()
    } else {
        let (rp, _) = crate::paths::greedy::random_greedy_unchecked_with_objective(
            &prepared.reduced,
            ntrials,
            seed,
            objective,
        );
        stitch(net.n_tensors(), &prepared.prefix, &prepared.map, &rp)
    };
    let stats = simulate_path(net, &full).expect("simplified 路径非法");
    (full, stats)
}

/// Tempers random-greedy and bisection starts on a reduced network, optionally
/// runs leaf-order DP, and stitches the best path to the original network.
#[allow(clippy::too_many_arguments)]
pub fn temper_simplified(
    net: &TensorNetwork,
    ntrials: usize,
    seed: u64,
    chains: usize,
    rounds: usize,
    moves: usize,
    tmin: f64,
    tmax: f64,
    reconf_interval: usize,
    reconf_size: usize,
) -> Result<(SsaPath, PathStats), String> {
    temper_simplified_with_objective(
        net,
        ntrials,
        seed,
        chains,
        rounds,
        moves,
        tmin,
        tmax,
        reconf_interval,
        reconf_size,
        crate::PlannerObjective::FIXED,
    )
}

/// [`temper_simplified`] with a per-call planner objective.
#[allow(clippy::too_many_arguments)]
pub fn temper_simplified_with_objective(
    net: &TensorNetwork,
    ntrials: usize,
    seed: u64,
    chains: usize,
    rounds: usize,
    moves: usize,
    tmin: f64,
    tmax: f64,
    reconf_interval: usize,
    reconf_size: usize,
    objective: crate::PlannerObjective,
) -> Result<(SsaPath, PathStats), String> {
    net.validate()?;
    let s = simplify(net);
    temper_simplified_prepared_with_objective(
        net,
        &s,
        ntrials,
        seed,
        chains,
        rounds,
        moves,
        tmin,
        tmax,
        reconf_interval,
        reconf_size,
        objective,
    )
}

/// Uses an existing [`Simplified`] network for objective-aware tempering.
///
/// `prepared` must have been produced from the validated `net` supplied here.
#[allow(clippy::too_many_arguments)]
pub(crate) fn temper_simplified_prepared_with_objective(
    net: &TensorNetwork,
    prepared: &Simplified,
    ntrials: usize,
    seed: u64,
    chains: usize,
    rounds: usize,
    moves: usize,
    tmin: f64,
    tmax: f64,
    reconf_interval: usize,
    reconf_size: usize,
    objective: crate::PlannerObjective,
) -> Result<(SsaPath, PathStats), String> {
    let s = prepared;
    // Zero selects a size-dependent round count.
    let rounds = if rounds == 0 {
        (net.n_tensors() * 1000 / 25_000).clamp(8, 80)
    } else {
        rounds
    };
    let full = if s.reduced.n_tensors() <= 1 {
        s.prefix.clone()
    } else {
        // Collect available random-greedy and bisection starts.
        let mut inits: Vec<(SsaPath, PathStats)> = Vec::new();
        let (rp, rs) = crate::paths::greedy::random_greedy_unchecked_with_objective(
            &s.reduced, ntrials, seed, objective,
        );
        inits.push((rp, rs));
        if let Ok((bp, bs)) =
            crate::paths::bisect::bisect_with_objective(&s.reduced, ntrials, seed, 12, objective)
        {
            inits.push((bp, bs));
        }
        inits.sort_by(|a, b| {
            objective
                .score_path_log2(&a.1)
                .total_cmp(&objective.score_path_log2(&b.1))
        }); // Assign the best candidate to the lowest temperature.
        let init_paths: Vec<SsaPath> = inits.into_iter().map(|(p, _)| p).collect();
        let (tp, ts) = crate::tree::temper_paths_with_objective(
            &s.reduced,
            &init_paths,
            chains,
            rounds,
            moves,
            tmin,
            tmax,
            reconf_interval,
            reconf_size,
            seed,
            0,
            objective,
        )?;
        // Skip O(n^3) time and O(n^2) memory leaf-order DP on large networks.
        let better = if s.reduced.n_tensors() <= 5_000 {
            let order = crate::paths::ordertree::leaf_order_of_path_with_objective(
                &s.reduced, &tp, objective,
            )?;
            let (dp, ds) =
                crate::paths::ordertree::order_dp_with_objective(&s.reduced, &order, objective)?;
            let accept = objective.score_path_log2(&ds) <= objective.score_path_log2(&ts);
            if accept {
                dp
            } else {
                tp
            }
        } else {
            tp
        };
        stitch(net.n_tensors(), &s.prefix, &s.map, &better)
    };
    let stats = simulate_path(net, &full)?;
    Ok((full, stats))
}

/// Cooperative-deadline variant of [`temper_simplified`].
///
/// Optional work does not start after the deadline. An atomic unit that crosses
/// the deadline still delivers and scores its incumbent, so callers should
/// reserve a bounded finalization window.
#[allow(clippy::too_many_arguments)]
pub fn temper_simplified_until(
    net: &TensorNetwork,
    ntrials: usize,
    seed: u64,
    chains: usize,
    rounds: usize,
    moves: usize,
    tmin: f64,
    tmax: f64,
    reconf_interval: usize,
    reconf_size: usize,
    deadline: Option<std::time::Instant>,
) -> Result<(SsaPath, PathStats), String> {
    temper_simplified_until_with_objective(
        net,
        ntrials,
        seed,
        chains,
        rounds,
        moves,
        tmin,
        tmax,
        reconf_interval,
        reconf_size,
        deadline,
        crate::PlannerObjective::FIXED,
    )
}

/// [`temper_simplified_until`] with a per-call planner objective.
#[allow(clippy::too_many_arguments)]
pub fn temper_simplified_until_with_objective(
    net: &TensorNetwork,
    ntrials: usize,
    seed: u64,
    chains: usize,
    rounds: usize,
    moves: usize,
    tmin: f64,
    tmax: f64,
    reconf_interval: usize,
    reconf_size: usize,
    deadline: Option<std::time::Instant>,
    objective: crate::PlannerObjective,
) -> Result<(SsaPath, PathStats), String> {
    if deadline.is_none() {
        return temper_simplified_with_objective(
            net,
            ntrials,
            seed,
            chains,
            rounds,
            moves,
            tmin,
            tmax,
            reconf_interval,
            reconf_size,
            objective,
        );
    }
    if deadline_reached(deadline) {
        return Err("s-temper deadline exceeded".into());
    }
    // Validate before simplification so malformed networks return errors.
    net.validate()?;
    let s = simplify(net);
    temper_simplified_prepared_until_with_objective(
        net,
        &s,
        ntrials,
        seed,
        chains,
        rounds,
        moves,
        tmin,
        tmax,
        reconf_interval,
        reconf_size,
        deadline,
        objective,
    )
}

/// Cooperative-deadline variant of [`temper_simplified_prepared_with_objective`].
///
/// The caller has already completed `simplify(net)`. If the deadline expires,
/// a single-tensor reduced network can still deliver its complete prefix.
#[allow(clippy::too_many_arguments)]
pub(crate) fn temper_simplified_prepared_until_with_objective(
    net: &TensorNetwork,
    prepared: &Simplified,
    ntrials: usize,
    seed: u64,
    chains: usize,
    rounds: usize,
    moves: usize,
    tmin: f64,
    tmax: f64,
    reconf_interval: usize,
    reconf_size: usize,
    deadline: Option<std::time::Instant>,
    objective: crate::PlannerObjective,
) -> Result<(SsaPath, PathStats), String> {
    if deadline.is_none() {
        return temper_simplified_prepared_with_objective(
            net,
            prepared,
            ntrials,
            seed,
            chains,
            rounds,
            moves,
            tmin,
            tmax,
            reconf_interval,
            reconf_size,
            objective,
        );
    }
    let s = prepared;
    if deadline_reached(deadline) && s.reduced.n_tensors() > 1 {
        return Err("s-temper deadline exceeded".into());
    }
    let rounds = if rounds == 0 {
        (net.n_tensors() * 1000 / 25_000).clamp(8, 80)
    } else {
        rounds
    };
    let full = if s.reduced.n_tensors() <= 1 {
        s.prefix.clone()
    } else {
        let mut inits: Vec<(SsaPath, PathStats)> = Vec::new();
        let (rp, rs) = crate::paths::greedy::random_greedy_unchecked_with_objective(
            &s.reduced, ntrials, seed, objective,
        );
        inits.push((rp, rs));
        if !deadline_reached(deadline) {
            if let Ok((bp, bs)) = crate::paths::bisect::bisect_with_objective(
                &s.reduced, ntrials, seed, 12, objective,
            ) {
                inits.push((bp, bs));
            }
        }
        inits.sort_by(|a, b| {
            objective
                .score_path_log2(&a.1)
                .total_cmp(&objective.score_path_log2(&b.1))
        });
        let mut best = inits[0].clone();
        if !deadline_reached(deadline) {
            let init_paths: Vec<SsaPath> = inits.iter().map(|(p, _)| p.clone()).collect();
            best = crate::tree::temper_paths_until_with_objective(
                &s.reduced,
                &init_paths,
                chains,
                rounds,
                moves,
                tmin,
                tmax,
                reconf_interval,
                reconf_size,
                seed,
                0,
                deadline,
                objective,
            )?;
        }
        if s.reduced.n_tensors() <= 5_000 && !deadline_reached(deadline) {
            let order = crate::paths::ordertree::leaf_order_of_path_with_objective(
                &s.reduced, &best.0, objective,
            )?;
            if !deadline_reached(deadline) {
                if let Ok((dp, ds)) = crate::paths::ordertree::order_dp_until_with_objective(
                    &s.reduced, &order, deadline, objective,
                ) {
                    let accept =
                        objective.score_path_log2(&ds) <= objective.score_path_log2(&best.1);
                    if accept {
                        best = (dp, ds);
                    }
                }
            }
        }
        // Stitching and simulation are required to deliver a completed incumbent.
        stitch(net.n_tensors(), &s.prefix, &s.map, &best.0)
    };
    let stats = simulate_path(net, &full)?;
    Ok((full, stats))
}

#[inline]
fn deadline_reached(deadline: Option<std::time::Instant>) -> bool {
    deadline
        .map(|d| std::time::Instant::now() >= d)
        .unwrap_or(false)
}

/// Simplifies, runs bisection, and stitches the full path.
pub fn bisect_simplified(
    net: &TensorNetwork,
    ntrials: usize,
    seed: u64,
    cutoff: usize,
) -> Result<(SsaPath, PathStats), String> {
    bisect_simplified_with_objective(net, ntrials, seed, cutoff, crate::PlannerObjective::FIXED)
}

/// [`bisect_simplified`] with a per-call planner objective.
pub fn bisect_simplified_with_objective(
    net: &TensorNetwork,
    ntrials: usize,
    seed: u64,
    cutoff: usize,
    objective: crate::PlannerObjective,
) -> Result<(SsaPath, PathStats), String> {
    net.validate()?;
    let s = simplify(net);
    bisect_simplified_prepared_with_objective(net, &s, ntrials, seed, cutoff, objective)
}

/// Reuses a prepared [`Simplified`] network for bisection.
///
/// `prepared` must come from the same validated `net`.
pub(crate) fn bisect_simplified_prepared_with_objective(
    net: &TensorNetwork,
    prepared: &Simplified,
    ntrials: usize,
    seed: u64,
    cutoff: usize,
    objective: crate::PlannerObjective,
) -> Result<(SsaPath, PathStats), String> {
    let full = if prepared.reduced.n_tensors() <= 1 {
        prepared.prefix.clone()
    } else {
        let (bp, _) = crate::paths::bisect::bisect_with_objective(
            &prepared.reduced,
            ntrials,
            seed,
            cutoff,
            objective,
        )?;
        stitch(net.n_tensors(), &prepared.prefix, &prepared.map, &bp)
    };
    let stats = simulate_path(net, &full)?;
    Ok((full, stats))
}

#[cfg(test)]
mod prepared_tests {
    use super::*;
    use std::time::{Duration, Instant};

    /// Test network that retains four rank-3 tensors after scalar simplification.
    fn reusable_toy() -> TensorNetwork {
        TensorNetwork {
            name: "prepared-simplify-toy".into(),
            inputs: vec![
                vec![],
                vec![0, 1, 2],
                vec![0, 3, 4],
                vec![1, 3, 5],
                vec![2, 4, 5],
            ],
            output: vec![],
            size_dict: (0..6u32).map(|leg| (leg, 2usize)).collect(),
        }
    }

    fn assert_same_candidate(a: &(SsaPath, PathStats), b: &(SsaPath, PathStats)) {
        assert_eq!(a.0, b.0);
        assert_eq!(a.1.log10_flops.to_bits(), b.1.log10_flops.to_bits());
        assert_eq!(a.1.log2_max_size.to_bits(), b.1.log2_max_size.to_bits());
        assert_eq!(
            a.1.log2_max_contraction_size.to_bits(),
            b.1.log2_max_contraction_size.to_bits()
        );
        assert_eq!(a.1.log2_total_size.to_bits(), b.1.log2_total_size.to_bits());
        assert_eq!(a.1.log2_peak_size.to_bits(), b.1.log2_peak_size.to_bits());
    }

    #[test]
    fn prepared_random_and_bisect_match_public_wrappers_bitwise() {
        let net = reusable_toy();
        net.validate().unwrap();
        let prepared = simplify(&net);
        assert_eq!(prepared.prefix.len(), 1);
        assert_eq!(prepared.reduced.n_tensors(), 4);

        let fresh_random = random_greedy_simplified(&net, 8, 17).unwrap();
        let reused_random = random_greedy_simplified_prepared_with_objective(
            &net,
            &prepared,
            8,
            17,
            crate::PlannerObjective::FIXED,
        );
        assert_same_candidate(&fresh_random, &reused_random);

        let fresh_bisect = bisect_simplified(&net, 8, 23, 2).unwrap();
        let reused_bisect = bisect_simplified_prepared_with_objective(
            &net,
            &prepared,
            8,
            23,
            2,
            crate::PlannerObjective::FIXED,
        )
        .unwrap();
        assert_same_candidate(&fresh_bisect, &reused_bisect);

        let fresh_random = random_greedy_simplified(&net, 8, 31).unwrap();
        let reused_random = random_greedy_simplified_prepared_with_objective(
            &net,
            &prepared,
            8,
            31,
            crate::PlannerObjective::FIXED,
        );
        assert_same_candidate(&fresh_random, &reused_random);
        let fresh_bisect = bisect_simplified(&net, 8, 37, 2).unwrap();
        let reused_bisect = bisect_simplified_prepared_with_objective(
            &net,
            &prepared,
            8,
            37,
            2,
            crate::PlannerObjective::FIXED,
        )
        .unwrap();
        assert_same_candidate(&fresh_bisect, &reused_bisect);
    }

    #[test]
    fn prepared_temper_matches_public_wrapper_bitwise() {
        let net = reusable_toy();
        let prepared = simplify(&net);
        let fresh = temper_simplified(&net, 6, 41, 2, 2, 64, 1e-4, 0.15, 0, 4).unwrap();
        let reused = temper_simplified_prepared_with_objective(
            &net,
            &prepared,
            6,
            41,
            2,
            2,
            64,
            1e-4,
            0.15,
            0,
            4,
            crate::PlannerObjective::FIXED,
        )
        .unwrap();
        assert_same_candidate(&fresh, &reused);
    }

    #[test]
    fn prepared_temper_deadline_matches_public_wrapper() {
        let net = reusable_toy();
        let prepared = simplify(&net);
        let fresh = temper_simplified_until(
            &net,
            4,
            53,
            2,
            1,
            32,
            1e-4,
            0.15,
            0,
            4,
            Some(Instant::now() + Duration::from_secs(5)),
        )
        .unwrap();
        let reused = temper_simplified_prepared_until_with_objective(
            &net,
            &prepared,
            4,
            53,
            2,
            1,
            32,
            1e-4,
            0.15,
            0,
            4,
            Some(Instant::now() + Duration::from_secs(5)),
            crate::PlannerObjective::FIXED,
        )
        .unwrap();
        assert_same_candidate(&fresh, &reused);

        let expired = temper_simplified_prepared_until_with_objective(
            &net,
            &prepared,
            4,
            53,
            2,
            1,
            32,
            1e-4,
            0.15,
            0,
            4,
            Some(Instant::now() - Duration::from_millis(1)),
            crate::PlannerObjective::FIXED,
        );
        assert!(expired.is_err());
    }
}

#[cfg(test)]
mod queue_tests {
    use super::*;

    // Verbatim State and simplify from the main-branch baseline. This copy must
    // not share the new queue, live-rank index, or contraction implementation.
    mod scan_reference {
        use super::*;
        /// Lightweight simplification state with leg slots, reference counts, and holders.
        struct State {
            legs: Vec<Option<Vec<LegId>>>,
            refcount: HashMap<LegId, usize>,
            holders: HashMap<LegId, HashSet<usize>>,
            in_output: HashSet<LegId>,
            prefix: SsaPath,
        }

        impl State {
            fn new(net: &TensorNetwork) -> Self {
                let in_output: HashSet<LegId> = net.output.iter().copied().collect();
                let mut refcount: HashMap<LegId, usize> = HashMap::new();
                let mut holders: HashMap<LegId, HashSet<usize>> = HashMap::new();
                let mut legs = Vec::new();
                for (i, t) in net.inputs.iter().enumerate() {
                    let d = sorted_dedup(t);
                    for &l in &d {
                        *refcount.entry(l).or_insert(0) += 1;
                        holders.entry(l).or_default().insert(i);
                    }
                    legs.push(Some(d));
                }
                State {
                    legs,
                    refcount,
                    holders,
                    in_output,
                    prefix: SsaPath::new(),
                }
            }

            fn alive(&self, i: usize) -> bool {
                self.legs.get(i).map(|x| x.is_some()).unwrap_or(false)
            }

            fn rank(&self, i: usize) -> usize {
                self.legs[i].as_ref().map(|l| l.len()).unwrap_or(usize::MAX)
            }

            /// Returns result legs using the shared contraction rule.
            fn result_legs(&self, a: usize, b: usize) -> Vec<LegId> {
                crate::path::result_legs(
                    self.legs[a].as_ref().unwrap(),
                    self.legs[b].as_ref().unwrap(),
                    &self.refcount,
                    &self.in_output,
                )
            }

            /// Returns a live holder of `l` other than `me`.
            fn other_holder(&self, l: LegId, me: usize) -> Option<usize> {
                self.holders
                    .get(&l)?
                    .iter()
                    .copied()
                    .filter(|&h| h != me && self.alive(h))
                    .min() // Choose the minimum ID for deterministic output.
            }

            /// Applies a contraction, updates state, and returns the new node ID.
            fn contract(&mut self, a: usize, b: usize) -> usize {
                let result = self.result_legs(a, b);
                let la = self.legs[a].take().unwrap();
                let lb = self.legs[b].take().unwrap();
                for &l in &la {
                    *self.refcount.get_mut(&l).unwrap() -= 1;
                    self.holders.get_mut(&l).unwrap().remove(&a);
                }
                for &l in &lb {
                    *self.refcount.get_mut(&l).unwrap() -= 1;
                    self.holders.get_mut(&l).unwrap().remove(&b);
                }
                let new_id = self.legs.len();
                for &l in &result {
                    *self.refcount.get_mut(&l).unwrap() += 1;
                    self.holders.get_mut(&l).unwrap().insert(new_id);
                }
                self.legs.push(Some(result));
                let (x, y) = if a < b { (a, b) } else { (b, a) };
                self.prefix.push((x, y));
                new_id
            }
        }

        /// Simplifies a network to a fixed point; every contraction removes one tensor.
        pub fn simplify(net: &TensorNetwork) -> Simplified {
            let mut st = State::new(net);
            loop {
                let ids: Vec<usize> = (0..st.legs.len()).filter(|&i| st.alive(i)).collect();
                if ids.len() <= 1 {
                    break;
                }
                let mut fired = false;
                for &i in &ids {
                    if !st.alive(i) {
                        continue;
                    }
                    match st.rank(i) {
                        // Absorb a scalar into the lowest-rank live tensor.
                        0 => {
                            if let Some(j) = ids
                                .iter()
                                .copied()
                                .filter(|&j| j != i && st.alive(j))
                                .min_by_key(|&j| st.rank(j))
                            {
                                st.contract(i, j);
                                fired = true;
                            }
                        }
                        // Absorb a vector into a neighbor sharing its only leg.
                        1 => {
                            let l = st.legs[i].as_ref().unwrap()[0];
                            if let Some(j) = st.other_holder(l, i) {
                                st.contract(i, j);
                                fired = true;
                            }
                        }
                        // Absorb a matrix when the neighbor's rank does not increase.
                        2 => {
                            let ls = st.legs[i].as_ref().unwrap().clone();
                            for &l in &ls {
                                if let Some(j) = st.other_holder(l, i) {
                                    let res = st.result_legs(i.min(j), i.max(j));
                                    if res.len() <= st.rank(j) {
                                        st.contract(i, j);
                                        fired = true;
                                        break;
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                    if fired {
                        break; // Restart the scan after a structural change.
                    }
                }
                if !fired {
                    break; // Fixed point.
                }
            }

            // Collect live tensors into the reduced network.
            let mut map = Vec::new();
            let mut inputs = Vec::new();
            for (id, l) in st.legs.iter().enumerate() {
                if let Some(l) = l {
                    map.push(id);
                    inputs.push(l.clone());
                }
            }
            let mut size_dict = HashMap::new();
            for legs in &inputs {
                for &l in legs {
                    size_dict.insert(l, net.dim(l));
                }
            }
            for &l in &net.output {
                size_dict.insert(l, net.dim(l));
            }
            Simplified {
                prefix: st.prefix,
                reduced: TensorNetwork {
                    name: format!("{}-s", net.name),
                    inputs,
                    output: net.output.clone(),
                    size_dict,
                },
                map,
            }
        }
    }

    fn network(name: &str, inputs: Vec<Vec<LegId>>, output: Vec<LegId>) -> TensorNetwork {
        let size_dict = inputs
            .iter()
            .flatten()
            .copied()
            .map(|leg| (leg, 2 + leg as usize % 3))
            .collect();
        let net = TensorNetwork {
            name: name.into(),
            inputs,
            output,
            size_dict,
        };
        net.validate().unwrap();
        net
    }

    fn assert_same(actual: &Simplified, expected: &Simplified) {
        assert_eq!(actual.prefix, expected.prefix);
        assert_eq!(actual.map, expected.map);
        assert_eq!(actual.reduced.name, expected.reduced.name);
        assert_eq!(actual.reduced.inputs, expected.reduced.inputs);
        assert_eq!(actual.reduced.output, expected.reduced.output);
        assert_eq!(actual.reduced.size_dict, expected.reduced.size_dict);
    }

    // Replay the prefix without State or the production result_legs helper.
    // Counts are rebuilt from the complete live tensor set before every step.
    fn check_prefix(net: &TensorNetwork, simplified: &Simplified) {
        let mut live: std::collections::BTreeMap<usize, BTreeSet<LegId>> = net
            .inputs
            .iter()
            .enumerate()
            .map(|(id, legs)| (id, legs.iter().copied().collect()))
            .collect();
        for (step, &(a, b)) in simplified.prefix.iter().enumerate() {
            assert!(a < b);
            let left = live.get(&a).expect("left operand must be alive");
            let right = live.get(&b).expect("right operand must be alive");
            let result: BTreeSet<_> = left
                .union(right)
                .copied()
                .filter(|leg| {
                    net.output.contains(leg)
                        || live
                            .iter()
                            .any(|(&id, legs)| id != a && id != b && legs.contains(leg))
                })
                .collect();
            live.remove(&a);
            live.remove(&b);
            live.insert(net.n_tensors() + step, result);
        }
        assert_eq!(live.keys().copied().collect::<Vec<_>>(), simplified.map);
        let inputs: Vec<Vec<_>> = live
            .values()
            .map(|legs| legs.iter().copied().collect())
            .collect();
        assert_eq!(inputs, simplified.reduced.inputs);
        assert_eq!(net.output, simplified.reduced.output);
        let dimensions: HashMap<_, _> = inputs
            .iter()
            .flatten()
            .chain(net.output.iter())
            .map(|&leg| (leg, net.size_dict[&leg]))
            .collect();
        assert_eq!(dimensions, simplified.reduced.size_dict);
    }

    fn complete_path(net: &TensorNetwork, simplified: &Simplified) -> SsaPath {
        let n = simplified.map.len();
        let mut reduced = Vec::new();
        if n > 1 {
            let mut root = 0;
            for other in 1..n {
                reduced.push((root, other));
                root = n + reduced.len() - 1;
            }
        }
        stitch(
            net.n_tensors(),
            &simplified.prefix,
            &simplified.map,
            &reduced,
        )
    }

    fn compare_with_scan(net: &TensorNetwork) {
        let expected = scan_reference::simplify(net);
        let actual = simplify(net);
        assert_same(&actual, &expected);
        check_prefix(net, &actual);
        let path = complete_path(net, &actual);
        assert_eq!(path, complete_path(net, &expected));
        assert_eq!(path.len(), net.n_tensors().saturating_sub(1));
        simulate_path(net, &path).unwrap();
    }

    #[test]
    fn queue_matches_scan_for_scalar_open_diagonal_and_disconnected_networks() {
        let scalar_tie = network(
            "scalar-tie",
            vec![vec![], vec![0, 1, 2], vec![3], vec![4]],
            vec![4, 3],
        );
        let simplified = simplify(&scalar_tie);
        assert_eq!(simplified.prefix, vec![(0, 2)]);
        assert_eq!(simplified.map, vec![1, 3, 4]);
        assert_eq!(
            simplified.reduced.inputs,
            vec![vec![0, 1, 2], vec![4], vec![3]]
        );
        let fixtures = vec![
            scalar_tie,
            network("scalars", vec![vec![], vec![], vec![]], vec![]),
            network(
                "hyperedge-open",
                vec![vec![0], vec![0, 1, 2], vec![0, 3, 4]],
                vec![2, 0],
            ),
            network(
                "repeated-axes",
                vec![vec![0, 0, 1], vec![1, 2], vec![2, 3, 3]],
                vec![3, 0],
            ),
            network(
                "disconnected",
                vec![vec![0], vec![0, 1], vec![1], vec![2, 3], vec![3], vec![4]],
                vec![4, 2],
            ),
            network(
                "no-simplification",
                vec![vec![0, 1, 2], vec![0, 3, 4], vec![1, 3, 5], vec![2, 4, 5]],
                vec![],
            ),
            network(
                "isolated-output-vector",
                vec![vec![0], vec![1, 2, 3]],
                vec![0],
            ),
            network("one-diagonal", vec![vec![0, 0]], vec![0]),
        ];
        for net in fixtures {
            compare_with_scan(&net);
        }
    }

    #[test]
    fn queue_matches_scan_on_fixed_generated_valid_networks() {
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            state ^= state << 7;
            state ^= state >> 9;
            state ^= state << 8;
            state
        };
        for case in 0..1000 {
            let n_tensors = 1 + next() as usize % 12;
            let n_legs = 1 + next() as usize % 10;
            let mut inputs = Vec::new();
            for _ in 0..n_tensors {
                let rank = next() as usize % 6;
                let mut legs = Vec::new();
                for _ in 0..rank {
                    let leg = (next() as usize % n_legs) as LegId;
                    if legs.iter().filter(|&&known| known == leg).count() < 2 {
                        legs.push(leg);
                    }
                }
                inputs.push(legs);
            }
            let held: BTreeSet<_> = inputs.iter().flatten().copied().collect();
            let output = held.into_iter().rev().filter(|_| next() % 4 == 0).collect();
            compare_with_scan(&network(&format!("generated-{case}"), inputs, output));
        }
    }

    // The broad rule used before narrowing: every low-rank tensor within two
    // hops of a changed leg. Kept test-only as an upper bound for the queue.
    fn requeue_two_hops(state: &State, effect: &ContractEffect, pending: &mut BTreeSet<usize>) {
        let mut affected = HashSet::from([effect.new_id]);
        for changed in &effect.changed {
            if let Some(ids) = state.holders.get(&changed.leg) {
                affected.extend(ids.iter().copied());
            }
        }
        for holder in affected {
            if state.rank(holder) <= 2 {
                pending.insert(holder);
            }
            for leg in state.legs[holder].as_ref().unwrap() {
                if let Some(ids) = state.holders.get(leg) {
                    pending.extend(ids.iter().copied().filter(|&id| state.rank(id) <= 2));
                }
            }
        }
    }

    /// Live tensors that would contract if examined now.
    fn ready(state: &State) -> Vec<usize> {
        (0..state.legs.len())
            .filter(|&id| state.decide(id).is_some())
            .collect()
    }

    /// Applies arbitrary contractions and checks that the narrowed queue keeps
    /// every ready tensor pending while queueing no more than the two-hop rule.
    fn check_requeue_invariant(
        net: &TensorNetwork,
        steps: usize,
        mut pick: impl FnMut(usize) -> usize,
    ) {
        let mut state = State::new(net);
        let mut pending: BTreeSet<usize> = ready(&state).into_iter().collect();
        for _ in 0..steps {
            let alive: Vec<_> = (0..state.legs.len())
                .filter(|&id| state.alive(id))
                .collect();
            if alive.len() < 2 {
                break;
            }
            let a = alive[pick(alive.len())];
            let mut b = alive[pick(alive.len())];
            if a == b {
                b = alive[(alive.iter().position(|&id| id == a).unwrap() + 1) % alive.len()];
            }
            let effect = state.contract(a, b);
            let mut upper = pending.clone();
            state.requeue_affected(&effect, &mut pending);
            requeue_two_hops(&state, &effect, &mut upper);
            pending.retain(|&id| state.alive(id));
            upper.retain(|&id| state.alive(id));
            for id in ready(&state) {
                assert!(
                    pending.contains(&id),
                    "{}: ready tensor {id} not queued",
                    net.name
                );
            }
            assert!(
                pending.is_subset(&upper),
                "{}: queue exceeds two-hop rule",
                net.name
            );
        }
    }

    fn shared_leg_network(open: bool) -> TensorNetwork {
        let mut inputs = vec![vec![0, 0]];
        for i in 0..24 {
            inputs.push(vec![0, 1 + i, 25 + i]);
            inputs.push(vec![1 + i]);
            inputs.push(vec![25 + i]);
        }
        inputs.push(vec![]);
        network("shared-leg", inputs, if open { vec![0] } else { vec![] })
    }

    /// A hub leg held by many low-rank tensors, as in large model-counting networks.
    fn hub_network(open: bool) -> TensorNetwork {
        let mut inputs = Vec::new();
        for i in 0..30 {
            match i % 3 {
                0 => inputs.push(vec![0, 1 + i]),
                1 => inputs.push(vec![0, 1 + i, 2 + i]),
                _ => inputs.push(vec![0, 1 + i, 2 + i, 3 + i]),
            }
            inputs.push(vec![1 + i]);
        }
        inputs.push(vec![0]);
        network("hub", inputs, if open { vec![0] } else { vec![] })
    }

    #[test]
    fn queue_matches_scan_on_hub_networks() {
        for open in [false, true] {
            compare_with_scan(&shared_leg_network(open));
            compare_with_scan(&hub_network(open));
        }
    }

    #[test]
    fn narrowed_requeue_keeps_every_ready_tensor_pending() {
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for open in [false, true] {
            for net in [shared_leg_network(open), hub_network(open)] {
                check_requeue_invariant(&net, 60, |n| next() as usize % n);
                check_requeue_invariant(&net, 60, |_| 0);
            }
        }
        for case in 0..400 {
            let n_tensors = 2 + next() as usize % 14;
            let n_legs = 1 + next() as usize % 8;
            let mut inputs = Vec::new();
            for _ in 0..n_tensors {
                let rank = next() as usize % 5;
                let mut legs: Vec<LegId> = Vec::new();
                for _ in 0..rank {
                    let leg = (next() as usize % n_legs) as LegId;
                    if legs.iter().filter(|&&known| known == leg).count() < 2 {
                        legs.push(leg);
                    }
                }
                inputs.push(legs);
            }
            let held: BTreeSet<_> = inputs.iter().flatten().copied().collect();
            let output = held.into_iter().filter(|_| next() % 4 == 0).collect();
            let net = network(&format!("arbitrary-{case}"), inputs, output);
            check_requeue_invariant(&net, n_tensors, |n| next() as usize % n);
        }
    }
}
