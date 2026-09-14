//! Tensor-network simplification before path search.
//!
//! The pass applies path-independent low-rank contractions:
//! - rank 0: absorb into the smallest live tensor;
//! - rank 1: absorb into a tensor sharing its only leg;
//! - rank 2: absorb when the neighbor's rank does not increase.
//!
//! Simplification emits valid pairwise contractions as an SSA prefix. [`stitch`]
//! maps a path on the reduced network back to the original network.

use std::collections::{HashMap, HashSet};

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
