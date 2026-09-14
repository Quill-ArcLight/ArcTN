//! Heap-based greedy and temperature-weighted random-greedy pathfinding.
//!
//! Candidate cost is `size(result) - costmod * (size(a) + size(b))`.
//! Random mode samples the first `nbranch` candidates with Boltzmann weights.

use std::collections::{BinaryHeap, HashMap, HashSet};

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::network::{LegId, TensorNetwork};
use crate::objective::PlannerObjective;
use crate::path::{simulate_path, sorted_dedup, PathStats, SsaPath};

/// Greedy cost-function variants, partly following Orgler and Blacher (arXiv:2405.09644).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CostFn {
    /// Base expression: `size12 - costmod * (s1 + s2)`.
    MemRemoved,
    /// Add `sk * |s1 - s2|`; positive values favor balanced inputs.
    Skew(f64),
    /// Normalize the base expression by `size12 + 1`.
    Ratio,
    /// Log ratio: `log2(size12 + 2) / log2(0.65 * (s1 + s2) + 2)`.
    LogRatio,
    /// Add `hd * sum(refcount(leg))`; negative values favor high-degree legs.
    HyperDeg(f64),
}

#[derive(Clone, Copy, Debug)]
struct Cand {
    cost: f64,
    a: usize,
    b: usize,
}

// Candidate costs are never NaN.
impl PartialEq for Cand {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost && self.a == other.a && self.b == other.b
    }
}
impl Eq for Cand {}
// Reverse ordering implements a min-heap with `(a, b)` tie-breaking.
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

struct GreedyState<'a> {
    net: &'a TensorNetwork,
    node_legs: Vec<Option<Vec<LegId>>>,
    node_size: Vec<f64>,
    refcount: HashMap<LegId, usize>,
    in_output: HashSet<LegId>,
    holders: HashMap<LegId, HashSet<usize>>,
    heap: BinaryHeap<Cand>,
    costmod: f64,
    cost_fn: CostFn,
}

impl<'a> GreedyState<'a> {
    fn new(net: &'a TensorNetwork, costmod: f64, cost_fn: CostFn) -> Self {
        let in_output: HashSet<LegId> = net.output.iter().copied().collect();
        let mut refcount: HashMap<LegId, usize> = HashMap::new();
        let mut holders: HashMap<LegId, HashSet<usize>> = HashMap::new();
        let mut node_legs = Vec::new();
        let mut node_size = Vec::new();
        for (i, t) in net.inputs.iter().enumerate() {
            let d = sorted_dedup(t);
            for &l in &d {
                *refcount.entry(l).or_insert(0) += 1;
                holders.entry(l).or_default().insert(i);
            }
            // Sum dimensions in log2 space before converting to a linear size.
            let sz: f64 = d.iter().map(|&l| net.log2_dim(l)).sum::<f64>().exp2();
            node_legs.push(Some(d));
            node_size.push(sz);
        }
        let mut st = GreedyState {
            net,
            node_legs,
            node_size,
            refcount,
            in_output,
            holders,
            heap: BinaryHeap::new(),
            costmod,
            cost_fn,
        };
        // Initial candidates are input pairs sharing at least one leg.
        let mut seen: HashSet<(usize, usize)> = HashSet::new();
        for hs in st.holders.clone().values() {
            let mut v: Vec<usize> = hs.iter().copied().collect();
            v.sort_unstable();
            for i in 0..v.len() {
                for j in (i + 1)..v.len() {
                    if seen.insert((v[i], v[j])) {
                        st.push_cand(v[i], v[j]);
                    }
                }
            }
        }
        st
    }

    /// Return the retained legs after contracting `a` and `b`.
    fn result_legs(&self, a: usize, b: usize) -> Vec<LegId> {
        crate::path::result_legs(
            self.node_legs[a].as_ref().unwrap(),
            self.node_legs[b].as_ref().unwrap(),
            &self.refcount,
            &self.in_output,
        )
    }

    /// Return the linear tensor size for a set of legs.
    fn legs_size(&self, legs: &[LegId]) -> f64 {
        legs.iter()
            .map(|&l| self.net.log2_dim(l))
            .sum::<f64>()
            .exp2()
    }

    /// Score a candidate and push it into the heap.
    fn push_cand(&mut self, a: usize, b: usize) {
        let result = self.result_legs(a, b);
        let rsize = self.legs_size(&result);
        let (sa, sb) = (self.node_size[a], self.node_size[b]);
        let base = rsize - self.costmod * (sa + sb);
        let cost = match self.cost_fn {
            CostFn::MemRemoved => base,
            CostFn::Skew(sk) => base + sk * (sa - sb).abs(),
            CostFn::Ratio => base / (rsize + 1.0),
            CostFn::LogRatio => (rsize + 2.0).log2() / (0.65 * (sa + sb) + 2.0).log2(),
            CostFn::HyperDeg(hd) => {
                let deg: usize = result.iter().map(|l| self.refcount[l]).sum();
                base + hd * deg as f64
            }
        };
        self.heap.push(Cand { cost, a, b });
    }

    /// Return whether an SSA node remains live.
    fn alive(&self, i: usize) -> bool {
        self.node_legs.get(i).map(|x| x.is_some()).unwrap_or(false)
    }

    /// Contract two nodes and return the new SSA id.
    fn contract(&mut self, a: usize, b: usize) -> usize {
        let result = self.result_legs(a, b);
        let la = self.node_legs[a].take().unwrap();
        let lb = self.node_legs[b].take().unwrap();
        for &l in &la {
            *self.refcount.get_mut(&l).unwrap() -= 1;
            let hs = self.holders.get_mut(&l).unwrap();
            hs.remove(&a);
        }
        for &l in &lb {
            *self.refcount.get_mut(&l).unwrap() -= 1;
            let hs = self.holders.get_mut(&l).unwrap();
            hs.remove(&b);
        }
        let new_id = self.node_legs.len();
        for &l in &result {
            *self.refcount.get_mut(&l).unwrap() += 1;
            self.holders.get_mut(&l).unwrap().insert(new_id);
        }
        let sz = self.legs_size(&result);
        self.node_legs.push(Some(result));
        self.node_size.push(sz);
        // Add candidates only between the new node and its neighbors.
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
            self.push_cand(x, y);
        }
        new_id
    }

    /// Choose the minimum candidate or sample with Boltzmann weights.
    fn choose<R: Rng>(
        &mut self,
        temperature: f64,
        nbranch: usize,
        rng: &mut R,
    ) -> Option<(usize, usize)> {
        // Lazily discard dead candidates while collecting the first branch window.
        let mut choices: Vec<Cand> = Vec::new();
        while choices.len() < nbranch.max(1) {
            match self.heap.pop() {
                None => break,
                Some(c) => {
                    if self.alive(c.a) && self.alive(c.b) {
                        choices.push(c);
                    }
                }
            }
        }
        if choices.is_empty() {
            return None;
        }
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
            // Fall back to the minimum if non-finite costs invalidate the weights.
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
        // Return unselected candidates to the heap.
        for other in choices {
            self.heap.push(other);
        }
        Some((c.a, c.b))
    }
}

/// Run one greedy trial, optionally with temperature sampling.
///
/// The caller must validate the network first.
pub fn ssa_greedy<R: Rng>(
    net: &TensorNetwork,
    costmod: f64,
    temperature: f64,
    nbranch: usize,
    rng: &mut R,
) -> SsaPath {
    ssa_greedy_v(net, costmod, temperature, nbranch, CostFn::MemRemoved, rng)
}

/// Run one greedy trial with an explicit cost-function variant.
///
/// The caller must validate the network first.
pub fn ssa_greedy_v<R: Rng>(
    net: &TensorNetwork,
    costmod: f64,
    temperature: f64,
    nbranch: usize,
    cost_fn: CostFn,
    rng: &mut R,
) -> SsaPath {
    let n = net.n_tensors();
    let mut st = GreedyState::new(net, costmod, cost_fn);
    let mut path: SsaPath = Vec::with_capacity(n.saturating_sub(1));
    while let Some((a, b)) = st.choose(temperature, nbranch, rng) {
        st.contract(a, b);
        path.push((a, b));
    }
    // Finish disconnected components with smallest-first outer products.
    let mut rest: Vec<usize> = (0..st.node_legs.len()).filter(|&i| st.alive(i)).collect();
    while rest.len() > 1 {
        rest.sort_by(|&x, &y| st.node_size[x].total_cmp(&st.node_size[y]));
        let (a, b) = (rest[0], rest[1]);
        let (a, b) = if a < b { (a, b) } else { (b, a) };
        let new_id = st.contract(a, b);
        path.push((a, b));
        rest.remove(0);
        rest.remove(0);
        rest.push(new_id);
    }
    path
}

/// Deterministic greedy kernel for a validated network.
pub(crate) fn greedy_unchecked(net: &TensorNetwork) -> (SsaPath, PathStats) {
    let mut rng = ChaCha8Rng::seed_from_u64(0);
    let path = ssa_greedy(net, 1.0, 0.0, 1, &mut rng);
    let stats = simulate_path(net, &path).expect("greedy 产生了非法路径");
    (path, stats)
}

/// Deterministic greedy search with `costmod=1`, zero temperature, and one branch.
///
/// Returns `Err` for an invalid network.
pub fn greedy(net: &TensorNetwork) -> Result<(SsaPath, PathStats), String> {
    net.validate()?;
    Ok(greedy_unchecked(net))
}

/// Run one random-greedy trial using Orgler-Blacher-inspired parameter ranges.
/// Trials 0--2 use deterministic anchors at `costmod=1,4,8`; later trials
/// sample `costmod` in `[0.1, 50]` and temperature in `[0.001, 1]` log-uniformly.
fn rgreedy_trial(net: &TensorNetwork, seed: u64, trial: usize) -> (SsaPath, PathStats) {
    // Each trial has an independent deterministic random stream.
    let mut rng = ChaCha8Rng::seed_from_u64(seed.wrapping_add(trial as u64));
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
    // Assign cost variants deterministically by trial index.
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
    let path = ssa_greedy_v(net, costmod, temperature, 8, cost_fn, &mut rng);
    let stats = simulate_path(net, &path).expect("random_greedy 产生了非法路径");
    (path, stats)
}

#[inline]
fn compare_trial_stats(
    objective: PlannerObjective,
    a: &PathStats,
    b: &PathStats,
) -> std::cmp::Ordering {
    objective
        .score_path_log2(a)
        .total_cmp(&objective.score_path_log2(b))
}

/// Run parallel random-greedy trials and return the best fixed-objective result.
pub fn random_greedy(
    net: &TensorNetwork,
    ntrials: usize,
    seed: u64,
) -> Result<(SsaPath, PathStats), String> {
    random_greedy_with_objective(net, ntrials, seed, PlannerObjective::FIXED)
}

/// [`random_greedy`] with a per-call planner objective for selecting the best trial.
pub fn random_greedy_with_objective(
    net: &TensorNetwork,
    ntrials: usize,
    seed: u64,
    objective: PlannerObjective,
) -> Result<(SsaPath, PathStats), String> {
    if ntrials == 0 {
        return Err("random_greedy requires ntrials >= 1".to_owned());
    }
    net.validate()?;
    Ok(random_greedy_unchecked_with_objective(
        net, ntrials, seed, objective,
    ))
}

/// Validated-network kernel for [`random_greedy_with_objective`].
pub(crate) fn random_greedy_unchecked_with_objective(
    net: &TensorNetwork,
    ntrials: usize,
    seed: u64,
    objective: PlannerObjective,
) -> (SsaPath, PathStats) {
    use rayon::prelude::*;
    (0..ntrials.max(1))
        .into_par_iter()
        .map(|trial| rgreedy_trial(net, seed, trial))
        .min_by(|a, b| compare_trial_stats(objective, &a.1, &b.1))
        .unwrap()
}

/// Run random-greedy trials until a cooperative deadline, selecting by `objective`.
///
/// Returns the best completed path, its metrics, and the number of completed
/// trials. Returns `None` if no trial completes. A trial already in progress
/// may finish after `deadline`; this is not a preemptive time limit.
/// A zero `ntrials` requests one trial, as in the underlying low-level kernel.
///
/// Call [`TensorNetwork::validate`] before using this low-level entry point.
/// Unlike [`random_greedy_with_objective`], it assumes a valid network and can
/// panic on malformed input.
pub fn random_greedy_until_with_objective(
    net: &TensorNetwork,
    ntrials: usize,
    seed: u64,
    deadline: std::time::Instant,
    objective: PlannerObjective,
) -> Option<((SsaPath, PathStats), usize)> {
    use rayon::prelude::*;

    let candidates: Vec<_> = (0..ntrials.max(1))
        .into_par_iter()
        .filter_map(|trial| {
            if std::time::Instant::now() >= deadline {
                None
            } else {
                Some(rgreedy_trial(net, seed, trial))
            }
        })
        .collect();
    let completed = candidates.len();
    candidates
        .into_iter()
        .min_by(|a, b| compare_trial_stats(objective, &a.1, &b.1))
        .map(|candidate| (candidate, completed))
}

#[cfg(test)]
mod deadline_tests {
    use super::*;

    fn toy() -> TensorNetwork {
        let mut size_dict = std::collections::HashMap::new();
        size_dict.insert(0, 2);
        size_dict.insert(1, 2);
        size_dict.insert(2, 2);
        TensorNetwork {
            name: "greedy-deadline".into(),
            inputs: vec![vec![0, 1], vec![1, 2]],
            output: vec![0, 2],
            size_dict,
        }
    }

    fn nontrivial_toy() -> TensorNetwork {
        let mut size_dict = std::collections::HashMap::new();
        for label in 0..6 {
            size_dict.insert(label, 2);
        }
        TensorNetwork {
            name: "greedy-live-deadline".into(),
            inputs: vec![
                vec![0, 1, 4],
                vec![1, 2],
                vec![2, 3, 5],
                vec![3, 0],
                vec![4, 5],
            ],
            output: vec![],
            size_dict,
        }
    }

    #[test]
    fn expired_random_greedy_does_not_start_a_trial() {
        let deadline = std::time::Instant::now() - std::time::Duration::from_millis(1);
        assert!(random_greedy_until_with_objective(
            &toy(),
            24,
            7,
            deadline,
            PlannerObjective::FIXED,
        )
        .is_none());
    }

    #[test]
    fn live_random_greedy_returns_a_legal_candidate() {
        let net = toy();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(1);
        let (candidate, completed) =
            random_greedy_until_with_objective(&net, 2, 7, deadline, PlannerObjective::FIXED)
                .expect("trial must fit");
        assert_eq!(completed, 2);
        let replay = simulate_path(&net, &candidate.0).expect("candidate must be legal");
        assert_eq!(
            replay.log10_flops.to_bits(),
            candidate.1.log10_flops.to_bits()
        );
    }

    #[test]
    fn generous_deadline_matches_full_random_greedy_in_one_thread() {
        let net = nontrivial_toy();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("build isolated one-thread pool");
        pool.install(|| {
            let expected = random_greedy(&net, 24, 11).unwrap();
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
            let (actual, completed) =
                random_greedy_until_with_objective(&net, 24, 11, deadline, PlannerObjective::FIXED)
                    .expect("all trials must fit");
            assert_eq!(completed, 24);
            assert_eq!(actual.0, expected.0);
            assert_eq!(
                actual.1.log10_flops.to_bits(),
                expected.1.log10_flops.to_bits()
            );
            assert_eq!(
                actual.1.log2_max_size.to_bits(),
                expected.1.log2_max_size.to_bits()
            );
            assert_eq!(
                actual.1.log2_max_contraction_size.to_bits(),
                expected.1.log2_max_contraction_size.to_bits()
            );
            assert_eq!(
                actual.1.log2_total_size.to_bits(),
                expected.1.log2_total_size.to_bits()
            );
            assert_eq!(
                actual.1.log2_read_write.to_bits(),
                expected.1.log2_read_write.to_bits()
            );
            assert_eq!(
                actual.1.log2_peak_size.to_bits(),
                expected.1.log2_peak_size.to_bits()
            );
        });
    }

    #[test]
    fn fixed_score_accounts_for_writes() {
        let low_flops_high_writes = PathStats {
            log10_flops: 10.0 * std::f64::consts::LOG10_2,
            log2_max_size: 0.0,
            log2_max_contraction_size: 0.0,
            log2_total_size: 30.0,
            log2_read_write: 30.0,
            log2_peak_size: 0.0,
        };
        let high_flops_low_writes = PathStats {
            log10_flops: 20.0 * std::f64::consts::LOG10_2,
            log2_max_size: 0.0,
            log2_max_contraction_size: 0.0,
            log2_total_size: 0.0,
            log2_read_write: 0.0,
            log2_peak_size: 0.0,
        };

        assert_eq!(
            compare_trial_stats(
                PlannerObjective::FIXED,
                &low_flops_high_writes,
                &high_flops_low_writes,
            ),
            std::cmp::Ordering::Greater
        );
    }

    #[test]
    fn fixed_objective_entry_matches_the_public_wrapper() {
        let net = nontrivial_toy();
        let expected = random_greedy(&net, 24, 11).unwrap();
        let actual = random_greedy_with_objective(&net, 24, 11, PlannerObjective::FIXED).unwrap();

        assert_eq!(actual.0, expected.0);
        assert_eq!(
            actual.1.log10_flops.to_bits(),
            expected.1.log10_flops.to_bits()
        );
        assert_eq!(
            actual.1.log2_total_size.to_bits(),
            expected.1.log2_total_size.to_bits()
        );
        assert_eq!(
            actual.1.log2_read_write.to_bits(),
            expected.1.log2_read_write.to_bits()
        );
    }

    #[test]
    fn runtime_objective_changes_trial_order() {
        let low_flops_high_writes = PathStats {
            log10_flops: 10.0 * std::f64::consts::LOG10_2,
            log2_max_size: 0.0,
            log2_max_contraction_size: 0.0,
            log2_total_size: 30.0,
            log2_read_write: 30.0,
            log2_peak_size: 0.0,
        };
        let high_flops_low_writes = PathStats {
            log10_flops: 20.0 * std::f64::consts::LOG10_2,
            log2_max_size: 0.0,
            log2_max_contraction_size: 0.0,
            log2_total_size: 0.0,
            log2_read_write: 0.0,
            log2_peak_size: 0.0,
        };
        let pure_flops = PlannerObjective::new(1.0, 0.0).unwrap();
        let pure_read_write = PlannerObjective::new(0.0, 1.0).unwrap();

        assert_eq!(
            compare_trial_stats(pure_flops, &low_flops_high_writes, &high_flops_low_writes),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            compare_trial_stats(
                pure_read_write,
                &low_flops_high_writes,
                &high_flops_low_writes,
            ),
            std::cmp::Ordering::Greater
        );
    }

    #[test]
    fn log_ratio_really_has_no_costmod_dimension() {
        let net = nontrivial_toy();
        let mut a_rng = ChaCha8Rng::seed_from_u64(91);
        let mut b_rng = ChaCha8Rng::seed_from_u64(91);
        let a = ssa_greedy_v(&net, 0.1, 0.03, 8, CostFn::LogRatio, &mut a_rng);
        let b = ssa_greedy_v(&net, 50.0, 0.03, 8, CostFn::LogRatio, &mut b_rng);
        assert_eq!(a, b);
    }
}
