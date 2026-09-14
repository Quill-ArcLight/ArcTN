//! Multilevel hypergraph-bisection pathfinder.
//!
//! Minimizes the cut-net objective weighted by `log2(dim)`. Heavy-edge matching
//! coarsens the graph, and Fiduccia-Mattheyses passes refine each uncoarsened level.

use std::collections::{HashMap, HashSet};

use rand::seq::SliceRandom;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::network::{LegId, TensorNetwork};
use crate::objective::PlannerObjective;
use crate::path::{simulate_path, sorted_dedup, PathStats, SsaPath};
use crate::paths::optimal::optimal_dp;

/// Local hypergraph for one recursive subproblem.
struct HG {
    n: usize,
    /// Node weights used for balance constraints.
    node_w: Vec<usize>,
    /// Incident hyperedges for each node.
    legs_of: Vec<Vec<usize>>,
    /// Pins for each hyperedge.
    pins: Vec<Vec<usize>>,
    /// Hyperedge weight `log2(dimension)`.
    w: Vec<f64>,
}

impl HG {
    fn total_w(&self) -> usize {
        self.node_w.iter().sum()
    }
}

/// Build a local hypergraph from legs held by at least two members.
fn build_hg_with(
    members: &[usize],
    input_legs: &[Vec<LegId>],
    log2_dim: &dyn Fn(LegId) -> f64,
) -> HG {
    let n = members.len();
    let mut leg_pins: HashMap<LegId, Vec<usize>> = HashMap::new();
    for (li, &g) in members.iter().enumerate() {
        for &l in &sorted_dedup(&input_legs[g]) {
            leg_pins.entry(l).or_default().push(li);
        }
    }
    let mut legs_of = vec![Vec::new(); n];
    let mut pins = Vec::new();
    let mut w = Vec::new();
    // Sort legs for deterministic floating-point accumulation.
    let mut leg_pins: Vec<(LegId, Vec<usize>)> = leg_pins.into_iter().collect();
    leg_pins.sort_unstable_by_key(|(l, _)| *l);
    for (l, ps) in leg_pins {
        if ps.len() >= 2 {
            let e = pins.len();
            for &p in &ps {
                legs_of[p].push(e);
            }
            pins.push(ps);
            w.push(log2_dim(l));
        }
    }
    HG {
        n,
        node_w: vec![1; n],
        legs_of,
        pins,
        w,
    }
}

/// Return the cut reduction from moving one node to the opposite side.
fn gain_of(hg: &HG, side: &[bool], cnt: &[[usize; 2]], v: usize) -> f64 {
    let s = side[v] as usize;
    let o = 1 - s;
    let mut g = 0.0;
    for &l in &hg.legs_of[v] {
        if cnt[l][s] == 1 && cnt[l][o] > 0 {
            g += hg.w[l];
        } else if cnt[l][o] == 0 && cnt[l][s] > 1 {
            g -= hg.w[l];
        }
    }
    g
}

/// Run Fiduccia-Mattheyses passes and keep the best feasible move prefix.
fn fm_refine(hg: &HG, side: &mut [bool], eps: f64, passes: usize) {
    let total = hg.total_w();
    let lo = (((0.5 - eps) * total as f64).ceil() as usize).max(1);
    let hi = total - lo;
    for _ in 0..passes {
        // Pin counts on each side of every hyperedge.
        let mut cnt = vec![[0usize; 2]; hg.pins.len()];
        for v in 0..hg.n {
            for &l in &hg.legs_of[v] {
                cnt[l][side[v] as usize] += 1;
            }
        }
        let mut gain: Vec<f64> = (0..hg.n).map(|v| gain_of(hg, side, &cnt, v)).collect();
        let mut w_a: usize = (0..hg.n).filter(|&v| !side[v]).map(|v| hg.node_w[v]).sum();
        let mut locked = vec![false; hg.n];
        let mut moves: Vec<usize> = Vec::new();
        let start_feasible = w_a >= lo && w_a <= hi;
        let mut cum = 0.0f64;
        let mut best_cum = if start_feasible {
            0.0f64
        } else {
            f64::NEG_INFINITY
        };
        let mut best_k = 0usize;
        loop {
            // Select the highest-gain unlocked move that preserves or improves balance.
            let half = hg.total_w() as f64 / 2.0;
            let cur_dev = (w_a as f64 - half).abs();
            let mut best: Option<(f64, usize)> = None;
            for v in 0..hg.n {
                if locked[v] {
                    continue;
                }
                let nwa = if !side[v] {
                    w_a - hg.node_w[v]
                } else {
                    w_a + hg.node_w[v]
                };
                let in_range = nwa >= lo && nwa <= hi;
                let improves_balance = (nwa as f64 - half).abs() < cur_dev - 1e-9;
                if !in_range && !improves_balance {
                    continue;
                }
                if best.map(|(g, _)| gain[v] > g).unwrap_or(true) {
                    best = Some((gain[v], v));
                }
            }
            let Some((g, v)) = best else { break };
            locked[v] = true;
            moves.push(v);
            cum += g;
            let s = side[v] as usize;
            let o = 1 - s;
            if !side[v] {
                w_a -= hg.node_w[v];
            } else {
                w_a += hg.node_w[v];
            }
            side[v] = !side[v];
            for &l in &hg.legs_of[v] {
                cnt[l][s] -= 1;
                cnt[l][o] += 1;
            }
            // Recompute gains for unlocked neighbors on affected hyperedges.
            for &l in &hg.legs_of[v] {
                for &u in &hg.pins[l] {
                    if !locked[u] {
                        gain[u] = gain_of(hg, side, &cnt, u);
                    }
                }
            }
            // Only feasible prefixes can be committed. An infeasible start must
            // first return to the balance range.
            let feasible = w_a >= lo && w_a <= hi;
            if feasible && cum > best_cum + 1e-12 {
                best_cum = cum;
                best_k = moves.len();
            }
        }
        // Roll back all moves if an infeasible start never regains balance.
        if best_cum == f64::NEG_INFINITY {
            best_k = 0;
        }
        // Roll back moves after the best prefix.
        for &v in &moves[best_k..] {
            side[v] = !side[v];
        }
        if best_k == 0 {
            break;
        }
    }
}

/// Coarsen by heavy-edge matching with a 40% aggregate node-weight cap.
/// Returns the coarse graph and fine-to-coarse mapping.
fn coarsen<R: Rng>(hg: &HG, rng: &mut R) -> (HG, Vec<usize>) {
    let n = hg.n;
    let cap = ((hg.total_w() as f64) * 0.4).max(1.0) as usize;
    let mut matched = vec![usize::MAX; n];
    let mut order: Vec<usize> = (0..n).collect();
    order.shuffle(rng);
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for &v in &order {
        if matched[v] != usize::MAX {
            continue;
        }
        // Distribute each hyperedge weight across its other pins.
        let mut score: HashMap<usize, f64> = HashMap::new();
        for &l in &hg.legs_of[v] {
            let pl = hg.pins[l].len();
            for &u in &hg.pins[l] {
                if u != v && matched[u] == usize::MAX {
                    *score.entry(u).or_insert(0.0) += hg.w[l] / (pl - 1) as f64;
                }
            }
        }
        let best = score
            .into_iter()
            .filter(|&(u, _)| hg.node_w[v] + hg.node_w[u] <= cap)
            .max_by(|a, b| a.1.total_cmp(&b.1).then(b.0.cmp(&a.0)));
        let gid = groups.len();
        match best {
            Some((u, _)) => {
                matched[v] = gid;
                matched[u] = gid;
                groups.push(vec![v, u]);
            }
            None => {
                matched[v] = gid;
                groups.push(vec![v]);
            }
        }
    }
    // Map pins to groups and discard internalized one-pin hyperedges.
    let cn = groups.len();
    let mut node_w = vec![0usize; cn];
    for (g, mem) in groups.iter().enumerate() {
        node_w[g] = mem.iter().map(|&v| hg.node_w[v]).sum();
    }
    let mut legs_of = vec![Vec::new(); cn];
    let mut pins = Vec::new();
    let mut w = Vec::new();
    for (l, ps) in hg.pins.iter().enumerate() {
        let mut cp: Vec<usize> = ps.iter().map(|&v| matched[v]).collect();
        cp.sort_unstable();
        cp.dedup();
        if cp.len() >= 2 {
            let e = pins.len();
            for &p in &cp {
                legs_of[p].push(e);
            }
            pins.push(cp);
            w.push(hg.w[l]);
        }
    }
    (
        HG {
            n: cn,
            node_w,
            legs_of,
            pins,
            w,
        },
        matched,
    )
}

/// Build a coarse initial partition by seeded growth followed by FM refinement.
fn initial_partition<R: Rng>(hg: &HG, eps: f64, rng: &mut R) -> Vec<bool> {
    let n = hg.n;
    let total = hg.total_w();
    let target = total / 2;
    let mut side = vec![true; n]; // true = B
    let mut conn = vec![0.0f64; n];
    let start = rng.gen_range(0..n);
    side[start] = false;
    let mut w_a = hg.node_w[start];
    let bump = |conn: &mut Vec<f64>, hg: &HG, v: usize| {
        for &l in &hg.legs_of[v] {
            for &u in &hg.pins[l] {
                if u != v {
                    conn[u] += hg.w[l] / (hg.pins[l].len() - 1) as f64;
                }
            }
        }
    };
    bump(&mut conn, hg, start);
    // Prefer the strongest connected node that stays within the balance limit.
    let hi = ((0.5 + eps) * total as f64).floor() as usize;
    while w_a < target {
        let mut best: Option<(f64, usize)> = None;
        let mut best_any: Option<(f64, usize)> = None;
        for v in 0..n {
            if !side[v] {
                continue;
            }
            if best_any.map(|(c, _)| conn[v] > c).unwrap_or(true) {
                best_any = Some((conn[v], v));
            }
            if w_a + hg.node_w[v] <= hi && best.map(|(c, _)| conn[v] > c).unwrap_or(true) {
                best = Some((conn[v], v));
            }
        }
        let Some((_, v)) = best.or(best_any) else {
            break;
        };
        if w_a + hg.node_w[v] > hi && w_a >= total / 4 {
            break;
        }
        side[v] = false;
        w_a += hg.node_w[v];
        bump(&mut conn, hg, v);
    }
    fm_refine(hg, &mut side, eps, 2);
    side
}

/// Coarsen to at most 24 nodes, partition, then uncoarsen with FM refinement.
fn multilevel_bipartition<R: Rng>(hg0: HG, eps: f64, rng: &mut R) -> Vec<bool> {
    let mut levels: Vec<HG> = vec![hg0];
    let mut maps: Vec<Vec<usize>> = Vec::new();
    while levels.last().unwrap().n > 24 {
        let (c, map) = coarsen(levels.last().unwrap(), rng);
        if (c.n as f64) > levels.last().unwrap().n as f64 * 0.95 {
            break;
        }
        maps.push(map);
        levels.push(c);
    }
    let mut side = initial_partition(levels.last().unwrap(), eps, rng);
    // Project coarse assignments to each finer level before refinement.
    for i in (0..maps.len()).rev() {
        let fine = &levels[i];
        let map = &maps[i];
        let mut fine_side = vec![false; fine.n];
        for v in 0..fine.n {
            fine_side[v] = side[map[v]];
        }
        fm_refine(fine, &mut fine_side, eps, 2);
        side = fine_side;
    }
    side
}

/// Bipartition an explicit hypergraph using the same multilevel pipeline.
/// Returns one side flag per node.
pub fn bipartition_hypergraph(
    node_w: Vec<usize>,
    pins: Vec<Vec<usize>>,
    w: Vec<f64>,
    eps: f64,
    seed: u64,
) -> Vec<bool> {
    let n = node_w.len();
    let mut legs_of = vec![Vec::new(); n];
    for (e, ps) in pins.iter().enumerate() {
        for &p in ps {
            legs_of[p].push(e);
        }
    }
    let hg = HG {
        n,
        node_w,
        legs_of,
        pins,
        w,
    };
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    multilevel_bipartition(hg, eps, &mut rng)
}

/// Return legs that remain free after contracting a member set.
fn free_legs(
    members: &[usize],
    input_legs: &[Vec<LegId>],
    holder_count: &HashMap<LegId, usize>,
    in_output: &std::collections::HashSet<LegId>,
) -> Vec<LegId> {
    let mut inner: HashMap<LegId, usize> = HashMap::new();
    for &m in members {
        for &l in &sorted_dedup(&input_legs[m]) {
            *inner.entry(l).or_insert(0) += 1;
        }
    }
    let mut legs: Vec<LegId> = inner
        .iter()
        .filter(|(l, &c)| holder_count[l] > c || in_output.contains(l))
        .map(|(&l, _)| l)
        .collect();
    legs.sort_unstable();
    legs
}

/// Split members into connected components by shared legs.
fn components(members: &[usize], input_legs: &[Vec<LegId>]) -> Vec<Vec<usize>> {
    let n = members.len();
    let mut leg_holders: HashMap<LegId, Vec<usize>> = HashMap::new();
    for (li, &g) in members.iter().enumerate() {
        for &l in &sorted_dedup(&input_legs[g]) {
            leg_holders.entry(l).or_default().push(li);
        }
    }
    let mut comp = vec![usize::MAX; n];
    let mut nc = 0;
    for s in 0..n {
        if comp[s] != usize::MAX {
            continue;
        }
        let mut stack = vec![s];
        comp[s] = nc;
        while let Some(u) = stack.pop() {
            for &l in &sorted_dedup(&input_legs[members[u]]) {
                for &v in &leg_holders[&l] {
                    if comp[v] == usize::MAX {
                        comp[v] = nc;
                        stack.push(v);
                    }
                }
            }
        }
        nc += 1;
    }
    let mut out = vec![Vec::new(); nc];
    for (li, &c) in comp.iter().enumerate() {
        out[c].push(members[li]);
    }
    out
}

struct Ctx<'a> {
    net: &'a TensorNetwork,
    input_legs: Vec<Vec<LegId>>,
    holder_count: HashMap<LegId, usize>,
    in_output: std::collections::HashSet<LegId>,
    cutoff: usize,
    eps: f64,
}

/// Solve one member set recursively and return its SSA root and free legs.
fn solve<R: Rng>(
    ctx: &Ctx,
    members: &[usize],
    path: &mut SsaPath,
    rng: &mut R,
) -> Result<(usize, Vec<LegId>), String> {
    let n = ctx.net.n_tensors();
    if members.len() == 1 {
        return Ok((members[0], sorted_dedup(&ctx.input_legs[members[0]])));
    }
    // Solve connected components independently, then outer-product smallest-first.
    let comps = components(members, &ctx.input_legs);
    if comps.len() > 1 {
        let mut roots: Vec<(usize, Vec<LegId>, f64)> = Vec::new();
        for c in &comps {
            let (id, legs) = solve(ctx, c, path, rng)?;
            let sz: f64 = legs.iter().map(|&l| ctx.net.log2_dim(l)).sum();
            roots.push((id, legs, sz));
        }
        roots.sort_by(|a, b| a.2.total_cmp(&b.2));
        while roots.len() > 1 {
            let (ia, la, _) = roots.remove(0);
            let (ib, lb, _) = roots.remove(0);
            let (x, y) = if ia < ib { (ia, ib) } else { (ib, ia) };
            path.push((x, y));
            let id = n + path.len() - 1;
            let legs = crate::path::legs_union(&la, &lb);
            let sz: f64 = legs.iter().map(|&l| ctx.net.log2_dim(l)).sum();
            roots.push((id, legs, sz));
            roots.sort_by(|a, b| a.2.total_cmp(&b.2));
        }
        let (id, legs, _) = roots.pop().unwrap();
        return Ok((id, legs));
    }
    let out_legs = free_legs(members, &ctx.input_legs, &ctx.holder_count, &ctx.in_output);
    // Use exact bitmask DP below the cutoff.
    if members.len() <= ctx.cutoff {
        let mut size_dict = HashMap::new();
        for &m in members {
            for &l in &ctx.input_legs[m] {
                size_dict.insert(l, ctx.net.dim(l));
            }
        }
        let mini = TensorNetwork {
            name: String::new(),
            inputs: members.iter().map(|&m| ctx.input_legs[m].clone()).collect(),
            output: out_legs.clone(),
            size_dict,
        };
        let (mini_path, _) = optimal_dp(&mini, ctx.cutoff.max(16))?;
        let mut map: Vec<usize> = members.to_vec();
        for &(a, b) in &mini_path {
            let (ga, gb) = (map[a], map[b]);
            let (x, y) = if ga < gb { (ga, gb) } else { (gb, ga) };
            path.push((x, y));
            map.push(n + path.len() - 1);
        }
        return Ok((*map.last().unwrap(), out_legs));
    }
    // Recurse through a multilevel hypergraph bisection.
    let hg = build_hg_with(members, &ctx.input_legs, &|l| ctx.net.log2_dim(l));
    let side = multilevel_bipartition(hg, ctx.eps, rng);
    let mut a: Vec<usize> = Vec::new();
    let mut b: Vec<usize> = Vec::new();
    for (li, &g) in members.iter().enumerate() {
        if side[li] {
            b.push(g);
        } else {
            a.push(g);
        }
    }
    // Fall back to an even split if the partitioner returns an empty side.
    if a.is_empty() || b.is_empty() {
        let half = members.len() / 2;
        a = members[..half].to_vec();
        b = members[half..].to_vec();
    }
    let (ia, _) = solve(ctx, &a, path, rng)?;
    let (ib, _) = solve(ctx, &b, path, rng)?;
    let (x, y) = if ia < ib { (ia, ib) } else { (ib, ia) };
    path.push((x, y));
    Ok((n + path.len() - 1, out_legs))
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

/// Path and metrics from one recursive-bisection trial.
#[derive(Clone, Debug)]
pub(crate) struct BisectTrialRecord {
    pub(crate) path: SsaPath,
    pub(crate) stats: PathStats,
}

/// Read-only preprocessing shared across bisection trials.
#[derive(Clone, Debug)]
pub(crate) struct BisectPrepared {
    input_legs: Vec<Vec<LegId>>,
    holder_count: HashMap<LegId, usize>,
    in_output: HashSet<LegId>,
    all: Vec<usize>,
    cutoff: usize,
}

const EPSILON_MIN: f64 = 0.02;
const EPSILON_MAX: f64 = 0.35;

fn sample_epsilon<R: Rng>(rng: &mut R) -> f64 {
    let lo = EPSILON_MIN.ln();
    let hi = EPSILON_MAX.ln();
    rng.gen_range(lo..hi).exp()
}

/// Prepare structural data shared across bisection trials.
pub(crate) fn prepare_bisect(net: &TensorNetwork, cutoff: usize) -> Result<BisectPrepared, String> {
    net.validate()?;
    let input_legs: Vec<Vec<LegId>> = net.inputs.iter().map(|t| sorted_dedup(t)).collect();
    let mut holder_count: HashMap<LegId, usize> = HashMap::new();
    for legs in &input_legs {
        for &leg in legs {
            *holder_count.entry(leg).or_insert(0) += 1;
        }
    }
    let in_output = net.output.iter().copied().collect();
    Ok(BisectPrepared {
        input_legs,
        holder_count,
        in_output,
        all: (0..net.n_tensors()).collect(),
        cutoff: cutoff.clamp(2, 20),
    })
}

fn run_bisect_trial_with_rng<R: Rng>(
    net: &TensorNetwork,
    prepared: &BisectPrepared,
    epsilon: f64,
    rng: &mut R,
) -> Result<Option<BisectTrialRecord>, String> {
    let ctx = Ctx {
        net,
        input_legs: prepared.input_legs.clone(),
        holder_count: prepared.holder_count.clone(),
        in_output: prepared.in_output.clone(),
        cutoff: prepared.cutoff,
        eps: epsilon,
    };
    let mut path = SsaPath::new();
    if solve(&ctx, &prepared.all, &mut path, rng).is_err() {
        return Ok(None);
    }
    let stats = simulate_path(net, &path)?;
    Ok(Some(BisectTrialRecord { path, stats }))
}

/// Run parallel recursive-bisection trials and return the best result.
pub fn bisect(
    net: &TensorNetwork,
    ntrials: usize,
    seed: u64,
    cutoff: usize,
) -> Result<(SsaPath, PathStats), String> {
    bisect_with_objective(net, ntrials, seed, cutoff, PlannerObjective::FIXED)
}

/// [`bisect`] with a per-call planner objective for selecting the best trial.
pub fn bisect_with_objective(
    net: &TensorNetwork,
    ntrials: usize,
    seed: u64,
    cutoff: usize,
    objective: PlannerObjective,
) -> Result<(SsaPath, PathStats), String> {
    bisect_impl(net, ntrials, seed, cutoff, None, objective)
}

/// Run recursive-bisection trials with an explicit objective and deadline.
///
/// Returns the best completed path, its metrics, and the number of completed
/// trials. Returns an error for invalid input or if no trial completes. A
/// trial already in progress may finish after the cooperative deadline.
pub fn bisect_until_with_objective(
    net: &TensorNetwork,
    ntrials: usize,
    seed: u64,
    cutoff: usize,
    deadline: std::time::Instant,
    objective: PlannerObjective,
) -> Result<((SsaPath, PathStats), usize), String> {
    let candidates = bisect_candidates(net, ntrials, seed, cutoff, Some(deadline))?;
    let completed = candidates.len();
    candidates
        .into_iter()
        .min_by(|a, b| compare_trial_stats(objective, &a.1, &b.1))
        .map(|candidate| (candidate, completed))
        .ok_or_else(|| "截止前没有完成 bisect trial".to_string())
}

/// Run a reproducible batch of recursive-bisection trials.
#[allow(clippy::too_many_arguments)]
pub(crate) fn bisect_trial_records(
    net: &TensorNetwork,
    ntrials: usize,
    trial_offset: usize,
    seed: u64,
    cutoff: usize,
    deadline: Option<std::time::Instant>,
) -> Result<Vec<BisectTrialRecord>, String> {
    let prepared = prepare_bisect(net, cutoff)?;
    use rayon::prelude::*;
    (0..ntrials)
        .into_par_iter()
        .filter_map(|slot| {
            if deadline.is_some_and(|value| std::time::Instant::now() >= value) {
                return None;
            }
            let trial = trial_offset.saturating_add(slot);
            let mut rng = ChaCha8Rng::seed_from_u64(seed.wrapping_add(trial as u64));
            let epsilon = sample_epsilon(&mut rng);
            Some(run_bisect_trial_with_rng(net, &prepared, epsilon, &mut rng))
        })
        .collect::<Result<Vec<_>, String>>()
        .map(|records| records.into_iter().flatten().collect())
}

fn bisect_impl(
    net: &TensorNetwork,
    ntrials: usize,
    seed: u64,
    cutoff: usize,
    deadline: Option<std::time::Instant>,
    objective: PlannerObjective,
) -> Result<(SsaPath, PathStats), String> {
    bisect_candidates(net, ntrials, seed, cutoff, deadline)?
        .into_iter()
        .min_by(|a, b| compare_trial_stats(objective, &a.1, &b.1))
        .ok_or_else(|| "所有 bisect trial 都失败".to_string())
}

fn bisect_candidates(
    net: &TensorNetwork,
    ntrials: usize,
    seed: u64,
    cutoff: usize,
    deadline: Option<std::time::Instant>,
) -> Result<Vec<(SsaPath, PathStats)>, String> {
    Ok(
        bisect_trial_records(net, ntrials.max(1), 0, seed, cutoff, deadline)?
            .into_iter()
            .map(|record| (record.path, record.stats))
            .collect(),
    )
}

#[cfg(test)]
mod objective_tests {
    use super::*;

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
    fn expired_deadline_starts_no_bisect_trial() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x4445_4144_4c49_4e45);
        let net = TensorNetwork::random_connected(8, 3, &[2, 3, 4], 1, &mut rng);
        let result = bisect_until_with_objective(
            &net,
            8,
            17,
            3,
            std::time::Instant::now() - std::time::Duration::from_millis(1),
            PlannerObjective::FIXED,
        );
        assert!(result.is_err());
    }

    #[test]
    fn fixed_objective_wrapper_matches_explicit_entry() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x4f42_4a45_4354_4956);
        let net = TensorNetwork::random_connected(8, 3, &[2, 3, 4], 1, &mut rng);
        let expected = bisect(&net, 8, 17, 3).unwrap();
        let actual = bisect_with_objective(&net, 8, 17, 3, PlannerObjective::FIXED).unwrap();
        assert_eq!(actual.0, expected.0);
        assert_eq!(
            actual.1.log10_flops.to_bits(),
            expected.1.log10_flops.to_bits()
        );
        assert_eq!(
            actual.1.log2_read_write.to_bits(),
            expected.1.log2_read_write.to_bits()
        );
    }

    #[test]
    fn completed_trial_replay_errors_are_propagated() {
        let prepared_net = TensorNetwork {
            name: "bisect-prepared".into(),
            inputs: vec![Vec::new(); 4],
            output: Vec::new(),
            size_dict: HashMap::new(),
        };
        let replay_net = TensorNetwork {
            name: "bisect-replay".into(),
            inputs: vec![Vec::new(); 3],
            output: Vec::new(),
            size_dict: HashMap::new(),
        };
        let prepared = prepare_bisect(&prepared_net, 2).unwrap();

        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let error = run_bisect_trial_with_rng(&replay_net, &prepared, 0.1, &mut rng).unwrap_err();

        assert!(!error.is_empty());
    }
}
