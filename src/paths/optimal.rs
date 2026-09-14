//! Exact bitmask dynamic programming for contraction paths.
//!
//! Each tensor occupies one bit. Connected components are solved separately,
//! then their roots are merged by an exact outer-product DP. Cost is exponential.

use std::collections::HashMap;
use std::time::Instant;

/// Multiplicative hasher for integer-keyed tables.
/// Iteration order is controlled separately by `by_size`.
#[derive(Default)]
pub(crate) struct FastHasher(u64);
impl std::hash::Hasher for FastHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 ^ b as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        }
    }
    fn write_u64(&mut self, v: u64) {
        self.0 = v.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (v >> 32);
    }
    fn write_u32(&mut self, v: u32) {
        self.write_u64(v as u64);
    }
    fn write_usize(&mut self, v: usize) {
        self.write_u64(v as u64);
    }
}
pub(crate) type FastMap<K, V> = HashMap<K, V, std::hash::BuildHasherDefault<FastHasher>>;

/// Compact hot-path metadata indexed by leg id.
struct LegMetadata {
    index: LegIndex,
    log2_dims: Vec<f64>,
    is_output: Vec<bool>,
}

impl LegMetadata {
    fn new(net: &TensorNetwork) -> Self {
        let index = LegIndex::new(net.size_dict.keys().copied());
        let mut log2_dims = vec![0.0; index.len()];
        for (&leg, &dim) in &net.size_dict {
            log2_dims[index.slot(leg)] = (dim as f64).log2();
        }
        let mut is_output = vec![false; index.len()];
        for &leg in &net.output {
            is_output[index.slot(leg)] = true;
        }
        Self {
            index,
            log2_dims,
            is_output,
        }
    }

    #[inline]
    fn log2_dim(&self, leg: LegId) -> f64 {
        self.log2_dims[self.index.slot(leg)]
    }

    #[inline]
    fn is_output(&self, leg: LegId) -> bool {
        self.is_output[self.index.slot(leg)]
    }
}

/// Sum log2 dimensions over the union of two sorted leg lists without allocation.
/// Accumulation follows ascending leg order for bitwise reproducibility.
fn union_log2_sum(x: &[LegId], y: &[LegId], metadata: &LegMetadata) -> f64 {
    let (mut i, mut j, mut s) = (0usize, 0usize, 0f64);
    while i < x.len() && j < y.len() {
        match x[i].cmp(&y[j]) {
            std::cmp::Ordering::Less => {
                s += metadata.log2_dim(x[i]);
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                s += metadata.log2_dim(y[j]);
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                s += metadata.log2_dim(x[i]);
                i += 1;
                j += 1;
            }
        }
    }
    for &l in &x[i..] {
        s += metadata.log2_dim(l);
    }
    for &l in &y[j..] {
        s += metadata.log2_dim(l);
    }
    s
}

use crate::network::{LegId, LegIndex, TensorNetwork};
use crate::objective::{ObjectiveKind, PlannerObjective};
use crate::path::{
    legs_union, logaddexp2, simulate_path, simulate_path_flops_log2_and_legs, simulate_path_legs,
    sorted_dedup, PathStats, SsaPath,
};

/// Best contraction for one tensor subset.
#[derive(Clone, Debug)]
struct Entry {
    /// Minimum objective score for contracting this subset.
    cost: f64,
    /// Sorted free legs after contracting the subset.
    legs: Vec<LegId>,
    /// Adjacent tensors outside the subset, encoded as a bitmask.
    adj: u64,
    /// Left subset of the best split; zero identifies a leaf.
    left: u64,
    /// Log2 read size as input to the next contraction.
    read_log2: f64,
}

#[derive(Clone, Debug)]
struct OuterEntry {
    cost: f64,
    legs: Vec<LegId>,
    left: u64,
    read_log2: f64,
}

pub const DEFAULT_MAX_N: usize = 26;

pub(crate) struct ReconfigurationPlan {
    pub path: SsaPath,
    pub log2_flops: f64,
    pub step_legs: Vec<Vec<LegId>>,
}

pub fn optimal_dp(net: &TensorNetwork, max_n: usize) -> Result<(SsaPath, PathStats), String> {
    optimal_dp_with_objective(net, max_n, PlannerObjective::FIXED)
}

pub fn optimal_dp_with_objective(
    net: &TensorNetwork,
    max_n: usize,
    objective: PlannerObjective,
) -> Result<(SsaPath, PathStats), String> {
    optimal_dp_until_with_objective(net, max_n, None, objective)
}

/// Deadline-aware version of [`optimal_dp`].
/// Expiry returns an error because a partial DP table is not a valid SSA path.
pub fn optimal_dp_until(
    net: &TensorNetwork,
    max_n: usize,
    deadline: Option<Instant>,
) -> Result<(SsaPath, PathStats), String> {
    optimal_dp_until_with_objective(net, max_n, deadline, PlannerObjective::FIXED)
}

pub fn optimal_dp_until_with_objective(
    net: &TensorNetwork,
    max_n: usize,
    deadline: Option<Instant>,
    objective: PlannerObjective,
) -> Result<(SsaPath, PathStats), String> {
    let path = match objective.kind() {
        ObjectiveKind::TotalFlops => {
            optimal_dp_path::<true, false>(net, max_n, deadline, objective)?
        }
        ObjectiveKind::TotalReadWrite => {
            optimal_dp_path::<false, true>(net, max_n, deadline, objective)?
        }
        ObjectiveKind::Weighted => optimal_dp_path::<true, true>(net, max_n, deadline, objective)?,
    };
    let stats = simulate_path(net, &path)?;
    Ok((path, stats))
}

pub(crate) fn optimal_dp_reconfiguration_core<
    const TRACK_FLOPS: bool,
    const TRACK_READ_WRITE: bool,
>(
    net: &TensorNetwork,
    max_n: usize,
    objective: PlannerObjective,
) -> Result<ReconfigurationPlan, String> {
    let path = optimal_dp_path::<TRACK_FLOPS, TRACK_READ_WRITE>(net, max_n, None, objective)?;
    let (log2_flops, step_legs) = if TRACK_FLOPS {
        simulate_path_flops_log2_and_legs(net, &path)?
    } else {
        (f64::NEG_INFINITY, simulate_path_legs(net, &path)?)
    };
    Ok(ReconfigurationPlan {
        path,
        log2_flops,
        step_legs,
    })
}

fn optimal_dp_path<const TRACK_FLOPS: bool, const TRACK_READ_WRITE: bool>(
    net: &TensorNetwork,
    max_n: usize,
    deadline: Option<Instant>,
    objective: PlannerObjective,
) -> Result<SsaPath, String> {
    if deadline_reached(deadline) {
        return Err("optimal-dp deadline exceeded".into());
    }
    net.validate()?;
    let n = net.n_tensors();
    // One `u64` bit per tensor limits the representation to 64 tensors.
    if n > 64 {
        return Err(format!("optimal 只支持 n≤64（bitmask），当前 n={n}"));
    }
    // Reject sizes above the configured exponential-work limit.
    if n > max_n {
        return Err(format!(
            "n={n} 超过 max_n={max_n}，DP 会爆炸；请用 greedy/rgreedy 或调大 --max-n"
        ));
    }
    if n == 0 {
        return Err("空网络".into());
    }

    // Sorted, deduplicated input legs.
    let input_legs: Vec<Vec<LegId>> = net.inputs.iter().map(|t| sorted_dedup(t)).collect();
    let input_read_log2 = if TRACK_READ_WRITE {
        net.inputs
            .iter()
            .map(|tensor| tensor.iter().map(|&leg| net.log2_dim(leg)).sum::<f64>())
            .collect::<Vec<_>>()
    } else {
        vec![f64::NEG_INFINITY; n]
    };
    // Tensor-holder bitmask for each leg.
    let mut holders: FastMap<LegId, u64> = FastMap::default();
    for (i, legs) in input_legs.iter().enumerate() {
        if i % 64 == 0 && deadline_reached(deadline) {
            return Err("optimal-dp deadline exceeded".into());
        }
        for &l in legs {
            *holders.entry(l).or_insert(0) |= 1u64 << i;
        }
    }
    let metadata = LegMetadata::new(net);

    // Solve shared-leg connected components independently.
    let mut comp_id = vec![usize::MAX; n];
    let mut n_comps = 0;
    for start in 0..n {
        if deadline_reached(deadline) {
            return Err("optimal-dp deadline exceeded".into());
        }
        if comp_id[start] != usize::MAX {
            continue;
        }
        let mut stack = vec![start];
        comp_id[start] = n_comps;
        while let Some(u) = stack.pop() {
            if deadline_reached(deadline) {
                return Err("optimal-dp deadline exceeded".into());
            }
            for &l in &input_legs[u] {
                let m = holders[&l];
                for v in BitIter(m) {
                    if comp_id[v] == usize::MAX {
                        comp_id[v] = n_comps;
                        stack.push(v);
                    }
                }
            }
        }
        n_comps += 1;
    }

    let mut path: SsaPath = Vec::new();
    let mut roots: Vec<(usize, Vec<LegId>, f64)> = Vec::new();
    for c in 0..n_comps {
        if deadline_reached(deadline) {
            return Err("optimal-dp deadline exceeded".into());
        }
        let members: Vec<usize> = (0..n).filter(|&i| comp_id[i] == c).collect();
        let (root, legs) = dp_component::<TRACK_FLOPS, TRACK_READ_WRITE>(
            &input_legs,
            &input_read_log2,
            &holders,
            &metadata,
            &members,
            &mut path,
            n,
            deadline,
            objective,
        )?;
        let read_log2 = if !TRACK_READ_WRITE {
            f64::NEG_INFINITY
        } else if members.len() == 1 {
            input_read_log2[members[0]]
        } else {
            legs.iter().map(|&l| metadata.log2_dim(l)).sum()
        };
        roots.push((root, legs, read_log2));
    }
    append_optimal_outer_products::<TRACK_FLOPS, TRACK_READ_WRITE>(
        &roots, &metadata, &mut path, n, deadline, objective,
    )?;

    if deadline_reached(deadline) {
        return Err("optimal-dp deadline exceeded".into());
    }
    Ok(path)
}

fn append_optimal_outer_products<const TRACK_FLOPS: bool, const TRACK_READ_WRITE: bool>(
    roots: &[(usize, Vec<LegId>, f64)],
    metadata: &LegMetadata,
    path: &mut SsaPath,
    n_inputs: usize,
    deadline: Option<Instant>,
    objective: PlannerObjective,
) -> Result<(), String> {
    let k = roots.len();
    if k <= 1 {
        return Ok(());
    }
    let full = if k == 64 { !0u64 } else { (1u64 << k) - 1 };
    let mut table: FastMap<u64, OuterEntry> = FastMap::default();
    let mut by_size = vec![Vec::new(); k + 1];
    for (index, (_, legs, read_log2)) in roots.iter().enumerate() {
        let mask = 1u64 << index;
        table.insert(
            mask,
            OuterEntry {
                cost: f64::NEG_INFINITY,
                legs: legs.clone(),
                left: 0,
                read_log2: *read_log2,
            },
        );
        by_size[1].push(mask);
    }

    for size in 2..=k {
        if deadline_reached(deadline) {
            return Err("optimal-dp deadline exceeded".into());
        }
        for left_size in 1..=size / 2 {
            let right_size = size - left_size;
            let (lower, upper) = by_size.split_at_mut(size);
            let left_masks = &lower[left_size];
            let right_masks = &lower[right_size];
            let target_masks = &mut upper[0];
            for &left_mask in left_masks {
                for &right_mask in right_masks {
                    if left_mask & right_mask != 0
                        || (left_size == right_size && left_mask >= right_mask)
                    {
                        continue;
                    }
                    let left = &table[&left_mask];
                    let right = &table[&right_mask];
                    let step_log2 = if TRACK_FLOPS {
                        union_log2_sum(&left.legs, &right.legs, metadata)
                    } else {
                        f64::NEG_INFINITY
                    };
                    let result_legs = legs_union(&left.legs, &right.legs)
                        .into_iter()
                        .filter(|&leg| metadata.is_output(leg))
                        .collect::<Vec<_>>();
                    let result_log2 = if TRACK_READ_WRITE {
                        result_legs.iter().map(|&leg| metadata.log2_dim(leg)).sum()
                    } else {
                        f64::NEG_INFINITY
                    };
                    let read_write_log2 = if TRACK_READ_WRITE {
                        logaddexp2(logaddexp2(left.read_log2, right.read_log2), result_log2)
                    } else {
                        f64::NEG_INFINITY
                    };
                    let step_score = objective.score_terms_log2(step_log2, read_write_log2);
                    let cost = logaddexp2(logaddexp2(left.cost, right.cost), step_score);
                    let mask = left_mask | right_mask;
                    if table.get(&mask).is_some_and(|entry| entry.cost <= cost) {
                        continue;
                    }
                    let existed = table
                        .insert(
                            mask,
                            OuterEntry {
                                cost,
                                legs: result_legs,
                                left: left_mask,
                                read_log2: result_log2,
                            },
                        )
                        .is_some();
                    if !existed {
                        target_masks.push(mask);
                    }
                }
            }
        }
    }

    fn emit(
        mask: u64,
        table: &FastMap<u64, OuterEntry>,
        roots: &[(usize, Vec<LegId>, f64)],
        path: &mut SsaPath,
        n_inputs: usize,
    ) -> usize {
        let entry = &table[&mask];
        if entry.left == 0 {
            return roots[mask.trailing_zeros() as usize].0;
        }
        let left = emit(entry.left, table, roots, path, n_inputs);
        let right = emit(mask ^ entry.left, table, roots, path, n_inputs);
        path.push((left.min(right), left.max(right)));
        n_inputs + path.len() - 1
    }

    if !table.contains_key(&full) {
        return Err("optimal-dp failed to merge disconnected components".into());
    }
    emit(full, &table, roots, path, n_inputs);
    Ok(())
}

/// Solve one connected component and append globally numbered SSA steps.
#[allow(clippy::too_many_arguments)]
fn dp_component<const TRACK_FLOPS: bool, const TRACK_READ_WRITE: bool>(
    input_legs: &[Vec<LegId>],
    input_read_log2: &[f64],
    holders: &FastMap<LegId, u64>,
    metadata: &LegMetadata,
    members: &[usize],
    path: &mut SsaPath,
    n_inputs: usize,
    deadline: Option<Instant>,
    objective: PlannerObjective,
) -> Result<(usize, Vec<LegId>), String> {
    let k = members.len();
    if k == 1 {
        return Ok((members[0], input_legs[members[0]].clone()));
    }
    // Remap component members to dense local bit positions.
    let glob = |local_mask_bit: usize| members[local_mask_bit];
    let mut local_of: FastMap<usize, usize> = FastMap::default();
    for (li, &gi) in members.iter().enumerate() {
        local_of.insert(gi, li);
    }
    // Local holder masks use the dense component indices.
    let mut lholders: FastMap<LegId, u64> = FastMap::default();
    for (li, &gi) in members.iter().enumerate() {
        for &l in &input_legs[gi] {
            *lholders.entry(l).or_insert(0) |= 1u64 << li;
        }
    }
    let full: u64 = if k == 64 { !0u64 } else { (1u64 << k) - 1 };
    let member_mask_global: u64 = members.iter().fold(0u64, |m, &g| m | (1u64 << g));
    // Retain legs held outside the subset or included in the output.
    let is_free = |l: LegId, s: u64, lholders: &FastMap<LegId, u64>| -> bool {
        let lh = lholders.get(&l).copied().unwrap_or(0);
        if lh & !s != 0 {
            return true;
        }
        if holders[&l] & !member_mask_global != 0 {
            return true;
        }
        metadata.is_output(l)
    };

    let mut table: FastMap<u64, Entry> = FastMap::default();
    let mut by_size: Vec<Vec<u64>> = vec![Vec::new(); k + 1];
    for li in 0..k {
        let gi = glob(li);
        let mask = 1u64 << li;
        let legs = input_legs[gi].clone();
        let mut adj = 0u64;
        for &l in &legs {
            adj |= lholders[&l];
        }
        adj &= !mask & full;
        table.insert(
            mask,
            Entry {
                cost: if TRACK_FLOPS && !TRACK_READ_WRITE {
                    0.0
                } else {
                    f64::NEG_INFINITY
                },
                legs,
                adj,
                left: 0,
                read_log2: input_read_log2[gi],
            },
        );
        by_size[1].push(mask);
    }

    // Fill the table in increasing subset size.
    for s in 2..=k {
        if deadline_reached(deadline) {
            return Err("optimal-dp deadline exceeded".into());
        }
        for i in 1..=s / 2 {
            let j = s - i;
            // Borrow shorter levels while writing the current level.
            let (lower, upper) = by_size.split_at_mut(s);
            let list_i: &[u64] = &lower[i];
            let list_j: &[u64] = &lower[j];
            let push_target: &mut Vec<u64> = &mut upper[0];
            for (m1_i, &m1) in list_i.iter().enumerate() {
                if m1_i % 64 == 0 && deadline_reached(deadline) {
                    return Err("optimal-dp deadline exceeded".into());
                }
                let e1_adj = table[&m1].adj;
                for (m2_i, &m2) in list_j.iter().enumerate() {
                    if m2_i % 256 == 0 && deadline_reached(deadline) {
                        return Err("optimal-dp deadline exceeded".into());
                    }
                    if m1 & m2 != 0 {
                        continue;
                    }
                    if i == j && m1 >= m2 {
                        continue;
                    }
                    if e1_adj & m2 == 0 {
                        continue;
                    }
                    let e1 = &table[&m1];
                    let e2 = &table[&m2];
                    // Materialize result legs only after the score can improve.
                    let step_log2 = if TRACK_FLOPS {
                        union_log2_sum(&e1.legs, &e2.legs, metadata)
                    } else {
                        f64::NEG_INFINITY
                    };
                    let m = m1 | m2;
                    let (cost, result_legs, result_log2) = if TRACK_FLOPS && !TRACK_READ_WRITE {
                        let step = step_log2.exp2();
                        let total = e1.cost + e2.cost + step;
                        if !step.is_finite() || !total.is_finite() {
                            return Err("pure-FLOPs linear f64 overflow in local optimal DP".into());
                        }
                        if table.get(&m).is_some_and(|prev| prev.cost <= total) {
                            continue;
                        }
                        let legs = legs_union(&e1.legs, &e2.legs)
                            .into_iter()
                            .filter(|&l| is_free(l, m, &lholders))
                            .collect::<Vec<_>>();
                        (total, legs, f64::NEG_INFINITY)
                    } else {
                        let legs = legs_union(&e1.legs, &e2.legs)
                            .into_iter()
                            .filter(|&l| is_free(l, m, &lholders))
                            .collect::<Vec<_>>();
                        let result_log2 = legs.iter().map(|&l| metadata.log2_dim(l)).sum();
                        let read_write_log2 =
                            logaddexp2(logaddexp2(e1.read_log2, e2.read_log2), result_log2);
                        let step_score_log2 =
                            objective.score_terms_log2(step_log2, read_write_log2);
                        let total = logaddexp2(logaddexp2(e1.cost, e2.cost), step_score_log2);
                        if table.get(&m).is_some_and(|prev| prev.cost <= total) {
                            continue;
                        }
                        (total, legs, result_log2)
                    };
                    let adj = (table[&m1].adj | table[&m2].adj) & !m & full;
                    let existed = table
                        .insert(
                            m,
                            Entry {
                                cost,
                                legs: result_legs,
                                adj,
                                left: m1,
                                read_log2: result_log2,
                            },
                        )
                        .is_some();
                    if !existed {
                        push_target.push(m);
                    }
                }
            }
        }
    }

    let root_entry = table
        .get(&full)
        .ok_or("DP 未找到全集方案（分量内部不连通？不应发生）")?
        .clone();

    // Reconstruct the best split tree in postorder.
    fn emit(
        mask: u64,
        table: &FastMap<u64, Entry>,
        members: &[usize],
        path: &mut SsaPath,
        n_inputs: usize,
    ) -> usize {
        let e = &table[&mask];
        if e.left == 0 {
            return members[mask.trailing_zeros() as usize];
        }
        let l = emit(e.left, table, members, path, n_inputs);
        let r = emit(mask ^ e.left, table, members, path, n_inputs);
        let (x, y) = if l < r { (l, r) } else { (r, l) };
        path.push((x, y));
        n_inputs + path.len() - 1
    }
    if deadline_reached(deadline) {
        return Err("optimal-dp deadline exceeded".into());
    }
    let root = emit(full, &table, members, path, n_inputs);
    Ok((root, root_entry.legs))
}

#[inline]
fn deadline_reached(deadline: Option<Instant>) -> bool {
    deadline.map(|d| Instant::now() >= d).unwrap_or(false)
}

/// Iterate over set-bit indices in ascending order.
struct BitIter(u64);
impl Iterator for BitIter {
    type Item = usize;
    fn next(&mut self) -> Option<usize> {
        if self.0 == 0 {
            None
        } else {
            let i = self.0.trailing_zeros() as usize;
            self.0 &= self.0 - 1;
            Some(i)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixed_score_reversal_net() -> TensorNetwork {
        TensorNetwork {
            name: "optimal-fixed-score".into(),
            inputs: vec![vec![0, 1], vec![0, 2], vec![1, 3], vec![2, 4]],
            output: vec![3, 4],
            size_dict: [(0, 6), (1, 4), (2, 6), (3, 5), (4, 8)]
                .into_iter()
                .collect(),
        }
    }

    /// Enumerate all binary SSA contraction sequences, including outer products.
    fn all_binary_paths(n: usize) -> Vec<SsaPath> {
        fn rec(alive: &mut Vec<usize>, next_id: usize, path: &mut SsaPath, out: &mut Vec<SsaPath>) {
            if alive.len() == 1 {
                out.push(path.clone());
                return;
            }
            let len = alive.len();
            for i in 0..len {
                for j in i + 1..len {
                    let b = alive.remove(j);
                    let a = alive.remove(i);
                    alive.push(next_id);
                    path.push((a.min(b), a.max(b)));
                    rec(alive, next_id + 1, path, out);
                    path.pop();
                    alive.pop();
                    alive.insert(i, a);
                    alive.insert(j, b);
                }
            }
        }

        let mut out = Vec::new();
        rec(&mut (0..n).collect(), n, &mut SsaPath::new(), &mut out);
        out
    }

    fn linear_totals(stats: &PathStats) -> (f64, f64) {
        (10f64.powf(stats.log10_flops), stats.log2_read_write.exp2())
    }

    #[test]
    fn fixed_score_dp_matches_exhaustive_optimum() {
        let net = fixed_score_reversal_net();
        let all = all_binary_paths(net.n_tensors());
        assert_eq!(all.len(), 18);

        let exhaustive_best = || {
            all.iter()
                .map(|path| {
                    let stats = simulate_path(&net, path).unwrap();
                    (
                        PlannerObjective::FIXED.score_path_log2(&stats),
                        path.clone(),
                        stats,
                    )
                })
                .min_by(|a, b| a.0.total_cmp(&b.0))
                .unwrap()
        };

        let (combo_path, combo_stats) = optimal_dp(&net, 26).unwrap();
        let brute_combo = exhaustive_best();

        assert_eq!(combo_path, brute_combo.1);
        assert_eq!(
            PlannerObjective::FIXED
                .score_path_log2(&combo_stats)
                .to_bits(),
            brute_combo.0.to_bits()
        );
    }

    #[test]
    fn runtime_objectives_match_exhaustive_optima() {
        let net = fixed_score_reversal_net();
        let all = all_binary_paths(net.n_tensors());
        let objectives = [
            PlannerObjective::new(1.0, 0.0).unwrap(),
            PlannerObjective::new(0.0, 1.0).unwrap(),
            PlannerObjective::new(3.0, 7.0).unwrap(),
        ];

        for objective in objectives {
            let brute = all
                .iter()
                .map(|path| {
                    let stats = simulate_path(&net, path).unwrap();
                    objective.score_path_log2(&stats)
                })
                .min_by(f64::total_cmp)
                .unwrap();
            let (_, stats) = optimal_dp_with_objective(&net, 26, objective).unwrap();
            assert_eq!(objective.score_path_log2(&stats).to_bits(), brute.to_bits());
        }
    }

    #[test]
    fn fixed_score_dp_matches_exhaustive_with_a_repeated_leaf_leg() {
        let net = TensorNetwork {
            name: "fixed-score-repeated-leaf".into(),
            inputs: vec![vec![0, 0, 1], vec![1, 2], vec![2, 3], vec![3]],
            output: vec![],
            size_dict: [(0, 3), (1, 2), (2, 5), (3, 7)].into_iter().collect(),
        };
        let all = all_binary_paths(net.n_tensors());
        let brute = all
            .iter()
            .map(|path| {
                let stats = simulate_path(&net, path).unwrap();
                PlannerObjective::FIXED.score_path_log2(&stats)
            })
            .min_by(f64::total_cmp)
            .unwrap();
        let (path, stats) = optimal_dp(&net, 26).unwrap();

        assert_eq!(
            PlannerObjective::FIXED.score_path_log2(&stats).to_bits(),
            brute.to_bits()
        );
        let replay = simulate_path(&net, &path).unwrap();
        assert_eq!(
            stats.log2_read_write.to_bits(),
            replay.log2_read_write.to_bits()
        );
    }

    #[test]
    fn disconnected_components_use_the_exact_outer_product_tree() {
        let net = TensorNetwork {
            name: "disconnected-outer-product".into(),
            inputs: vec![vec![0], vec![1], vec![2], vec![3]],
            output: vec![0, 1, 2, 3],
            size_dict: [(0, 2), (1, 2), (2, 3), (3, 3)].into_iter().collect(),
        };
        let brute = all_binary_paths(net.n_tensors())
            .into_iter()
            .map(|path| {
                let stats = simulate_path(&net, &path).unwrap();
                (PlannerObjective::FIXED.score_path_log2(&stats), path)
            })
            .min_by(|a, b| a.0.total_cmp(&b.0))
            .unwrap();
        let (path, stats) = optimal_dp(&net, 4).unwrap();

        assert_eq!(
            PlannerObjective::FIXED.score_path_log2(&stats).to_bits(),
            brute.0.to_bits()
        );
        assert_eq!(
            PlannerObjective::FIXED.score_path_log2(&stats).exp2(),
            4528.0
        );
        assert_ne!(path, vec![(0, 1), (2, 3), (4, 5)]);
    }

    #[test]
    fn fixed_score_counts_both_inputs_and_the_final_scalar() {
        let net = TensorNetwork {
            name: "scalar-write".into(),
            inputs: vec![vec![0], vec![0]],
            output: vec![],
            size_dict: [(0, 7)].into_iter().collect(),
        };
        let (_, stats) = optimal_dp(&net, 2).unwrap();
        let (flops, read_write) = linear_totals(&stats);
        assert!((flops - 7.0).abs() < 1e-12);
        assert!((read_write - 15.0).abs() < 1e-12);
        assert!((PlannerObjective::FIXED.score_path_log2(&stats).exp2() - 967.0).abs() < 1e-9);
    }

    #[test]
    fn reconfiguration_plan_matches_public_optimal_replay() {
        let net = fixed_score_reversal_net();
        let plan = optimal_dp_reconfiguration_core::<true, true>(&net, 26, PlannerObjective::FIXED)
            .unwrap();
        let (path, stats) = optimal_dp(&net, 26).unwrap();
        let (_, step_legs) = crate::path::simulate_path_full(&net, &path).unwrap();

        assert_eq!(plan.path, path);
        assert_eq!(plan.step_legs, step_legs);
        assert_eq!(
            (plan.log2_flops * std::f64::consts::LOG10_2).to_bits(),
            stats.log10_flops.to_bits()
        );
    }
}
