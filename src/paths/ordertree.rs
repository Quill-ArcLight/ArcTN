//! Interval dynamic programming for an optimal tree under a fixed leaf order.
//!
//! Leaf orders may come from an existing path or reverse Cuthill-McKee ordering.
//!
//! `logS(i,j)` stores the result size of interval `i..=j`. Adjacent occurrence
//! pairs for each leg form a two-dimensional prefix sum, allowing the shared
//! leg weight of a split to be queried in `O(1)`. Adjacent pairs cross any
//! boundary uniquely, including legs with more than two holders.
//!
//! Scores accumulate in log2 space where required. Complexity is `O(n^3)` time
//! and `O(n^2)` memory.

use crate::network::TensorNetwork;
use crate::objective::{ObjectiveKind, PlannerObjective};
use crate::path::{logaddexp2, simulate_path, sorted_dedup, PathStats, SsaPath};
use crate::tree::CTreeCore;
use std::time::Instant;

/// Public size limit for the `O(n^3)`-time, `O(n^2)`-memory order DP.
pub const MAX_ORDER_DP_N: usize = 5_000;

/// Return the in-order leaf sequence of an existing contraction tree.
pub fn leaf_order_of_path(net: &TensorNetwork, path: &SsaPath) -> Result<Vec<usize>, String> {
    leaf_order_of_path_with_objective(net, path, PlannerObjective::FIXED)
}

pub fn leaf_order_of_path_with_objective(
    net: &TensorNetwork,
    path: &SsaPath,
    objective: PlannerObjective,
) -> Result<Vec<usize>, String> {
    match objective.kind() {
        ObjectiveKind::TotalFlops => leaf_order_of_path_core::<true, false>(net, path, objective),
        ObjectiveKind::TotalReadWrite => {
            leaf_order_of_path_core::<false, true>(net, path, objective)
        }
        ObjectiveKind::Weighted => leaf_order_of_path_core::<true, true>(net, path, objective),
    }
}

fn leaf_order_of_path_core<const TRACK_FLOPS: bool, const TRACK_READ_WRITE: bool>(
    net: &TensorNetwork,
    path: &SsaPath,
    objective: PlannerObjective,
) -> Result<Vec<usize>, String> {
    let tree =
        CTreeCore::<TRACK_FLOPS, TRACK_READ_WRITE>::from_path_with_objective(net, path, objective)?;
    Ok(tree.leaf_inorder())
}

/// Return a reverse Cuthill-McKee order.
/// BFS starts from minimum-degree tensors and visits neighbors by degree.
pub fn rcm_order(net: &TensorNetwork) -> Vec<usize> {
    let n = net.n_tensors();
    // Tensors are adjacent when they share any leg.
    let holders = net.leg_holders();
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for hs in holders.values() {
        for (ii, &a) in hs.iter().enumerate() {
            for &b in &hs[ii + 1..] {
                adj[a].push(b);
                adj[b].push(a);
            }
        }
    }
    for a in &mut adj {
        a.sort_unstable();
        a.dedup();
    }
    let deg: Vec<usize> = adj.iter().map(|a| a.len()).collect();
    let mut order = Vec::with_capacity(n);
    let mut seen = vec![false; n];
    // Restart from the minimum-degree unseen tensor in each component.
    loop {
        let start = (0..n).filter(|&v| !seen[v]).min_by_key(|&v| (deg[v], v));
        let Some(start) = start else { break };
        seen[start] = true;
        let mut queue = std::collections::VecDeque::from([start]);
        while let Some(v) = queue.pop_front() {
            order.push(v);
            let mut nb: Vec<usize> = adj[v].iter().copied().filter(|&u| !seen[u]).collect();
            nb.sort_by_key(|&u| (deg[u], u));
            for u in nb {
                seen[u] = true;
                queue.push_back(u);
            }
        }
    }
    order.reverse();
    order
}

/// Return the optimal SSA tree for a fixed leaf order.
/// For `n > 1`, `order` must be a permutation of `0..n`.
pub fn order_dp(net: &TensorNetwork, order: &[usize]) -> Result<(SsaPath, PathStats), String> {
    order_dp_with_objective(net, order, PlannerObjective::FIXED)
}

pub fn order_dp_with_objective(
    net: &TensorNetwork,
    order: &[usize],
    objective: PlannerObjective,
) -> Result<(SsaPath, PathStats), String> {
    order_dp_until_with_objective(net, order, None, objective)
}

/// Deadline-aware version of [`order_dp`].
///
/// A partial DP table is not a valid SSA path, so expiry returns `Err`.
pub fn order_dp_until(
    net: &TensorNetwork,
    order: &[usize],
    deadline: Option<Instant>,
) -> Result<(SsaPath, PathStats), String> {
    order_dp_until_with_objective(net, order, deadline, PlannerObjective::FIXED)
}

pub fn order_dp_until_with_objective(
    net: &TensorNetwork,
    order: &[usize],
    deadline: Option<Instant>,
    objective: PlannerObjective,
) -> Result<(SsaPath, PathStats), String> {
    match order_dp_dispatch(net, order, deadline, objective) {
        Ok(result) => Ok(result),
        Err(OrderDpError::Deadline) => Err("order-dp deadline exceeded".into()),
        Err(OrderDpError::Other(error)) => Err(error),
    }
}

/// Optimize a fixed leaf order with an explicit objective and optional deadline.
///
/// Returns `Ok(Some((path, stats)))` after completing the dynamic-programming
/// table, `Ok(None)` if a cooperative deadline expires, and `Err` for invalid
/// input or a computation error. A partial table is never returned as a path.
/// Use [`order_dp_until_with_objective`] to treat deadline expiry as an error.
pub fn order_dp_cooperative_with_objective(
    net: &TensorNetwork,
    order: &[usize],
    deadline: Option<Instant>,
    objective: PlannerObjective,
) -> Result<Option<(SsaPath, PathStats)>, String> {
    match order_dp_dispatch(net, order, deadline, objective) {
        Ok(result) => Ok(Some(result)),
        Err(OrderDpError::Deadline) => Ok(None),
        Err(OrderDpError::Other(error)) => Err(error),
    }
}

#[derive(Debug)]
enum OrderDpError {
    Deadline,
    Other(String),
}

impl From<String> for OrderDpError {
    fn from(error: String) -> Self {
        Self::Other(error)
    }
}

impl From<&str> for OrderDpError {
    fn from(error: &str) -> Self {
        Self::Other(error.to_owned())
    }
}

fn order_dp_dispatch(
    net: &TensorNetwork,
    order: &[usize],
    deadline: Option<Instant>,
    objective: PlannerObjective,
) -> Result<(SsaPath, PathStats), OrderDpError> {
    match objective.kind() {
        ObjectiveKind::TotalFlops => order_dp_core::<true, false>(net, order, deadline, objective),
        ObjectiveKind::TotalReadWrite => {
            order_dp_core::<false, true>(net, order, deadline, objective)
        }
        ObjectiveKind::Weighted => order_dp_core::<true, true>(net, order, deadline, objective),
    }
}

fn order_dp_core<const TRACK_FLOPS: bool, const TRACK_READ_WRITE: bool>(
    net: &TensorNetwork,
    order: &[usize],
    deadline: Option<Instant>,
    objective: PlannerObjective,
) -> Result<(SsaPath, PathStats), OrderDpError> {
    if deadline_reached(deadline) {
        return Err(OrderDpError::Deadline);
    }
    // Validate before reading dimensions in the DP tables.
    net.validate()?;
    let n = net.n_tensors();
    if n > MAX_ORDER_DP_N {
        return Err(
            format!("order-dp 为 O(n³)/O(n²)，只支持 n≤{MAX_ORDER_DP_N}，当前 n={n}").into(),
        );
    }
    if order.len() != n {
        return Err(format!("order 长度 {} ≠ 张量数 {n}", order.len()).into());
    }
    if n <= 1 {
        return Ok((SsaPath::new(), simulate_path(net, &SsaPath::new())?));
    }
    {
        let mut seen = vec![false; n];
        for &t in order {
            if t >= n || std::mem::replace(&mut seen[t], true) {
                return Err("order 不是 0..n 的排列".into());
            }
        }
    }
    let in_output: std::collections::HashSet<u32> = net.output.iter().copied().collect();

    // Sorted occurrence positions for each leg in the leaf order.
    let mut occ: std::collections::HashMap<u32, Vec<usize>> = std::collections::HashMap::new();
    for (pos, &t) in order.iter().enumerate() {
        if deadline_reached(deadline) {
            return Err(OrderDpError::Deadline);
        }
        for l in sorted_dedup(&net.inputs[t]) {
            occ.entry(l).or_default().push(pos);
        }
    }

    // Prefix sum over adjacent occurrence pairs: P[a][b] uses lo < a and hi < b.
    // Check every square-table size before allocation.
    let np1 = n
        .checked_add(1)
        .ok_or_else(|| "order-dp 的 n+1 溢出 usize".to_string())?;
    let pref_len = np1
        .checked_mul(np1)
        .ok_or_else(|| "order-dp 的 (n+1)² 前缀表长度溢出 usize".to_string())?;
    let table_len = n
        .checked_mul(n)
        .ok_or_else(|| "order-dp 的 n² DP 表长度溢出 usize".to_string())?;
    let mut pref = vec![0f64; pref_len];
    // Sort legs for deterministic floating-point accumulation.
    let mut occ_legs: Vec<u32> = occ.keys().copied().collect();
    occ_legs.sort_unstable();
    for l in occ_legs {
        let ps = &occ[&l];
        let w = net.log2_dim(l);
        for t in 0..ps.len().saturating_sub(1) {
            let (a, b) = (ps[t], ps[t + 1]);
            pref[(a + 1) * np1 + (b + 1)] += w;
        }
    }
    for i in 1..np1 {
        if deadline_reached(deadline) {
            return Err(OrderDpError::Deadline);
        }
        for j in 1..np1 {
            pref[i * np1 + j] +=
                pref[(i - 1) * np1 + j] + pref[i * np1 + (j - 1)] - pref[(i - 1) * np1 + (j - 1)];
        }
    }
    // Sum points with lo in [i, k] and hi in [k + 1, j].
    let rect = |i: usize, k: usize, j: usize| -> f64 {
        let (r1, r2, c1, c2) = (i, k + 1, k + 1, j + 1); // Half-open rectangle.
        pref[r2 * np1 + c2] - pref[r1 * np1 + c2] - pref[r2 * np1 + c1] + pref[r1 * np1 + c1]
    };

    // Build logS(i, j) incrementally for each fixed i.
    let total_cnt: std::collections::HashMap<u32, usize> =
        occ.iter().map(|(&l, ps)| (l, ps.len())).collect();
    let mut log_s = vec![0f64; table_len];
    for i in 0..n {
        if deadline_reached(deadline) {
            return Err(OrderDpError::Deadline);
        }
        let mut cnt: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
        let mut acc = 0f64;
        for j in i..n {
            for l in sorted_dedup(&net.inputs[order[j]]) {
                let c = cnt.entry(l).or_insert(0);
                let w = net.log2_dim(l);
                let tot = total_cnt[&l];
                let was_kept = *c > 0 && (*c < tot || in_output.contains(&l));
                *c += 1;
                let now_kept = *c < tot || in_output.contains(&l);
                match (*c == 1, was_kept, now_kept) {
                    (true, _, true) => acc += w,
                    (true, _, false) => {}
                    (false, true, false) => acc -= w,
                    _ => {}
                }
            }
            log_s[i * n + j] = acc;
        }
        // Leaves retain all logical legs for the first contraction cost.
        log_s[i * n + i] = sorted_dedup(&net.inputs[order[i]])
            .iter()
            .map(|&l| net.log2_dim(l))
            .sum();
    }
    let leaf_read_log2 = if TRACK_READ_WRITE {
        order
            .iter()
            .map(|&tensor| {
                net.inputs[tensor]
                    .iter()
                    .map(|&leg| net.log2_dim(leg))
                    .sum::<f64>()
            })
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };

    // Entries of one interval length read only shorter intervals and are parallel-safe.
    use rayon::prelude::*;
    let leaf_cost = if TRACK_FLOPS && !TRACK_READ_WRITE {
        0.0
    } else {
        f64::NEG_INFINITY
    };
    let mut cost = vec![leaf_cost; table_len];
    let mut split = vec![0u32; table_len];
    for len in 2..=n {
        if deadline_reached(deadline) {
            return Err(OrderDpError::Deadline);
        }
        let level: Vec<(usize, f64, u32)> = (0..=(n - len))
            .into_par_iter()
            .map(|i| -> Result<(usize, f64, u32), OrderDpError> {
                if deadline_reached(deadline) {
                    return Err(OrderDpError::Deadline);
                }
                let j = i + len - 1;
                let (mut best, mut bk) = (f64::INFINITY, i);
                for k in i..j {
                    if k % 64 == 0 && deadline_reached(deadline) {
                        return Err(OrderDpError::Deadline);
                    }
                    let step_log2 = if TRACK_FLOPS {
                        log_s[i * n + k] + log_s[(k + 1) * n + j] - rect(i, k, j)
                    } else {
                        f64::NEG_INFINITY
                    };
                    let read_write_log2 = if TRACK_READ_WRITE {
                        let left_read_log2 = if i == k {
                            leaf_read_log2[i]
                        } else {
                            log_s[i * n + k]
                        };
                        let right_read_log2 = if k + 1 == j {
                            leaf_read_log2[j]
                        } else {
                            log_s[(k + 1) * n + j]
                        };
                        logaddexp2(
                            logaddexp2(left_read_log2, right_read_log2),
                            log_s[i * n + j],
                        )
                    } else {
                        f64::NEG_INFINITY
                    };
                    let tot = if TRACK_FLOPS && !TRACK_READ_WRITE {
                        let step = step_log2.exp2();
                        let total = cost[i * n + k] + cost[(k + 1) * n + j] + step;
                        if !step.is_finite() || !total.is_finite() {
                            return Err("pure-FLOPs linear f64 overflow in leaf-order DP".into());
                        }
                        total
                    } else {
                        let step_score_log2 =
                            objective.score_terms_log2(step_log2, read_write_log2);
                        logaddexp2(
                            logaddexp2(cost[i * n + k], cost[(k + 1) * n + j]),
                            step_score_log2,
                        )
                    };
                    if tot < best {
                        best = tot;
                        bk = k;
                    }
                }
                Ok((i, best, bk as u32))
            })
            .collect::<Result<Vec<_>, OrderDpError>>()?;
        for (i, best, bk) in level {
            let j = i + len - 1;
            cost[i * n + j] = best;
            split[i * n + j] = bk;
        }
    }

    // Reconstruct iteratively to avoid recursion on unbalanced trees.
    let mut path = SsaPath::with_capacity(n - 1);
    let mut next_id = n;
    let mut ids = vec![usize::MAX; table_len]; // SSA id for interval [i, j].
    let mut stack = vec![(0usize, n - 1, false)];
    while let Some((i, j, expanded)) = stack.pop() {
        if deadline_reached(deadline) {
            return Err(OrderDpError::Deadline);
        }
        if i == j {
            ids[i * n + j] = order[i];
            continue;
        }
        let k = split[i * n + j] as usize;
        if !expanded {
            stack.push((i, j, true));
            stack.push((i, k, false));
            stack.push((k + 1, j, false));
        } else {
            let (a, b) = (ids[i * n + k], ids[(k + 1) * n + j]);
            path.push((a.min(b), a.max(b)));
            ids[i * n + j] = next_id;
            next_id += 1;
        }
    }
    if deadline_reached(deadline) {
        return Err(OrderDpError::Deadline);
    }
    let stats = simulate_path(net, &path)?;
    Ok((path, stats))
}

#[inline]
fn deadline_reached(deadline: Option<Instant>) -> bool {
    deadline.map(|d| Instant::now() >= d).unwrap_or(false)
}

#[cfg(test)]
mod objective_tests {
    use super::{
        leaf_order_of_path, leaf_order_of_path_with_objective, order_dp,
        order_dp_cooperative_with_objective, order_dp_with_objective,
    };
    use crate::network::TensorNetwork;
    use crate::objective::PlannerObjective;
    use crate::path::{simulate_path, SsaPath};

    fn fixed_order_paths() -> Vec<SsaPath> {
        vec![
            vec![(0, 1), (2, 4), (3, 5)],
            vec![(1, 2), (0, 4), (3, 5)],
            vec![(0, 1), (2, 3), (4, 5)],
            vec![(1, 2), (3, 4), (0, 5)],
            vec![(2, 3), (1, 4), (0, 5)],
        ]
    }

    fn open_hyperedge_net() -> TensorNetwork {
        TensorNetwork {
            name: "order-fixed-score".into(),
            inputs: vec![vec![0, 1], vec![0, 2], vec![0, 3], vec![2, 4]],
            output: vec![1, 3, 4],
            size_dict: [(0, 3), (1, 5), (2, 7), (3, 2), (4, 11)]
                .into_iter()
                .collect(),
        }
    }

    #[test]
    fn fixed_score_matches_exhaustive_fixed_order_optimum() {
        let net = open_hyperedge_net();
        let order = [0, 1, 2, 3];
        let (_, stats) = order_dp(&net, &order).unwrap();
        let score = PlannerObjective::FIXED.score_path_log2(&stats);
        let exhaustive = fixed_order_paths()
            .iter()
            .map(|path| {
                PlannerObjective::FIXED.score_path_log2(&simulate_path(&net, path).unwrap())
            })
            .min_by(f64::total_cmp)
            .unwrap();
        assert!((score - exhaustive).abs() < 1e-12);
    }

    #[test]
    fn runtime_objectives_match_exhaustive_fixed_order_optima() {
        let net = open_hyperedge_net();
        let order = [0, 1, 2, 3];
        for objective in [
            PlannerObjective::new(1.0, 0.0).unwrap(),
            PlannerObjective::new(0.0, 1.0).unwrap(),
            PlannerObjective::new(3.0, 7.0).unwrap(),
        ] {
            let exhaustive = fixed_order_paths()
                .iter()
                .map(|path| objective.score_path_log2(&simulate_path(&net, path).unwrap()))
                .min_by(f64::total_cmp)
                .unwrap();
            let (_, stats) = order_dp_with_objective(&net, &order, objective).unwrap();
            assert_eq!(
                objective.score_path_log2(&stats).to_bits(),
                exhaustive.to_bits()
            );
        }
    }

    #[test]
    fn objective_specialized_leaf_order_matches_the_tree_and_returns_replayable_paths() {
        let net = open_hyperedge_net();
        let input: SsaPath = vec![(0, 1), (2, 4), (3, 5)];
        let expected_order = leaf_order_of_path(&net, &input).unwrap();
        for objective in [
            PlannerObjective::new(1.0, 0.0).unwrap(),
            PlannerObjective::new(0.0, 1.0).unwrap(),
            PlannerObjective::new(3.0, 7.0).unwrap(),
        ] {
            let order = leaf_order_of_path_with_objective(&net, &input, objective).unwrap();
            assert_eq!(order, expected_order);
            let (path, stats) = order_dp_with_objective(&net, &order, objective).unwrap();
            let replay = simulate_path(&net, &path).unwrap();
            assert_eq!(stats.log10_flops.to_bits(), replay.log10_flops.to_bits());
            assert_eq!(
                stats.log2_read_write.to_bits(),
                replay.log2_read_write.to_bits()
            );
        }
    }

    #[test]
    fn cooperative_order_dp_distinguishes_deadlines_from_invalid_orders() {
        let net = open_hyperedge_net();
        let expired = std::time::Instant::now() - std::time::Duration::from_millis(1);
        let stopped = order_dp_cooperative_with_objective(
            &net,
            &[0, 1, 2, 3],
            Some(expired),
            PlannerObjective::FIXED,
        )
        .unwrap();
        assert!(stopped.is_none());

        let error =
            order_dp_cooperative_with_objective(&net, &[0, 0, 2, 3], None, PlannerObjective::FIXED)
                .unwrap_err();
        assert!(error.contains("order"));
    }

    #[test]
    fn cooperative_order_dp_propagates_pure_flops_overflow() {
        let net = TensorNetwork {
            name: "order-dp-overflow".into(),
            inputs: (0..72u32).map(|tensor| vec![tensor % 36]).collect(),
            output: vec![],
            size_dict: (0..36u32).map(|leg| (leg, 1usize << 30)).collect(),
        };
        let objective = PlannerObjective::new(1.0, 0.0).unwrap();
        let order = (0..72usize).collect::<Vec<_>>();

        let error = order_dp_cooperative_with_objective(&net, &order, None, objective).unwrap_err();

        assert!(error.contains("pure-FLOPs linear f64 overflow"), "{error}");
    }

    #[test]
    fn fixed_score_matches_exhaustive_order_with_a_repeated_leaf_leg() {
        let net = TensorNetwork {
            name: "order-fixed-score-repeated-leaf".into(),
            inputs: vec![vec![0, 0, 1], vec![1, 2], vec![2, 3], vec![3]],
            output: vec![],
            size_dict: [(0, 3), (1, 2), (2, 5), (3, 7)].into_iter().collect(),
        };
        let order = [0, 1, 2, 3];
        let (path, stats) = order_dp(&net, &order).unwrap();
        let exhaustive = fixed_order_paths()
            .into_iter()
            .map(|candidate| {
                let stats = simulate_path(&net, &candidate).unwrap();
                PlannerObjective::FIXED.score_path_log2(&stats)
            })
            .min_by(f64::total_cmp)
            .unwrap();

        assert_eq!(
            PlannerObjective::FIXED.score_path_log2(&stats).to_bits(),
            exhaustive.to_bits()
        );
        let replay = simulate_path(&net, &path).unwrap();
        assert_eq!(
            stats.log2_read_write.to_bits(),
            replay.log2_read_write.to_bits()
        );
    }

    #[test]
    fn fixed_score_is_not_worse_than_the_path_that_supplied_its_leaf_order() {
        let net = open_hyperedge_net();
        for input in fixed_order_paths() {
            let before = simulate_path(&net, &input).unwrap();
            let order = leaf_order_of_path(&net, &input).unwrap();
            let (_, after) = order_dp(&net, &order).unwrap();
            assert!(
                PlannerObjective::FIXED.score_path_log2(&after)
                    <= PlannerObjective::FIXED.score_path_log2(&before) + 1e-12
            );
        }
    }

    #[test]
    fn fixed_score_counts_both_inputs_and_the_final_scalar() {
        let net = TensorNetwork {
            name: "order-scalar-write".into(),
            inputs: vec![vec![0], vec![0]],
            output: vec![],
            size_dict: [(0, 7)].into_iter().collect(),
        };
        let (_, stats) = order_dp(&net, &[0, 1]).unwrap();
        assert!((10f64.powf(stats.log10_flops) - 7.0).abs() < 1e-12);
        assert_eq!(stats.log2_total_size, 0.0);
        assert!((stats.log2_read_write.exp2() - 15.0).abs() < 1e-12);
        assert!((PlannerObjective::FIXED.score_path_log2(&stats).exp2() - 967.0).abs() < 1e-9);
    }
}
