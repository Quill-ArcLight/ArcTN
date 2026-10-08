//! Interval dynamic programming for an optimal tree under a fixed leaf order.
//!
//! Leaf orders may come from an existing path or reverse Cuthill-McKee ordering.
//!
//! `logS(i,j)` stores the result size of interval `i..=j`. Adjacent occurrence
//! pairs for each leg form a two-dimensional prefix sum, allowing the shared
//! leg weight of a split to be queried in `O(1)`. Adjacent pairs cross any
//! boundary uniquely, including legs with more than two holders.
//!
//! With the opt-in `integer-order-dp` feature, binary dimensions use checked
//! integer costs for supported objectives when a complete cost fits u128.
//! Default builds and unsupported cases retain the original arithmetic.
//! Complexity is `O(n^3)` time and `O(n^2)` memory.

mod cost;

use cost::{BinaryCosts, IntervalCosts, LogCosts};

use crate::network::TensorNetwork;
use crate::objective::{ObjectiveKind, PlannerObjective};
use crate::path::{simulate_path, sorted_dedup, PathStats, SsaPath};
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
    if cfg!(feature = "integer-order-dp") && crate::integer_cost::supports(net, objective) {
        if let Some(result) = order_dp_arithmetic::<BinaryCosts, TRACK_FLOPS, TRACK_READ_WRITE>(
            net, order, deadline, objective,
        )? {
            return Ok(result);
        }
        // A large intermediate/candidate is not itself a failure. Only retry
        // when every complete fixed-order tree exceeds the integer range.
        // Pass the original deadline so a retry never receives extra time.
    }
    order_dp_arithmetic::<LogCosts, TRACK_FLOPS, TRACK_READ_WRITE>(net, order, deadline, objective)?
        .ok_or_else(|| "pure-FLOPs linear f64 overflow in leaf-order DP".into())
}

fn order_dp_arithmetic<C: IntervalCosts, const TRACK_FLOPS: bool, const TRACK_READ_WRITE: bool>(
    net: &TensorNetwork,
    order: &[usize],
    deadline: Option<Instant>,
    objective: PlannerObjective,
) -> Result<Option<(SsaPath, PathStats)>, OrderDpError> {
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
        return Ok(Some((SsaPath::new(), simulate_path(net, &SsaPath::new())?)));
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
    let mut pref = vec![C::ZERO_EXPONENT; pref_len];
    // Sort legs for deterministic floating-point accumulation.
    let mut occ_legs: Vec<u32> = occ.keys().copied().collect();
    occ_legs.sort_unstable();
    for l in occ_legs {
        let ps = &occ[&l];
        let w = C::dimension(net, l);
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
            let neighbors =
                pref[(i - 1) * np1 + j] + pref[i * np1 + (j - 1)] - pref[(i - 1) * np1 + (j - 1)];
            pref[i * np1 + j] += neighbors;
        }
    }
    // Sum points with lo in [i, k] and hi in [k + 1, j].
    let rect = |i: usize, k: usize, j: usize| -> C::Exponent {
        let (r1, r2, c1, c2) = (i, k + 1, k + 1, j + 1); // Half-open rectangle.
        pref[r2 * np1 + c2] - pref[r1 * np1 + c2] - pref[r2 * np1 + c1] + pref[r1 * np1 + c1]
    };

    // Build logS(i, j) incrementally for each fixed i.
    let total_cnt: std::collections::HashMap<u32, usize> =
        occ.iter().map(|(&l, ps)| (l, ps.len())).collect();
    let mut log_s = vec![C::ZERO_EXPONENT; table_len];
    for i in 0..n {
        if deadline_reached(deadline) {
            return Err(OrderDpError::Deadline);
        }
        let mut cnt: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
        let mut acc = C::ZERO_EXPONENT;
        for j in i..n {
            for l in sorted_dedup(&net.inputs[order[j]]) {
                let c = cnt.entry(l).or_insert(0);
                let w = C::dimension(net, l);
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
            .map(|&l| C::dimension(net, l))
            .sum();
    }
    let leaf_read_log2 = if TRACK_READ_WRITE {
        order
            .iter()
            .map(|&tensor| {
                net.inputs[tensor]
                    .iter()
                    .map(|&leg| C::dimension(net, leg))
                    .sum::<C::Exponent>()
            })
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };

    // Entries of one interval length read only shorter intervals and are parallel-safe.
    use rayon::prelude::*;
    let leaf_cost = C::leaf_cost::<TRACK_FLOPS, TRACK_READ_WRITE>();
    let mut cost = vec![leaf_cost; table_len];
    let mut split = vec![0u32; table_len];
    for len in 2..=n {
        if deadline_reached(deadline) {
            return Err(OrderDpError::Deadline);
        }
        let level: Vec<(usize, C::Cost, u32)> = (0..=(n - len))
            .into_par_iter()
            .map(|i| -> Result<(usize, C::Cost, u32), OrderDpError> {
                if deadline_reached(deadline) {
                    return Err(OrderDpError::Deadline);
                }
                let j = i + len - 1;
                let (mut best, mut bk) = (C::unreachable(), i);
                for k in i..j {
                    if k % 64 == 0 && deadline_reached(deadline) {
                        return Err(OrderDpError::Deadline);
                    }
                    let step_log2 = if TRACK_FLOPS {
                        log_s[i * n + k] + log_s[(k + 1) * n + j] - rect(i, k, j)
                    } else {
                        C::ZERO_EXPONENT
                    };
                    let (left_read, right_read) = if TRACK_READ_WRITE {
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
                        (left_read_log2, right_read_log2)
                    } else {
                        (C::ZERO_EXPONENT, C::ZERO_EXPONENT)
                    };
                    let tot = C::candidate::<TRACK_FLOPS, TRACK_READ_WRITE>(
                        cost[i * n + k],
                        cost[(k + 1) * n + j],
                        step_log2,
                        left_read,
                        right_read,
                        log_s[i * n + j],
                        objective,
                    )?;
                    if C::better(tot, best) {
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

    if !C::complete::<TRACK_FLOPS, TRACK_READ_WRITE>(cost[n - 1]) {
        return Ok(None);
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
    Ok(Some((path, stats)))
}

#[inline]
fn deadline_reached(deadline: Option<Instant>) -> bool {
    deadline.map(|d| Instant::now() >= d).unwrap_or(false)
}

#[cfg(test)]
mod objective_tests {
    use super::{
        leaf_order_of_path, leaf_order_of_path_with_objective, order_dp, order_dp_arithmetic,
        order_dp_cooperative_with_objective, order_dp_with_objective, BinaryCosts, LogCosts,
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

    /// Enumerate ordered binary trees independently of the interval DP tables.
    fn all_ordered_paths(order: &[usize]) -> Vec<SsaPath> {
        #[derive(Clone)]
        enum Tree {
            Leaf(usize),
            Pair(Box<Tree>, Box<Tree>),
        }
        fn trees(order: &[usize]) -> Vec<Tree> {
            if order.len() == 1 {
                return vec![Tree::Leaf(order[0])];
            }
            let mut out = Vec::new();
            for k in 1..order.len() {
                for left in trees(&order[..k]) {
                    for right in trees(&order[k..]) {
                        out.push(Tree::Pair(Box::new(left.clone()), Box::new(right)));
                    }
                }
            }
            out
        }
        fn emit(tree: &Tree, n: usize, path: &mut SsaPath) -> usize {
            match tree {
                Tree::Leaf(id) => *id,
                Tree::Pair(left, right) => {
                    let a = emit(left, n, path);
                    let b = emit(right, n, path);
                    let id = n + path.len();
                    path.push((a.min(b), a.max(b)));
                    id
                }
            }
        }
        trees(order)
            .iter()
            .map(|tree| {
                let mut path = Vec::new();
                emit(tree, order.len(), &mut path);
                path
            })
            .collect()
    }

    /// Exact replay uses sets and active holders, not the DP's prefix tables.
    fn exact_binary_score(net: &TensorNetwork, path: &SsaPath, wf: u128, wr: u128) -> Option<u128> {
        use std::collections::{BTreeMap, BTreeSet};
        let mut alive: BTreeMap<_, _> = net
            .inputs
            .iter()
            .enumerate()
            .map(|(i, axes)| {
                (
                    i,
                    (
                        axes.iter().copied().collect::<BTreeSet<_>>(),
                        1u128 << axes.len(),
                    ),
                )
            })
            .collect();
        let mut total = 0u128;
        for (step, &(a, b)) in path.iter().enumerate() {
            let (left, left_size) = alive.remove(&a)?;
            let (right, right_size) = alive.remove(&b)?;
            let union: BTreeSet<_> = left.union(&right).copied().collect();
            let result: BTreeSet<_> = union
                .iter()
                .copied()
                .filter(|leg| {
                    net.output.contains(leg) || alive.values().any(|(legs, _)| legs.contains(leg))
                })
                .collect();
            let result_size = 1u128.checked_shl(result.len().try_into().ok()?)?;
            let flops = if wf == 0 {
                0
            } else {
                1u128
                    .checked_shl(union.len().try_into().ok()?)?
                    .checked_mul(wf)?
            };
            let read_write = if wr == 0 {
                0
            } else {
                left_size
                    .checked_add(right_size)?
                    .checked_add(result_size)?
                    .checked_mul(wr)?
            };
            total = total.checked_add(flops)?.checked_add(read_write)?;
            alive.insert(net.n_tensors() + step, (result, result_size));
        }
        assert_eq!(alive.len(), 1);
        assert_eq!(
            alive.values().next().unwrap().0,
            net.output.iter().copied().collect()
        );
        Some(total)
    }

    fn binary_graph(n: usize, edges: &[(usize, usize)], multiplicity: usize) -> TensorNetwork {
        let mut inputs = vec![Vec::new(); n];
        let mut size_dict = std::collections::HashMap::new();
        for &(a, b) in edges {
            for _ in 0..multiplicity {
                let leg = size_dict.len() as u32;
                inputs[a].push(leg);
                inputs[b].push(leg);
                size_dict.insert(leg, 2);
            }
        }
        TensorNetwork {
            name: "binary-order-graph".into(),
            inputs,
            output: vec![],
            size_dict,
        }
    }

    #[test]
    fn binary_order_objectives_match_independent_exhaustive_search() {
        let cases = [
            // A shared output hyperedge remains after a merge.
            (
                vec![vec![0, 1], vec![0, 2], vec![0, 3], vec![0, 4]],
                vec![0, 1, 4],
            ),
            // Raw input reads preserve repeated axes; the result is a scalar.
            (vec![vec![0, 0, 1], vec![1, 2], vec![2, 3], vec![3]], vec![]),
            // Outer products, a scalar input, and a repeated axis.
            (vec![vec![0, 0, 1], vec![2], vec![], vec![3]], vec![1, 2, 3]),
        ];
        for (inputs, output) in cases {
            let net = TensorNetwork {
                name: "binary-order-exhaustive".into(),
                size_dict: inputs.iter().flatten().map(|&leg| (leg, 2)).collect(),
                inputs,
                output,
            };
            for order in [[0, 1, 2, 3], [2, 0, 3, 1]] {
                for (wf, wr) in [(1, 0), (0, 1), (1, 64)] {
                    let objective = PlannerObjective::new(wf as f64, wr as f64).unwrap();
                    let expected = all_ordered_paths(&order)
                        .iter()
                        .filter_map(|path| exact_binary_score(&net, path, wf, wr))
                        .min()
                        .unwrap();
                    let (path, _) = order_dp_with_objective(&net, &order, objective).unwrap();
                    assert_eq!(exact_binary_score(&net, &path, wf, wr), Some(expected));
                }
            }
        }
    }

    #[test]
    fn binary_order_skips_overflowing_candidates_and_intervals() {
        // Input ranks are 62, but some outer-product intervals are much larger.
        let net = binary_graph(5, &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)], 31);
        net.validate().unwrap();
        let order = [0, 2, 1, 4, 3];
        assert!(net.size_dict.len() > 128);
        for (wf, wr) in [(1, 0), (0, 1), (1, 64)] {
            let objective = PlannerObjective::new(wf as f64, wr as f64).unwrap();
            let scores: Vec<_> = all_ordered_paths(&order)
                .iter()
                .map(|path| exact_binary_score(&net, path, wf, wr))
                .collect();
            if wf != 0 {
                assert!(
                    scores.iter().any(Option::is_none),
                    "objective={objective:?}"
                );
            }
            let expected = scores.into_iter().flatten().min().unwrap();
            let integer = match (wf, wr) {
                (1, 0) => {
                    order_dp_arithmetic::<BinaryCosts, true, false>(&net, &order, None, objective)
                }
                (0, 1) => {
                    order_dp_arithmetic::<BinaryCosts, false, true>(&net, &order, None, objective)
                }
                _ => order_dp_arithmetic::<BinaryCosts, true, true>(&net, &order, None, objective),
            };
            let (path, _) = integer
                .unwrap()
                .expect("representable complete tree must survive");
            assert_eq!(exact_binary_score(&net, &path, wf, wr), Some(expected));
            #[cfg(feature = "integer-order-dp")]
            {
                let (dispatched, _) = order_dp_with_objective(&net, &order, objective).unwrap();
                assert_eq!(dispatched, path);
            }
        }
    }

    #[test]
    fn binary_order_complete_overflow_falls_back_without_changing_deadline_errors() {
        // Every internal merge above a leaf pair involves at least 132 indices.
        // All input ranks are 60, so the network itself remains valid.
        let edges: Vec<_> = (0..6)
            .flat_map(|a| (a + 1..6).map(move |b| (a, b)))
            .collect();
        let net = binary_graph(6, &edges, 12);
        net.validate().unwrap();
        let order = [0, 1, 2, 3, 4, 5];
        let objective = PlannerObjective::new(1.0, 0.0).unwrap();
        assert!(
            order_dp_arithmetic::<BinaryCosts, true, false>(&net, &order, None, objective)
                .unwrap()
                .is_none()
        );
        let expected = order_dp_arithmetic::<LogCosts, true, false>(&net, &order, None, objective)
            .unwrap()
            .unwrap();
        let actual = order_dp_with_objective(&net, &order, objective).unwrap();
        assert_eq!(actual.0, expected.0);
        assert_eq!(
            actual.1.log10_flops.to_bits(),
            expected.1.log10_flops.to_bits()
        );
        let expired = std::time::Instant::now() - std::time::Duration::from_millis(1);
        assert!(
            order_dp_cooperative_with_objective(&net, &order, Some(expired), objective)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    #[cfg(not(feature = "integer-order-dp"))]
    fn disabled_integer_order_feature_uses_original_arithmetic() {
        let mut net = open_hyperedge_net();
        net.size_dict.values_mut().for_each(|dim| *dim = 2);
        let order = [2, 0, 3, 1];
        for objective in [
            PlannerObjective::new(1.0, 0.0).unwrap(),
            PlannerObjective::new(0.0, 1.0).unwrap(),
            PlannerObjective::FIXED,
        ] {
            assert!(crate::integer_cost::supports(&net, objective));
            let expected = match objective.kind() {
                crate::objective::ObjectiveKind::TotalFlops => {
                    order_dp_arithmetic::<LogCosts, true, false>(&net, &order, None, objective)
                }
                crate::objective::ObjectiveKind::TotalReadWrite => {
                    order_dp_arithmetic::<LogCosts, false, true>(&net, &order, None, objective)
                }
                crate::objective::ObjectiveKind::Weighted => {
                    order_dp_arithmetic::<LogCosts, true, true>(&net, &order, None, objective)
                }
            }
            .unwrap()
            .unwrap();
            let actual = order_dp_with_objective(&net, &order, objective).unwrap();
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
    }

    #[test]
    fn unsupported_order_inputs_keep_the_original_arithmetic() {
        let mut binary = open_hyperedge_net();
        binary.size_dict.values_mut().for_each(|dim| *dim = 2);
        let order = [2, 0, 3, 1];
        for (net, objective) in [
            (open_hyperedge_net(), PlannerObjective::FIXED),
            (binary, PlannerObjective::new(3.0, 7.0).unwrap()),
        ] {
            assert!(!crate::integer_cost::supports(&net, objective));
            let expected =
                order_dp_arithmetic::<LogCosts, true, true>(&net, &order, None, objective)
                    .unwrap()
                    .unwrap();
            let actual = order_dp_with_objective(&net, &order, objective).unwrap();
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
    fn pure_flops_order_dp_ignores_an_overflowing_unused_interval() {
        let (net, expected) = nested_pair_network(vec![1usize << 30; 36]);
        let objective = PlannerObjective::new(1.0, 0.0).unwrap();
        let finite = simulate_path(&net, &expected).unwrap();
        assert!(objective.score_path_log2(&finite).exp2().is_finite());
        let order: Vec<_> = (0..net.n_tensors()).collect();
        let (path, stats) = order_dp_cooperative_with_objective(&net, &order, None, objective)
            .unwrap()
            .unwrap();
        assert_eq!(path.len(), expected.len());
        assert!(objective.score_path_log2(&stats) <= objective.score_path_log2(&finite) + 1e-12);
    }

    #[test]
    fn pure_flops_order_dp_ignores_an_overflowing_candidate_sum() {
        let mut dimensions = vec![1usize << 30; 33];
        dimensions.push(1usize << 33);
        let (net, expected) = nested_pair_network(dimensions);
        let objective = PlannerObjective::new(1.0, 0.0).unwrap();
        let finite = simulate_path(&net, &expected).unwrap();
        assert!(objective.score_path_log2(&finite).exp2().is_finite());
        // Every individual step is representable; only a sum can overflow.
        let all_legs_log2: f64 = net.size_dict.keys().map(|&leg| net.log2_dim(leg)).sum();
        assert_eq!(all_legs_log2, 1023.0);
        assert!(all_legs_log2.exp2().is_finite());
        let order: Vec<_> = (0..net.n_tensors()).collect();
        let (path, stats) = order_dp_with_objective(&net, &order, objective).unwrap();
        assert_eq!(path.len(), expected.len());
        assert!(objective.score_path_log2(&stats) <= objective.score_path_log2(&finite) + 1e-12);
    }

    fn nested_pair_network(dimensions: Vec<usize>) -> (TensorNetwork, SsaPath) {
        let pairs = dimensions.len();
        let net = TensorNetwork {
            name: "nested-vector-pairs".into(),
            inputs: (0..pairs)
                .chain((0..pairs).rev())
                .map(|leg| vec![leg as u32])
                .collect(),
            output: vec![],
            size_dict: dimensions
                .into_iter()
                .enumerate()
                .map(|(leg, dim)| (leg as u32, dim))
                .collect(),
        };
        let mut path = vec![(pairs - 1, pairs)];
        let mut last = 2 * pairs;
        for offset in 1..pairs {
            path.push((pairs - 1 - offset, last));
            last += 1;
            path.push((pairs + offset, last));
            last += 1;
        }
        (net, path)
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
