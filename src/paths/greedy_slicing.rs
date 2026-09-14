//! Greedy path construction with integrated slice selection.
//!
//! When an intermediate exceeds the target, prefer non-output legs by holder
//! count, dimension, and leg id. Returns an SSA path and its slice plan.

use std::collections::{HashMap, HashSet};

use crate::network::{LegId, TensorNetwork};
use crate::path::{result_legs_by, simulate_path, SsaPath};
use crate::slice::{SliceResult, SliceTarget};

/// Build a path and slice plan for a log2 intermediate-size target.
pub fn greedy_path_with_slicing(
    net: &TensorNetwork,
    target_log2_size: f64,
    costmod: f64,
) -> Option<(SsaPath, SliceResult)> {
    greedy_path_with_slicing_until(net, target_log2_size, costmod, None)
}

/// Exact element-count variant of [`greedy_path_with_slicing`].
pub fn greedy_path_with_slicing_to_size(
    net: &TensorNetwork,
    target_size: usize,
    costmod: f64,
) -> Option<(SsaPath, SliceResult)> {
    greedy_path_with_slicing_for_target_until(
        net,
        SliceTarget::from_elements(target_size)?,
        costmod,
        None,
    )
}

/// Deadline-aware version of [`greedy_path_with_slicing`].
///
/// Expiry returns `None` because a partial SSA path is not a valid result.
pub fn greedy_path_with_slicing_until(
    net: &TensorNetwork,
    target_log2_size: f64,
    costmod: f64,
    deadline: Option<std::time::Instant>,
) -> Option<(SsaPath, SliceResult)> {
    greedy_path_with_slicing_for_target_until(
        net,
        SliceTarget::LegacyLog2(target_log2_size),
        costmod,
        deadline,
    )
}

pub(crate) fn greedy_path_with_slicing_for_target_until(
    net: &TensorNetwork,
    target: SliceTarget,
    costmod: f64,
    deadline: Option<std::time::Instant>,
) -> Option<(SsaPath, SliceResult)> {
    if deadline_reached(deadline) {
        return None;
    }
    let n = net.n_tensors();
    if n == 0 {
        return None;
    }
    // Sliced legs have log2 dimension zero.
    let sliced: HashSet<LegId> = HashSet::new();
    let log2dim = |l: LegId, sliced: &HashSet<LegId>| -> f64 {
        if sliced.contains(&l) {
            0.0
        } else {
            net.log2_dim(l)
        }
    };
    // Tensor size is the sum of unique leg dimensions in log2 space.
    let node_log2size = |legs: &[LegId], sliced: &HashSet<LegId>| -> f64 {
        legs.iter().map(|&l| log2dim(l, sliced)).sum()
    };
    // Log2 targets use a strict comparison; element targets use integer arithmetic.
    let target_fits = |legs: &[LegId], sliced: &HashSet<LegId>| -> bool {
        match target {
            SliceTarget::LegacyLog2(limit) => node_log2size(legs, sliced) <= limit,
            SliceTarget::Elements(_) => target.dimensions_are_feasible(legs.iter().map(|&leg| {
                if sliced.contains(&leg) {
                    1
                } else {
                    net.dim(leg)
                }
            })),
        }
    };

    // Input SSA ids are `0..n`; contracted nodes follow in order.
    let mut node_legs: Vec<Option<Vec<LegId>>> = Vec::with_capacity(2 * n);
    for legs in &net.inputs {
        let mut v = legs.clone();
        v.sort_unstable();
        v.dedup();
        node_legs.push(Some(v));
    }
    // Map each leg to its live SSA holders.
    let mut holders: HashMap<LegId, HashSet<usize>> = HashMap::new();
    for (id, legs) in node_legs.iter().enumerate() {
        if let Some(legs) = legs {
            for &l in legs {
                holders.entry(l).or_default().insert(id);
            }
        }
    }
    let in_output: HashSet<LegId> = net.output.iter().copied().collect();

    let mut sliced = sliced;
    let mut path: SsaPath = Vec::with_capacity(n.saturating_sub(1));
    let mut n_alive = n;

    // Recompute the best pair among live tensors sharing at least one leg.
    while n_alive > 1 {
        if deadline_reached(deadline) {
            return None;
        }
        let mut best: Option<(f64, usize, usize)> = None;
        let mut seen_pairs: HashSet<(usize, usize)> = HashSet::new();
        for (holder_i, holder_set) in holders.values().enumerate() {
            if holder_i % 64 == 0 && deadline_reached(deadline) {
                return None;
            }
            if holder_set.len() < 2 {
                continue;
            }
            let ids: Vec<usize> = holder_set.iter().copied().collect();
            for i in 0..ids.len() {
                if i % 64 == 0 && deadline_reached(deadline) {
                    return None;
                }
                for j in (i + 1)..ids.len() {
                    let (a, b) = if ids[i] < ids[j] {
                        (ids[i], ids[j])
                    } else {
                        (ids[j], ids[i])
                    };
                    if !seen_pairs.insert((a, b)) {
                        continue;
                    }
                    let la = node_legs[a].as_ref().unwrap();
                    let lb = node_legs[b].as_ref().unwrap();
                    let rlegs = result_legs_by(la, lb, |l| holders[&l].len(), &in_output);
                    let sa = node_log2size(la, &sliced);
                    let sb = node_log2size(lb, &sliced);
                    let sr = node_log2size(&rlegs, &sliced);
                    // Compare `size_r - costmod * (size_a + size_b)` in linear space.
                    let cost = pow2(sr) - costmod * (pow2(sa) + pow2(sb));
                    // Break ties by SSA ids for deterministic iteration.
                    let better = match &best {
                        None => true,
                        Some((bc, ba, bb)) => {
                            cost.total_cmp(bc).then(a.cmp(ba)).then(b.cmp(bb))
                                == std::cmp::Ordering::Less
                        }
                    };
                    if better {
                        best = Some((cost, a, b));
                    }
                }
            }
        }
        // Outer-product the two smallest tensors when no pair shares a leg.
        if deadline_reached(deadline) {
            return None;
        }
        let (a, b) = match best {
            Some((_, a, b)) => (a, b),
            None => {
                let mut alive: Vec<usize> = (0..node_legs.len())
                    .filter(|&i| node_legs[i].is_some())
                    .collect();
                alive.sort_by(|&x, &y| {
                    node_log2size(node_legs[x].as_ref().unwrap(), &sliced)
                        .total_cmp(&node_log2size(node_legs[y].as_ref().unwrap(), &sliced))
                        .then(x.cmp(&y))
                });
                (alive[0].min(alive[1]), alive[0].max(alive[1]))
            }
        };

        let la = node_legs[a].take().unwrap();
        let lb = node_legs[b].take().unwrap();
        let rlegs = result_legs_by(&la, &lb, |l| holders[&l].len(), &in_output);
        for &l in &la {
            if let Some(h) = holders.get_mut(&l) {
                h.remove(&a);
            }
        }
        for &l in &lb {
            if let Some(h) = holders.get_mut(&l) {
                h.remove(&b);
            }
        }
        let new_id = node_legs.len();
        // Slice result legs until the target is met.
        while !target_fits(&rlegs, &sliced) {
            if deadline_reached(deadline) {
                return None;
            }
            let pick = rlegs
                .iter()
                .filter(|&&l| {
                    !sliced.contains(&l)
                        && !in_output.contains(&l)
                        && log2dim(l, &sliced) > 0.0
                        && crate::slice::slice_leg_is_executable(net, l)
                })
                .max_by(|&&x, &&y| {
                    let hx = holders.get(&x).map(|h| h.len()).unwrap_or(0);
                    let hy = holders.get(&y).map(|h| h.len()).unwrap_or(0);
                    // Reverse the final comparison because `max_by` keeps the later tie.
                    hx.cmp(&hy)
                        .then(net.log2_dim(x).total_cmp(&net.log2_dim(y)))
                        .then(y.cmp(&x))
                })
                .copied();
            match pick {
                Some(l) => {
                    sliced.insert(l);
                }
                None => break, // Final validation rejects an unreachable target.
            }
        }
        for &l in &rlegs {
            holders.entry(l).or_default().insert(new_id);
        }
        node_legs.push(Some(rlegs));
        path.push((a, b));
        n_alive -= 1;
    }

    // Recompute authoritative metrics with sliced dimensions set to one.
    if deadline_reached(deadline) {
        return None;
    }
    let mut sliced_vec: Vec<LegId> = sliced.into_iter().collect();
    sliced_vec.sort_unstable();
    let mut cur = net.clone();
    for &l in &sliced_vec {
        cur.size_dict.insert(l, 1);
    }
    if deadline_reached(deadline) {
        return None;
    }
    let stats = simulate_path(&cur, &path).ok()?;
    // Reject targets that remain infeasible after all executable legs are sliced.
    let log2_n: f64 = sliced_vec.iter().map(|&l| net.log2_dim(l)).sum();
    let log10_total = stats.log10_flops + log2_n * std::f64::consts::LOG10_2;
    let result = SliceResult {
        legs: sliced_vec,
        log2_n_slices: log2_n,
        per_slice: stats,
        log10_flops_total: log10_total,
    };
    let feasible = match target {
        SliceTarget::LegacyLog2(limit) => result.per_slice.log2_max_size <= limit,
        SliceTarget::Elements(limit) => {
            crate::slice::slice_result_fits_target_size(net, &path, &result, limit.get()).ok()?
        }
    };
    if !feasible {
        return None;
    }
    Some((path, result))
}

#[inline]
fn deadline_reached(deadline: Option<std::time::Instant>) -> bool {
    deadline
        .map(|d| std::time::Instant::now() >= d)
        .unwrap_or(false)
}

#[inline]
fn pow2(log2: f64) -> f64 {
    2f64.powf(log2)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Three-tensor chain with output legs at both ends.
    fn toy(dim: usize) -> TensorNetwork {
        let mut size_dict = HashMap::new();
        for l in 0..4u32 {
            size_dict.insert(l, dim);
        }
        TensorNetwork {
            name: "toy".into(),
            inputs: vec![vec![0, 1], vec![1, 2], vec![2, 3]],
            output: vec![0, 3],
            size_dict,
        }
    }

    #[test]
    fn expired_deadline_does_not_start_greedy_path_with_slicing() {
        let net = toy(2);
        let expired = std::time::Instant::now() - std::time::Duration::from_millis(1);
        assert!(greedy_path_with_slicing_until(&net, 4.0, 1.0, Some(expired)).is_none());
    }

    #[test]
    fn greedy_path_with_slicing_produces_valid_path_under_target() {
        let net = toy(4);
        // The target fits without slicing.
        let (path, sr) = greedy_path_with_slicing(&net, 4.0, 1.0).unwrap();
        assert_eq!(path.len(), 2);
        assert!(
            sr.per_slice.log2_max_size <= 4.0 + 1e-9,
            "峰值超标: {}",
            sr.per_slice.log2_max_size
        );
    }

    #[test]
    fn greedy_path_with_slicing_hits_tight_target() {
        // A symmetric four-tensor network forces slicing under a tight target.
        let mut size_dict = HashMap::new();
        for l in 0..6u32 {
            size_dict.insert(l, 2);
        }
        let net = TensorNetwork {
            name: "loop".into(),
            inputs: vec![vec![0, 1, 2], vec![2, 3, 4], vec![4, 5, 0], vec![1, 3, 5]],
            output: vec![],
            size_dict,
        };
        let (path, sr) = greedy_path_with_slicing(&net, 1.0, 1.0).unwrap();
        assert!(
            sr.per_slice.log2_max_size <= 1.0 + 1e-9,
            "峰值未达标: {}",
            sr.per_slice.log2_max_size
        );
        assert!(!sr.legs.is_empty(), "紧目标应触发切片");
        assert_eq!(path.len(), 3);
    }

    /// Disconnected components complete through outer products.
    #[test]
    fn greedy_path_with_slicing_handles_disconnected_components() {
        let mut size_dict = HashMap::new();
        for l in 0..4u32 {
            size_dict.insert(l, 2);
        }
        let net = TensorNetwork {
            name: "disconnected".into(),
            inputs: vec![vec![0, 1], vec![0, 1], vec![2, 3], vec![2, 3]],
            output: vec![],
            size_dict,
        };
        let (path, sr) = greedy_path_with_slicing(&net, 8.0, 1.0).expect("不连通网也必须收完");
        assert_eq!(path.len(), 3, "4 张量应收 3 次");
        assert!(sr.per_slice.log10_flops.is_finite());
    }

    /// Repeated calls in one process are deterministic.
    #[test]
    fn greedy_path_with_slicing_is_deterministic_across_calls() {
        let mut size_dict = HashMap::new();
        for l in 0..8u32 {
            size_dict.insert(l, 2);
        }
        // Symmetry creates many equal-cost pairs.
        let net = TensorNetwork {
            name: "sym".into(),
            inputs: vec![
                vec![0, 1, 2],
                vec![2, 3, 4],
                vec![4, 5, 6],
                vec![6, 7, 0],
                vec![1, 3, 5],
                vec![7, 2, 4],
            ],
            output: vec![],
            size_dict,
        };
        let (p0, s0) = greedy_path_with_slicing(&net, 3.0, 1.0).expect("首次");
        for k in 1..12 {
            let (p, s) = greedy_path_with_slicing(&net, 3.0, 1.0).expect("重复调用");
            assert_eq!(p, p0, "第 {k} 次路径不同 → 迭代序泄漏");
            assert_eq!(s.legs, s0.legs, "第 {k} 次切腿集不同");
            assert_eq!(
                s.log10_flops_total.to_bits(),
                s0.log10_flops_total.to_bits(),
                "第 {k} 次总 flops 不逐位相同"
            );
        }
    }

    #[test]
    fn greedy_path_with_slicing_target_reachable_matches_simulate() {
        // Without slicing, per-slice metrics match direct simulation.
        let net = toy(2);
        let (path, sr) = greedy_path_with_slicing(&net, 10.0, 1.0).unwrap();
        let direct = simulate_path(&net, &path).unwrap();
        assert!(sr.legs.is_empty());
        assert!((sr.per_slice.log10_flops - direct.log10_flops).abs() < 1e-9);
        assert!((sr.per_slice.log2_max_size - direct.log2_max_size).abs() < 1e-9);
    }

    /// Return `None` when only output legs remain and the target is still infeasible.
    #[test]
    fn greedy_path_with_slicing_returns_none_when_output_only_result_exceeds_target() {
        let mut size_dict = HashMap::new();
        size_dict.insert(0, 2);
        size_dict.insert(1, 2);
        let net = TensorNetwork {
            name: "output_only".into(),
            inputs: vec![vec![0], vec![1]],
            output: vec![0, 1],
            size_dict,
        };
        assert!(greedy_path_with_slicing(&net, 1.0, 1.0).is_none());
    }
}
