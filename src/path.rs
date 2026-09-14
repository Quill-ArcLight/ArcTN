//! Contraction-path representation and cost metrics.
//!
//! Paths use SSA numbering: inputs are `0..n`, and step `k` creates node `n+k`.
//! Metrics follow cotengra: scalar multiplications, largest result, largest
//! two-input-plus-output step, and peak live elements.

use std::collections::HashMap;

use crate::network::{LegId, TensorNetwork};

pub type SsaPath = Vec<(usize, usize)>;

#[derive(Clone, Copy, Debug)]
pub struct PathStats {
    /// Log10 total scalar multiplications.
    pub log10_flops: f64,
    /// Log2 largest contraction result. One-tensor networks use the final root size.
    pub log2_max_size: f64,
    /// Log2 maximum `|A| + |B| + |C|` across binary contractions.
    /// Excludes backend workspace, concurrent slices, and allocator state.
    pub log2_max_contraction_size: f64,
    /// Log2 sum of all intermediate result elements.
    pub log2_total_size: f64,
    /// Log2 sum of `|A| + |B| + |C|` over binary contractions.
    /// This is logical read/write complexity, not measured traffic.
    pub log2_read_write: f64,
    /// Log2 peak live elements, counting both operands and the new result.
    pub log2_peak_size: f64,
}

/// Stable log2-domain addition.
pub fn logaddexp2(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b; // 2^-infinity is zero, so -infinity is the additive identity here.
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let (hi, lo) = if a > b { (a, b) } else { (b, a) };
    // Factor out the larger term so exponentiation never uses a positive exponent.
    hi + (1.0 + (lo - hi).exp2()).log2()
}

/// Linear-time union of sorted, deduplicated leg lists.
pub fn legs_union(a: &[LegId], b: &[LegId]) -> Vec<LegId> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => {
                out.push(a[i]);
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                out.push(b[j]);
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                out.push(a[i]);
                i += 1;
                j += 1;
            }
        }
    }
    out.extend_from_slice(&a[i..]);
    out.extend_from_slice(&b[j..]);
    out
}

/// Computes legs retained after contracting two tensors.
///
/// Inputs must be sorted and deduplicated. A leg remains if another live tensor
/// holds it or if it is an output leg.
pub fn result_legs_by<F: Fn(LegId) -> usize>(
    la: &[LegId],
    lb: &[LegId],
    refcount_of: F,
    in_output: &std::collections::HashSet<LegId>,
) -> Vec<LegId> {
    legs_union(la, lb)
        .into_iter()
        .filter(|l| {
            // Exclude the two contracted tensors from the holder count.
            let mut rc = refcount_of(*l);
            if la.binary_search(l).is_ok() {
                rc -= 1;
            }
            if lb.binary_search(l).is_ok() {
                rc -= 1;
            }
            rc > 0 || in_output.contains(l)
        })
        .collect()
}

/// Convenience wrapper for a `HashMap` reference count.
pub fn result_legs(
    la: &[LegId],
    lb: &[LegId],
    refcount: &HashMap<LegId, usize>,
    in_output: &std::collections::HashSet<LegId>,
) -> Vec<LegId> {
    result_legs_by(la, lb, |l| refcount[&l], in_output)
}

/// Sorts and deduplicates a tensor's leg list.
pub fn sorted_dedup(legs: &[LegId]) -> Vec<LegId> {
    let mut v = legs.to_vec();
    v.sort_unstable();
    v.dedup();
    v
}

/// Validates and simulates an SSA path, returning its metrics.
pub fn simulate_path(net: &TensorNetwork, path: &SsaPath) -> Result<PathStats, String> {
    simulate_path_full(net, path).map(|(s, _)| s)
}

/// Validate and replay an SSA path while computing only `log2(total FLOPs)`
/// and the result legs needed to rebuild a contraction tree. Complete metrics
/// remain the responsibility of [`simulate_path_full`] at reporting boundaries.
pub(crate) fn simulate_path_flops_log2_and_legs(
    net: &TensorNetwork,
    path: &SsaPath,
) -> Result<(f64, Vec<Vec<LegId>>), String> {
    simulate_path_legs_impl::<true>(net, path)
}

/// Validate and replay an SSA path while computing only the result legs.
/// Pure read/write search uses this boundary to avoid accumulating a FLOPs
/// total that cannot affect its objective.
pub(crate) fn simulate_path_legs(
    net: &TensorNetwork,
    path: &SsaPath,
) -> Result<Vec<Vec<LegId>>, String> {
    simulate_path_legs_impl::<false>(net, path).map(|(_, legs)| legs)
}

fn simulate_path_legs_impl<const TRACK_FLOPS: bool>(
    net: &TensorNetwork,
    path: &SsaPath,
) -> Result<(f64, Vec<Vec<LegId>>), String> {
    net.validate()?;
    let n = net.n_tensors();
    let n_slots = n + path.len();
    let mut legs: Vec<Option<Vec<LegId>>> = Vec::with_capacity(n_slots);
    let mut refcount: HashMap<LegId, usize> = HashMap::new();
    let mut in_output: HashMap<LegId, bool> = HashMap::new();
    for &leg in &net.output {
        in_output.insert(leg, true);
    }
    let mut output = net.output.clone();
    output.sort_unstable();
    if output.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(format!("output 含重复腿，暂不支持: {:?}", net.output));
    }

    for tensor in &net.inputs {
        let mut counts: HashMap<LegId, usize> = HashMap::new();
        for &leg in tensor {
            *counts.entry(leg).or_insert(0) += 1;
        }
        if let Some((&leg, _)) = counts.iter().find(|(_, count)| **count > 2) {
            return Err(format!("腿 {leg} 在同一张量出现 >2 次，暂不支持"));
        }
        let deduplicated = sorted_dedup(tensor);
        for &leg in &deduplicated {
            *refcount.entry(leg).or_insert(0) += 1;
        }
        legs.push(Some(deduplicated));
    }

    let mut log2_flops = f64::NEG_INFINITY;
    let mut step_legs = Vec::with_capacity(path.len());
    for (step, &(a, b)) in path.iter().enumerate() {
        if a == b {
            return Err(format!("第 {step} 步自收缩 ({a},{b})"));
        }
        let left = legs
            .get(a)
            .and_then(|value| value.clone())
            .ok_or(format!("第 {step} 步引用了无效/已消费的张量 {a}"))?;
        let right = legs
            .get(b)
            .and_then(|value| value.clone())
            .ok_or(format!("第 {step} 步引用了无效/已消费的张量 {b}"))?;
        legs[a] = None;
        legs[b] = None;

        let union = legs_union(&left, &right);
        if TRACK_FLOPS {
            let step_log2 = union.iter().map(|&leg| net.log2_dim(leg)).sum();
            log2_flops = logaddexp2(log2_flops, step_log2);
        }
        for &leg in &left {
            *refcount.get_mut(&leg).unwrap() -= 1;
        }
        for &leg in &right {
            *refcount.get_mut(&leg).unwrap() -= 1;
        }
        let result: Vec<LegId> = union
            .into_iter()
            .filter(|leg| refcount[leg] > 0 || *in_output.get(leg).unwrap_or(&false))
            .collect();
        for &leg in &result {
            *refcount.get_mut(&leg).unwrap() += 1;
        }
        step_legs.push(result.clone());
        legs.push(Some(result));
    }

    let alive: Vec<usize> = (0..n_slots)
        .filter(|&index| legs[index].is_some())
        .collect();
    if alive.len() != 1 {
        return Err(format!("路径不完整：剩余 {} 个张量", alive.len()));
    }
    let final_legs = sorted_dedup(legs[alive[0]].as_ref().unwrap());
    let expected_output = sorted_dedup(&net.output);
    if path.is_empty() {
        if !expected_output
            .iter()
            .all(|leg| final_legs.binary_search(leg).is_ok())
        {
            return Err(format!(
                "最终腿 {final_legs:?} 不含全部 output {expected_output:?}"
            ));
        }
    } else if final_legs != expected_output {
        return Err(format!(
            "最终腿 {final_legs:?} 与 output {expected_output:?} 不符"
        ));
    }

    Ok((log2_flops, step_legs))
}

/// Simulates an SSA path and also returns each step's result legs.
pub fn simulate_path_full(
    net: &TensorNetwork,
    path: &SsaPath,
) -> Result<(PathStats, Vec<Vec<LegId>>), String> {
    // Validate at this public fallible boundary before indexed metric calculations.
    net.validate()?;
    let n = net.n_tensors();
    let n_slots = n + path.len();
    // Each SSA slot is live until consumed; new nodes append to the list.
    let mut legs: Vec<Option<Vec<LegId>>> = Vec::with_capacity(n_slots);
    // Count live tensor holders; output membership is tracked separately.
    let mut refcount: HashMap<LegId, usize> = HashMap::new();
    let mut in_output: HashMap<LegId, bool> = HashMap::new();
    for &l in &net.output {
        in_output.insert(l, true);
    }
    // Repeated output legs cannot be represented as a final axis permutation.
    {
        let mut od = net.output.clone();
        od.sort_unstable();
        if od.windows(2).any(|w| w[0] == w[1]) {
            return Err(format!("output 含重复腿，暂不支持: {:?}", net.output));
        }
    }
    // Cotengra measures leaf size after unary reductions and diagonal extraction.
    // Count appearances once to compute those logical sizes without per-leaf maps.
    let mut appearances: HashMap<LegId, usize> = HashMap::new();
    for term in &net.inputs {
        for &leg in term {
            *appearances.entry(leg).or_insert(0) += 1;
        }
    }
    for &leg in &net.output {
        *appearances.entry(leg).or_insert(0) += 1;
    }
    let mut contraction_slot_log2: Vec<f64> = Vec::with_capacity(n_slots);
    let mut read_write_slot_log2: Vec<f64> = Vec::with_capacity(n_slots);

    // Initialize deduplicated leg slots and holder counts.
    for t in &net.inputs {
        let mut d = t.clone();
        d.sort_unstable();
        let mut leaf_log2 = 0.0;
        let mut i = 0;
        while i < d.len() {
            let leg = d[i];
            let mut j = i + 1;
            while j < d.len() && d[j] == leg {
                j += 1;
            }
            if j - i != appearances[&leg] {
                leaf_log2 += net.log2_dim(leg);
            }
            i = j;
        }
        d.dedup(); // Retain each traced or diagonal leg once.
        for &l in &d {
            *refcount.entry(l).or_insert(0) += 1;
        }
        legs.push(Some(d));
        contraction_slot_log2.push(leaf_log2);
        read_write_slot_log2.push(t.iter().map(|&l| net.log2_dim(l)).sum());
    }
    // Match execution: handle a repeated leg as trace or diagonal, and reject higher multiplicity.
    for (i, t) in net.inputs.iter().enumerate() {
        if t.len() == legs[i].as_ref().unwrap().len() {
            continue; // No repeated legs.
        }
        let mut counts: HashMap<LegId, usize> = HashMap::new();
        for &l in t {
            *counts.entry(l).or_insert(0) += 1;
        }
        for (&l, &c) in &counts {
            if c > 2 {
                return Err(format!("腿 {l} 在同一张量出现 >2 次，暂不支持"));
            }
        }
    }

    // Log-domain accumulators avoid overflow; negative infinity represents zero.
    let mut log2_flops_total = f64::NEG_INFINITY;
    // Cotengra reports the root size for a one-tensor tree but does not add an intermediate.
    let mut log2_max_size: f64 = if n == 1 {
        net.output.iter().map(|&l| net.log2_dim(l)).sum()
    } else {
        0.0
    };
    // A one-tensor network has no binary contraction.
    let mut log2_max_contraction_size: f64 = 0.0;
    let mut log2_total_size = f64::NEG_INFINITY;
    let mut log2_read_write = f64::NEG_INFINITY; // Sum of both inputs and the output per step.
    let mut step_legs: Vec<Vec<LegId>> = Vec::with_capacity(path.len());

    // Track the live element total in linear space because it needs subtraction.
    let size_of =
        |legs: &[LegId]| -> f64 { legs.iter().map(|&l| net.log2_dim(l)).sum::<f64>().exp2() };
    // Initial dense buffers include repeated axes before trace extraction. Holder
    // logic still uses deduplicated legs, but peak residency uses original shapes.
    let mut slot_size: Vec<f64> = net.inputs.iter().map(|t| size_of(t)).collect();
    let mut alive_sum: f64 = slot_size.iter().sum();
    let mut peak: f64 = alive_sum;

    // Simulate each contraction step.
    for (step, &(a, b)) in path.iter().enumerate() {
        if a == b {
            return Err(format!("第 {step} 步自收缩 ({a},{b})"));
        }
        // Clone both live leg lists; consumed or out-of-range slots are invalid.
        let la = legs
            .get(a)
            .and_then(|x| x.clone())
            .ok_or(format!("第 {step} 步引用了无效/已消费的张量 {a}"))?;
        let lb = legs
            .get(b)
            .and_then(|x| x.clone())
            .ok_or(format!("第 {step} 步引用了无效/已消费的张量 {b}"))?;
        legs[a] = None; // Consume both SSA operands.
        legs[b] = None;

        let union = legs_union(&la, &lb);
        // Step FLOPs equal the product of dimensions in the union.
        let step_log2: f64 = union.iter().map(|&l| net.log2_dim(l)).sum();
        log2_flops_total = logaddexp2(log2_flops_total, step_log2);

        // Remove the two operands from holder counts.
        for &l in &la {
            *refcount.get_mut(&l).unwrap() -= 1;
        }
        for &l in &lb {
            *refcount.get_mut(&l).unwrap() -= 1;
        }
        // Keep legs held elsewhere or required by the output.
        let result: Vec<LegId> = union
            .iter()
            .copied()
            .filter(|l| refcount[l] > 0 || *in_output.get(l).unwrap_or(&false))
            .collect();
        // Add the result as a holder of its legs.
        for &l in &result {
            *refcount.get_mut(&l).unwrap() += 1;
        }
        let size_log2: f64 = result.iter().map(|&l| net.log2_dim(l)).sum(); // Log2 output size.
        log2_max_size = log2_max_size.max(size_log2);

        // Cotengra's step metric counts both logical inputs plus the output.
        let left_log2 = contraction_slot_log2[a];
        let right_log2 = contraction_slot_log2[b];
        let contraction_log2 = logaddexp2(logaddexp2(left_log2, right_log2), size_log2);
        log2_max_contraction_size = log2_max_contraction_size.max(contraction_log2);
        let read_write_log2 = logaddexp2(
            logaddexp2(read_write_slot_log2[a], read_write_slot_log2[b]),
            size_log2,
        );
        log2_read_write = logaddexp2(log2_read_write, read_write_log2);
        contraction_slot_log2.push(size_log2);
        read_write_slot_log2.push(size_log2);
        log2_total_size = logaddexp2(log2_total_size, size_log2);
        let rsize = size_log2.exp2();
        // The result is allocated before either operand is released.
        peak = peak.max(alive_sum + rsize);
        alive_sum += rsize - slot_size[a] - slot_size[b];
        slot_size.push(rsize); // Result tensor at the next SSA index.
        step_legs.push(result.clone());
        legs.push(Some(result)); // Insert the result at index n + step.
    }

    // A complete path must leave exactly one live tensor.
    let alive: Vec<usize> = (0..n_slots).filter(|&i| legs[i].is_some()).collect();
    if alive.len() != 1 {
        return Err(format!("路径不完整：剩余 {} 个张量", alive.len()));
    }
    // Validate final legs against the declared output.
    let fin = sorted_dedup(legs[alive[0]].as_ref().unwrap());
    let out = sorted_dedup(&net.output);
    if path.is_empty() {
        // A one-tensor path may retain reduction legs, but must contain every output leg.
        if !out.iter().all(|l| fin.binary_search(l).is_ok()) {
            return Err(format!("最终腿 {fin:?} 不含全部 output {out:?}"));
        }
    } else if fin != out {
        // Non-empty paths must finish with exactly the output legs.
        return Err(format!("最终腿 {fin:?} 与 output {out:?} 不符"));
    }

    Ok((
        PathStats {
            log10_flops: log2_flops_total * std::f64::consts::LOG10_2,
            log2_max_size,
            log2_max_contraction_size,
            log2_total_size,
            log2_read_write,
            log2_peak_size: peak.log2(),
        },
        step_legs,
    ))
}

#[cfg(test)]
mod tests {
    use super::{
        simulate_path, simulate_path_flops_log2_and_legs, simulate_path_full, simulate_path_legs,
        SsaPath,
    };
    use crate::network::TensorNetwork;

    fn assert_lightweight_replay_matches_full(net: &TensorNetwork, path: &SsaPath) {
        let (log2_flops, step_legs) =
            simulate_path_flops_log2_and_legs(net, path).expect("轻量重放应成功");
        let (stats, full_step_legs) = simulate_path_full(net, path).expect("完整重放应成功");
        assert_eq!(step_legs, full_step_legs);
        assert_eq!(
            simulate_path_legs(net, path).expect("腿表重放应成功"),
            full_step_legs
        );
        assert_eq!(
            (log2_flops * std::f64::consts::LOG10_2).to_bits(),
            stats.log10_flops.to_bits()
        );
    }

    #[test]
    fn flops_only_replay_matches_full_replay() {
        let connected = TensorNetwork {
            name: "connected-replay".into(),
            inputs: vec![vec![0, 1], vec![1, 2], vec![2, 3]],
            output: vec![0, 3],
            size_dict: [(0, 2), (1, 3), (2, 5), (3, 7)].into_iter().collect(),
        };
        assert_lightweight_replay_matches_full(&connected, &vec![(0, 1), (3, 2)]);

        let disconnected = TensorNetwork {
            name: "disconnected-replay".into(),
            inputs: vec![vec![0], vec![0], vec![1], vec![1]],
            output: vec![],
            size_dict: [(0, 3), (1, 5)].into_iter().collect(),
        };
        assert_lightweight_replay_matches_full(&disconnected, &vec![(0, 1), (2, 3), (4, 5)]);

        let traced = TensorNetwork {
            name: "traced-replay".into(),
            inputs: vec![vec![0, 0]],
            output: vec![],
            size_dict: [(0, 7)].into_iter().collect(),
        };
        assert_lightweight_replay_matches_full(&traced, &SsaPath::new());
    }

    #[test]
    fn flops_only_replay_rejects_the_same_invalid_paths() {
        let net = TensorNetwork {
            name: "invalid-replay".into(),
            inputs: vec![vec![0], vec![0]],
            output: vec![],
            size_dict: [(0, 2)].into_iter().collect(),
        };
        for path in [vec![(0, 0)], vec![(0, 2)], SsaPath::new()] {
            assert!(simulate_path_flops_log2_and_legs(&net, &path).is_err());
            assert!(simulate_path_full(&net, &path).is_err());
        }

        let three_inputs = TensorNetwork {
            name: "repeated-consumption".into(),
            inputs: vec![vec![0], vec![0, 1], vec![1]],
            output: vec![],
            size_dict: [(0, 2), (1, 3)].into_iter().collect(),
        };
        let repeated = vec![(0, 1), (0, 2)];
        assert!(simulate_path_flops_log2_and_legs(&three_inputs, &repeated).is_err());
        assert!(simulate_path_full(&three_inputs, &repeated).is_err());

        let unsupported_trace = TensorNetwork {
            name: "unsupported-trace".into(),
            inputs: vec![vec![0, 0, 0]],
            output: vec![],
            size_dict: [(0, 2)].into_iter().collect(),
        };
        assert!(simulate_path_flops_log2_and_legs(&unsupported_trace, &SsaPath::new()).is_err());
        assert!(simulate_path_full(&unsupported_trace, &SsaPath::new()).is_err());
    }

    #[test]
    fn simulation_rejects_malformed_network_before_cost_accounting() {
        let missing_dim = TensorNetwork {
            name: "missing-dim".into(),
            inputs: vec![vec![0]],
            output: vec![0],
            size_dict: Default::default(),
        };
        let err = simulate_path_full(&missing_dim, &SsaPath::new())
            .expect_err("缺维网络应返回 Err，而不是在 cost/refcount 中 panic");
        assert!(err.contains("缺少 size_dict"), "unexpected error: {err}");

        let zero_dim = TensorNetwork {
            name: "zero-dim".into(),
            inputs: vec![vec![0]],
            output: vec![0],
            size_dict: [(0, 0)].into_iter().collect(),
        };
        let err = simulate_path_full(&zero_dim, &SsaPath::new())
            .expect_err("零维网络应在 cost 计算前被拒绝");
        assert!(err.contains("维度为 0"), "unexpected error: {err}");
    }

    #[test]
    fn single_tensor_uses_output_root_and_full_input_residency() {
        // The root has four elements while the dense input occupies 32 elements.
        let net = TensorNetwork {
            name: "single-root".into(),
            inputs: vec![vec![0, 1]],
            output: vec![0],
            size_dict: [(0, 4), (1, 8)].into_iter().collect(),
        };
        let stats = simulate_path(&net, &SsaPath::new()).expect("单张量网络应可模拟");
        assert_eq!(stats.log2_max_size, 2.0);
        assert_eq!(stats.log2_max_contraction_size, 0.0);
        assert_eq!(stats.log2_peak_size, 5.0);
        assert_eq!(stats.log10_flops, f64::NEG_INFINITY);
        assert_eq!(stats.log2_total_size, f64::NEG_INFINITY);
    }

    #[test]
    fn max_contraction_size_is_two_inputs_plus_output() {
        // A[2,3] x B[3,5] -> C[2,5] uses 6 + 15 + 10 = 31 logical elements.
        let net = TensorNetwork {
            name: "max-contraction-size".into(),
            inputs: vec![vec![0, 1], vec![1, 2]],
            output: vec![0, 2],
            size_dict: [(0, 2), (1, 3), (2, 5)].into_iter().collect(),
        };
        let stats = simulate_path(&net, &vec![(0, 1)]).expect("二元路径应可模拟");
        assert!((stats.log2_max_contraction_size - 31.0_f64.log2()).abs() < 1e-12);
        assert!((stats.log2_read_write - 31.0_f64.log2()).abs() < 1e-12);
        assert!((stats.log2_total_size - 10.0_f64.log2()).abs() < 1e-12);
        assert!((stats.log2_max_size - 10.0_f64.log2()).abs() < 1e-12);
        assert!((stats.log2_peak_size - 31.0_f64.log2()).abs() < 1e-12);
    }

    #[test]
    fn read_write_complexity_accumulates_inputs_and_outputs_across_steps() {
        let net = TensorNetwork {
            name: "read-write-complexity".into(),
            inputs: vec![vec![0, 1], vec![1, 2], vec![2, 3]],
            output: vec![0, 3],
            size_dict: [(0, 2), (1, 3), (2, 5), (3, 7)].into_iter().collect(),
        };
        let stats = simulate_path(&net, &vec![(0, 1), (3, 2)]).expect("路径应可模拟");
        assert!((stats.log2_read_write - 90.0_f64.log2()).abs() < 1e-12);
        assert!((stats.log2_total_size - 24.0_f64.log2()).abs() < 1e-12);
    }

    #[test]
    fn max_contraction_size_uses_cotengra_leaf_preprocessing() {
        // Cotengra removes a private reduction leg before measuring leaf size.
        let dangling = TensorNetwork {
            name: "leaf-sum-preprocessing".into(),
            inputs: vec![vec![0, 1], vec![1, 2]],
            output: vec![2],
            size_dict: [(0, 7), (1, 3), (2, 5)].into_iter().collect(),
        };
        let stats = simulate_path(&dangling, &vec![(0, 1)]).expect("待求和腿网络应可模拟");
        assert!((stats.log2_max_contraction_size - 23.0_f64.log2()).abs() < 1e-12);
        assert!((stats.log2_read_write - 41.0_f64.log2()).abs() < 1e-12);

        // A private repeated leg is traced before measuring the logical leaf.
        let traced = TensorNetwork {
            name: "leaf-trace-preprocessing".into(),
            inputs: vec![vec![0, 0, 1], vec![1, 2]],
            output: vec![2],
            size_dict: [(0, 7), (1, 3), (2, 5)].into_iter().collect(),
        };
        let traced_stats = simulate_path(&traced, &vec![(0, 1)]).expect("内部 trace 网络应可模拟");
        assert!((traced_stats.log2_max_contraction_size - 23.0_f64.log2()).abs() < 1e-12);
        assert!((traced_stats.log2_read_write - 167.0_f64.log2()).abs() < 1e-12);
    }

    #[test]
    fn max_contraction_size_handles_diagonal_hyperedge_and_sliced_dims() {
        // a,a is a diagonal rather than an internal trace because another input/output
        // still references a. Logical sizes are 21 + 105 + 35 = 161.
        let diagonal = TensorNetwork {
            name: "leaf-diagonal-preprocessing".into(),
            inputs: vec![vec![0, 0, 1], vec![0, 1, 2]],
            output: vec![0, 2],
            size_dict: [(0, 7), (1, 3), (2, 5)].into_iter().collect(),
        };
        let diagonal_stats = simulate_path(&diagonal, &vec![(0, 1)]).expect("对角网络应可模拟");
        assert!((diagonal_stats.log2_max_contraction_size - 161.0_f64.log2()).abs() < 1e-12);

        // h is shared by three inputs. The first contraction keeps it for the third input:
        // max(6 + 15 + 30, 30 + 21 + 70) = 121.
        let hyper = TensorNetwork {
            name: "leaf-hyperedge-preprocessing".into(),
            inputs: vec![vec![0, 1], vec![1, 2], vec![1, 3]],
            output: vec![0, 2, 3],
            size_dict: [(0, 2), (1, 3), (2, 5), (3, 7)].into_iter().collect(),
        };
        let path = vec![(0, 1), (3, 2)];
        let first = simulate_path(&hyper, &path).expect("超边网络应可模拟");
        let second = simulate_path(&hyper, &path).expect("重复模拟应可复现");
        assert!((first.log2_max_contraction_size - 121.0_f64.log2()).abs() < 1e-12);
        assert_eq!(
            first.log2_max_contraction_size.to_bits(),
            second.log2_max_contraction_size.to_bits(),
            "非二次幂维度的累加顺序必须逐位稳定"
        );

        // A sliced h has logical dimension one, equivalent to removing it from every
        // leaf for this metric: max(2 + 5 + 10, 10 + 7 + 70) = 87.
        let mut sliced = hyper.clone();
        sliced.size_dict.insert(1, 1);
        let sliced_stats = simulate_path(&sliced, &path).expect("切片超边网络应可模拟");
        assert!((sliced_stats.log2_max_contraction_size - 87.0_f64.log2()).abs() < 1e-12);
    }
}
