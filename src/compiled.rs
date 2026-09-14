//! Precompiled contraction plans.
//!
//! Compilation records reduction axes, permutations, GEMM dimensions, and result
//! shapes. Execution reuses these data-independent mappings. Stripped execution
//! separates a decimal exponent to reduce overflow risk.

use std::collections::{HashMap, HashSet};

use crate::contract::{checked_batch_gemm, checked_batch_gemm_geometry};
use crate::network::{LegId, TensorNetwork};
use crate::path::{simulate_path, sorted_dedup, SsaPath};
use crate::tensor::{DenseTensor, Scalar};

/// Preprocessing plan for one input tensor.
#[derive(Clone, Debug)]
struct InputPrep {
    /// Ordered `(ax_i, ax_j, is_diag)` operations on the current tensor rank.
    /// Diagonal extraction keeps one axis; a trace removes both.
    traces: Vec<(usize, usize, bool)>,
}

/// Integer-only plan for one pairwise contraction.
#[derive(Clone, Debug)]
struct Step {
    li: usize,
    ri: usize,
    /// Axes reduced from A in order as its rank decreases.
    sum_a: Vec<usize>,
    sum_b: Vec<usize>,
    /// Permutations to `[batch, free_a, con]` and `[batch, con, free_b]`.
    a_perm: Vec<usize>,
    b_perm: Vec<usize>,
    bt: usize,
    m: usize,
    k: usize,
    n: usize,
    /// Result shape in `[batch, free_a, free_b]` leg order.
    result_shape: Vec<usize>,
}

/// Reusable contraction plan independent of tensor values.
#[derive(Clone, Debug)]
pub struct CompiledContraction {
    n_inputs: usize,
    /// Expected input shapes validated before GEMM.
    input_shapes: Vec<Vec<usize>>,
    input_prep: Vec<InputPrep>,
    steps: Vec<Step>,
    /// Slot containing the sole final tensor.
    final_slot: usize,
    /// Final reduction axes for one-tensor networks.
    final_sum: Vec<usize>,
    /// Final permutation into output order.
    final_perm: Vec<usize>,
    final_shape: Vec<usize>,
    /// Planning statistics computed during compilation.
    pub log10_flops: f64,
    pub log2_peak_size: f64,
}

impl CompiledContraction {
    pub fn n_inputs(&self) -> usize {
        self.n_inputs
    }
    pub fn n_steps(&self) -> usize {
        self.steps.len()
    }

    /// Compiles a symbolic leg-accounting pass into integer execution steps.
    /// Leg semantics match `contract_network`.
    pub fn compile(net: &TensorNetwork, path: &SsaPath) -> Result<Self, String> {
        net.validate()?;
        // Match contract_network by rejecting repeated output legs.
        {
            let mut od = net.output.clone();
            od.sort_unstable();
            if od.windows(2).any(|w| w[0] == w[1]) {
                return Err(format!("output 含重复腿，暂不支持: {:?}", net.output));
            }
        }
        // Validate the path and compute planning statistics.
        let stats = simulate_path(net, path)?;

        let n = net.n_tensors();
        let in_output: HashSet<LegId> = net.output.iter().copied().collect();

        // Count distinct tensor holders for each leg.
        let mut refcount: HashMap<LegId, usize> = HashMap::new();
        for t in &net.inputs {
            for l in sorted_dedup(t) {
                *refcount.entry(l).or_insert(0) += 1;
            }
        }

        // Symbolic slots track only leg lists.
        let mut slots: Vec<Option<Vec<LegId>>> = Vec::with_capacity(n + path.len());
        let mut input_prep = Vec::with_capacity(n);
        for legs0 in &net.inputs {
            let mut legs = legs0.clone();
            let external = |l: LegId| refcount[&l] > 1 || in_output.contains(&l);
            let traces = plan_traces(&mut legs, &external)?;
            input_prep.push(InputPrep { traces });
            slots.push(Some(legs));
        }

        let mut steps = Vec::with_capacity(path.len());
        for (step, &(ia, ib)) in path.iter().enumerate() {
            if ia == ib {
                return Err(format!("第 {step} 步自收缩"));
            }
            let la = slots
                .get_mut(ia)
                .and_then(|s| s.take())
                .ok_or(format!("第 {step} 步引用无效张量 {ia}"))?;
            let lb = slots
                .get_mut(ib)
                .and_then(|s| s.take())
                .ok_or(format!("第 {step} 步引用无效张量 {ib}"))?;
            for &l in la.iter().chain(lb.iter()) {
                *refcount.get_mut(&l).unwrap() -= 1;
            }
            let keep = |l: LegId| refcount[&l] > 0 || in_output.contains(&l);
            let (rlegs, st) = plan_pairwise(net, &la, &lb, ia, ib, &keep)?;
            for &l in &rlegs {
                *refcount.get_mut(&l).unwrap() += 1;
            }
            steps.push(st);
            slots.push(Some(rlegs));
        }

        // Locate the sole live result.
        let alive: Vec<usize> = (0..slots.len()).filter(|&i| slots[i].is_some()).collect();
        if alive.len() != 1 {
            return Err(format!("路径不完整：剩余 {} 个张量", alive.len()));
        }
        let mut legs = slots[alive[0]].take().unwrap();

        // Reduce residual legs one at a time.
        let mut final_sum = Vec::new();
        let mut i = 0;
        while i < legs.len() {
            if !in_output.contains(&legs[i]) {
                final_sum.push(i);
                legs.remove(i);
            } else {
                i += 1;
            }
        }
        // Permute into output order.
        if sorted_dedup(&legs) != sorted_dedup(&net.output) {
            return Err(format!("最终腿 {legs:?} 与 output {:?} 不符", net.output));
        }
        let final_perm: Vec<usize> = net
            .output
            .iter()
            .map(|&l| legs.iter().position(|&x| x == l).unwrap())
            .collect();
        let final_shape: Vec<usize> = net.output.iter().map(|&l| net.dim(l)).collect();

        let input_shapes: Vec<Vec<usize>> = net
            .inputs
            .iter()
            .map(|legs| legs.iter().map(|&l| net.dim(l)).collect())
            .collect();
        Ok(CompiledContraction {
            n_inputs: n,
            input_shapes,
            input_prep,
            steps,
            final_slot: alive[0],
            final_sum,
            final_perm,
            final_shape,
            log10_flops: stats.log10_flops,
            // Report the live-set peak, not only the largest intermediate.
            log2_peak_size: stats.log2_peak_size,
        })
    }

    /// Executes the precompiled plan exactly.
    pub fn execute<T: Scalar>(&self, tensors: &[DenseTensor<T>]) -> Result<DenseTensor<T>, String> {
        let (t, exp) = self.execute_impl(tensors, false)?;
        debug_assert_eq!(exp, 0.0);
        Ok(t)
    }

    /// Executes with exponent stripping and returns `(normalized, log10_exponent)`.
    pub fn execute_stripped<T: Scalar>(
        &self,
        tensors: &[DenseTensor<T>],
    ) -> Result<(DenseTensor<T>, f64), String> {
        self.execute_impl(tensors, true)
    }

    fn execute_impl<T: Scalar>(
        &self,
        tensors: &[DenseTensor<T>],
        strip: bool,
    ) -> Result<(DenseTensor<T>, f64), String> {
        if tensors.len() != self.n_inputs {
            return Err(format!(
                "张量个数 {} 与编译计划 {} 不符",
                tensors.len(),
                self.n_inputs
            ));
        }
        // Apply the same shape contract as contract_network before GEMM.
        for (i, (want, t)) in self.input_shapes.iter().zip(tensors).enumerate() {
            t.validate_layout()
                .map_err(|e| format!("张量 {i} 布局非法: {e}"))?;
            if *want != t.shape {
                return Err(format!(
                    "张量 {i} shape {:?} 与编译计划期望维度 {:?} 不符",
                    t.shape, want
                ));
            }
        }
        let mut slots: Vec<Option<DenseTensor<T>>> =
            Vec::with_capacity(self.n_inputs + self.steps.len());
        // Apply input traces and diagonal extraction.
        for (i, prep) in self.input_prep.iter().enumerate() {
            let mut t = tensors[i].clone();
            for &(ax_i, ax_j, is_diag) in &prep.traces {
                t = if is_diag {
                    t.diag_pair(ax_i, ax_j)
                } else {
                    t.trace_pair(ax_i, ax_j)
                };
            }
            slots.push(Some(t));
        }

        let mut exponent = 0.0f64;
        for st in &self.steps {
            let mut a = slots[st.li].take().ok_or("计划引用了已消费的张量")?;
            let mut b = slots[st.ri].take().ok_or("计划引用了已消费的张量")?;
            for &ax in &st.sum_a {
                a = a.sum_axis(ax);
            }
            for &ax in &st.sum_b {
                b = b.sum_axis(ax);
            }
            let a = a.permute(&st.a_perm);
            let b = b.permute(&st.b_perm);

            let (bt, m, k, n2) = (st.bt, st.m, st.k, st.n);
            let mut c = checked_batch_gemm(&a, &b, bt, m, k, n2)?;
            if strip {
                // Normalize by the largest magnitude and accumulate its exponent.
                let mut fmax = 0.0f64;
                for v in &c {
                    // `abs_sq` can overflow for finite large values; use direct magnitude.
                    let magnitude = v.abs();
                    if magnitude > fmax {
                        fmax = magnitude;
                    }
                }
                let factor = fmax;
                if factor > 0.0 && factor.is_finite() {
                    exponent += factor.log10();
                    let inv = T::from_real(1.0 / factor);
                    for v in &mut c {
                        *v = *v * inv;
                    }
                }
            }
            slots.push(Some(
                DenseTensor::try_from_data(st.result_shape.clone(), c)
                    .map_err(|e| format!("预编译 GEMM 结果布局非法: {e}"))?,
            ));
        }

        // Retrieve the final slot selected at compile time.
        let mut t = slots[self.final_slot].take().ok_or("收尾张量缺失")?;
        for &ax in &self.final_sum {
            t = t.sum_axis(ax);
        }
        let t = t.permute(&self.final_perm);
        debug_assert_eq!(t.shape, self.final_shape);
        Ok((t, exponent))
    }
}

/// Plans trace and diagonal operations without touching tensor data.
fn plan_traces(
    legs: &mut Vec<LegId>,
    external: &dyn Fn(LegId) -> bool,
) -> Result<Vec<(usize, usize, bool)>, String> {
    let mut ops = Vec::new();
    loop {
        let mut dup: Option<(usize, usize)> = None;
        'outer: for i in 0..legs.len() {
            for j in (i + 1)..legs.len() {
                if legs[i] == legs[j] {
                    dup = Some((i, j));
                    break 'outer;
                }
            }
        }
        match dup {
            None => return Ok(ops),
            Some((i, j)) => {
                let l = legs[i];
                if legs.iter().filter(|&&x| x == l).count() > 2 {
                    return Err(format!("腿 {l} 在同一张量出现 >2 次，暂不支持"));
                }
                if external(l) {
                    ops.push((i, j, true)); // Extract the diagonal, retaining axis i.
                    legs.remove(j);
                } else {
                    ops.push((i, j, false)); // Trace removes both axes.
                    legs.remove(j);
                    legs.remove(i);
                }
            }
        }
    }
}

/// Plans reduction axes, permutations, GEMM geometry, result legs, and shape.
fn plan_pairwise(
    net: &TensorNetwork,
    la: &[LegId],
    lb: &[LegId],
    li: usize,
    ri: usize,
    keep: &dyn Fn(LegId) -> bool,
) -> Result<(Vec<LegId>, Step), String> {
    let set_a: HashSet<LegId> = la.iter().copied().collect();
    let set_b: HashSet<LegId> = lb.iter().copied().collect();

    let mut batch = Vec::new();
    let mut con = Vec::new();
    let mut free_a = Vec::new();
    let mut sum_a_legs = Vec::new();
    for &l in la {
        if set_b.contains(&l) {
            if keep(l) {
                batch.push(l);
            } else {
                con.push(l);
            }
        } else if keep(l) {
            free_a.push(l);
        } else {
            sum_a_legs.push(l);
        }
    }
    let mut free_b = Vec::new();
    let mut sum_b_legs = Vec::new();
    for &l in lb {
        if !set_a.contains(&l) {
            if keep(l) {
                free_b.push(l);
            } else {
                sum_b_legs.push(l);
            }
        }
    }

    // Record reduction axes against the rank at each step.
    let mut la_work: Vec<LegId> = la.to_vec();
    let mut sum_a = Vec::with_capacity(sum_a_legs.len());
    for l in &sum_a_legs {
        let ax = la_work.iter().position(|x| x == l).unwrap();
        sum_a.push(ax);
        la_work.remove(ax);
    }
    let mut lb_work: Vec<LegId> = lb.to_vec();
    let mut sum_b = Vec::with_capacity(sum_b_legs.len());
    for l in &sum_b_legs {
        let ax = lb_work.iter().position(|x| x == l).unwrap();
        sum_b.push(ax);
        lb_work.remove(ax);
    }

    // A -> [batch, free_a, con], B -> [batch, con, free_b].
    let pos = |legs: &[LegId], l: LegId| legs.iter().position(|&x| x == l).unwrap();
    let a_perm: Vec<usize> = batch
        .iter()
        .chain(free_a.iter())
        .chain(con.iter())
        .map(|&l| pos(&la_work, l))
        .collect();
    let b_perm: Vec<usize> = batch
        .iter()
        .chain(con.iter())
        .chain(free_b.iter())
        .map(|&l| pos(&lb_work, l))
        .collect();

    let dprod = |legs: &[LegId], label: &str| -> Result<usize, String> {
        legs.iter().try_fold(1usize, |acc, &leg| {
            acc.checked_mul(net.try_dim(leg)?)
                .ok_or_else(|| format!("预编译 GEMM {label} 维度乘积溢出 usize"))
        })
    };
    let bt = dprod(&batch, "batch")?;
    let m = dprod(&free_a, "m")?;
    let k = dprod(&con, "k")?;
    let n = dprod(&free_b, "n")?;
    // Reject pointer geometry that would exceed isize before execution.
    checked_batch_gemm_geometry(bt, m, k, n)?;

    let result_legs: Vec<LegId> = batch
        .iter()
        .chain(free_a.iter())
        .chain(free_b.iter())
        .copied()
        .collect();
    let result_shape: Vec<usize> = result_legs
        .iter()
        .map(|&l| net.try_dim(l))
        .collect::<Result<_, _>>()?;

    Ok((
        result_legs,
        Step {
            li,
            ri,
            sum_a,
            sum_b,
            a_perm,
            b_perm,
            bt,
            m,
            k,
            n,
            result_shape,
        },
    ))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use num_complex::Complex64;

    use super::CompiledContraction;
    use crate::network::TensorNetwork;
    use crate::path::simulate_path;
    use crate::tensor::DenseTensor;

    #[test]
    fn compile_reports_liveness_peak_not_largest_intermediate() {
        // Three 2x2 inputs and the first result coexist, giving a 16-element live peak.
        let mut size_dict = HashMap::new();
        for leg in 0..4 {
            size_dict.insert(leg, 2);
        }
        let net = TensorNetwork {
            name: "compiled-peak-regression".into(),
            inputs: vec![vec![0, 1], vec![1, 2], vec![2, 3]],
            output: vec![0, 3],
            size_dict,
        };
        let path = vec![(0, 1), (3, 2)];

        let stats = simulate_path(&net, &path).expect("链路径应合法");
        assert_eq!(stats.log2_max_size, 2.0);
        assert_eq!(stats.log2_peak_size, 4.0);

        let compiled = CompiledContraction::compile(&net, &path).expect("编译应成功");
        assert_eq!(compiled.log2_peak_size, stats.log2_peak_size);
        assert_ne!(compiled.log2_peak_size, stats.log2_max_size);
    }

    #[test]
    fn stripped_execution_normalizes_extreme_finite_f64_and_complex_values() {
        // Finite 1e200 values exercise the branch where squaring would overflow.
        let net = TensorNetwork {
            name: "compiled-strip-extreme-regression".into(),
            inputs: vec![vec![], vec![]],
            output: vec![],
            size_dict: HashMap::new(),
        };
        let compiled = CompiledContraction::compile(&net, &vec![(0, 1)]).expect("编译应成功");

        let (mantissa, exponent) = compiled
            .execute_stripped(&[DenseTensor::scalar(1e200f64), DenseTensor::scalar(1.0)])
            .expect("f64 strip 执行应成功");
        assert!(exponent.is_finite() && exponent > 199.0);
        assert!((mantissa.data[0].abs() - 1.0).abs() < 1e-12);
        let rebuilt = mantissa.data[0] * 10f64.powf(exponent);
        assert!((rebuilt - 1e200).abs() / 1e200 < 1e-12);

        let extreme = Complex64::new(1e200, -1e200);
        let (complex_mantissa, complex_exponent) = compiled
            .execute_stripped(&[
                DenseTensor::scalar(extreme),
                DenseTensor::scalar(Complex64::new(1.0, 0.0)),
            ])
            .expect("Complex64 strip 执行应成功");
        assert!(complex_exponent.is_finite() && complex_exponent > 199.0);
        assert!((complex_mantissa.data[0].norm() - 1.0).abs() < 1e-12);
        let complex_rebuilt = complex_mantissa.data[0] * 10f64.powf(complex_exponent);
        assert!((complex_rebuilt - extreme).norm() / extreme.norm() < 1e-12);
    }

    #[test]
    fn compiled_execute_rejects_malformed_layout_before_gemm() {
        let net = TensorNetwork {
            name: "compiled-layout".into(),
            inputs: vec![vec![0, 1], vec![1, 2]],
            output: vec![0, 2],
            size_dict: [(0, 2), (1, 2), (2, 2)].into_iter().collect(),
        };
        let compiled = CompiledContraction::compile(&net, &vec![(0, 1)]).unwrap();
        let malformed = DenseTensor {
            shape: vec![2, 2],
            data: vec![1.0],
        };
        let rhs = DenseTensor::from_data(vec![2, 2], vec![1.0; 4]);
        let err = compiled
            .execute(&[malformed, rhs])
            .expect_err("短 data 必须在预编译 raw GEMM 前被拒绝");
        assert!(err.contains("布局非法"), "unexpected error: {err}");
    }

    #[test]
    fn compile_rejects_overflowing_network_before_planning() {
        let net = TensorNetwork {
            name: "compiled-overflow".into(),
            inputs: vec![vec![0, 1], vec![1, 2]],
            output: vec![0, 2],
            size_dict: [(0, isize::MAX as usize), (1, 3), (2, 2)]
                .into_iter()
                .collect(),
        };
        let err = CompiledContraction::compile(&net, &vec![(0, 1)])
            .expect_err("尺寸溢出必须在预编译计划前被拒绝");
        assert!(err.contains("元素数溢出"), "unexpected error: {err}");
    }
}
