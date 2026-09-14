//! Executes a tensor-network contraction along an SSA path.
//!
//! Each pairwise step removes unused reduction legs, permutes and reshapes into
//! batched matrices, applies GEMM, and emits batch plus free legs.

use std::collections::{HashMap, HashSet};

use crate::network::{LegId, TensorNetwork};
use crate::path::{sorted_dedup, SsaPath};
use crate::tensor::{DenseTensor, Scalar};

/// Validates tensor counts and shapes against the network.
///
/// Sliced execution must call this before `select_axis`; otherwise zipping can
/// silently discard extra inputs or defer short-input failures.
pub(crate) fn validate_shapes<T: Scalar>(
    net: &TensorNetwork,
    tensors: &[DenseTensor<T>],
) -> Result<(), String> {
    net.validate()?;
    if tensors.len() != net.n_tensors() {
        return Err(format!(
            "张量个数 {} 与网络 {} 不符",
            tensors.len(),
            net.n_tensors()
        ));
    }
    for (i, (legs, t)) in net.inputs.iter().zip(tensors).enumerate() {
        t.validate_layout()
            .map_err(|e| format!("张量 {i} 布局非法: {e}"))?;
        let want: Vec<usize> = legs.iter().map(|&l| net.dim(l)).collect();
        if want != t.shape {
            return Err(format!(
                "张量 {i} shape {:?} 与腿 {:?} 的维度 {:?} 不符",
                t.shape, legs, want
            ));
        }
    }
    Ok(())
}

/// Validates scalar-independent dimensions, offsets, and strides for batched GEMM.
///
/// `matrixmultiply` cannot detect short buffers or wrapped dimensions. Contract
/// and compiled execution share this check, while [`checked_batch_gemm`] validates
/// layout and byte bounds before the only raw GEMM call.
///
/// This function checks element counts and `isize` pointer offsets only.
pub(crate) fn checked_batch_gemm_geometry(
    bt: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Result<(usize, usize, usize, usize, usize, usize), String> {
    let max_offset = isize::MAX as usize;
    for (name, value) in [("batch", bt), ("m", m), ("k", k), ("n", n)] {
        if value == 0 {
            return Err(format!("GEMM 维度 {name}=0；ArcTN 执行器不支持零维张量"));
        }
        if value > max_offset {
            return Err(format!(
                "GEMM 维度 {name}={value} 超过 isize stride 可表示范围"
            ));
        }
    }

    let a_block = m
        .checked_mul(k)
        .ok_or_else(|| "GEMM 左操作数单 batch 元素数溢出 usize".to_string())?;
    let b_block = k
        .checked_mul(n)
        .ok_or_else(|| "GEMM 右操作数单 batch 元素数溢出 usize".to_string())?;
    let c_block = m
        .checked_mul(n)
        .ok_or_else(|| "GEMM 输出单 batch 元素数溢出 usize".to_string())?;
    let a_len = bt
        .checked_mul(a_block)
        .ok_or_else(|| "GEMM 左操作数总元素数溢出 usize".to_string())?;
    let b_len = bt
        .checked_mul(b_block)
        .ok_or_else(|| "GEMM 右操作数总元素数溢出 usize".to_string())?;
    let c_len = bt
        .checked_mul(c_block)
        .ok_or_else(|| "GEMM 输出总元素数溢出 usize".to_string())?;

    // Raw strides and pointer offsets must remain representable as isize.
    for (name, len) in [
        ("左操作数单 batch", a_block),
        ("右操作数单 batch", b_block),
        ("输出单 batch", c_block),
        ("左操作数总", a_len),
        ("右操作数总", b_len),
        ("输出总", c_len),
    ] {
        if len > max_offset {
            return Err(format!("GEMM {name} 元素偏移 {len} 超过 isize 可表示范围"));
        }
    }
    Ok((a_block, b_block, c_block, a_len, b_len, c_len))
}

pub(crate) fn checked_batch_gemm<T: Scalar>(
    a: &DenseTensor<T>,
    b: &DenseTensor<T>,
    bt: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<T>, String> {
    a.validate_layout()
        .map_err(|e| format!("GEMM 左操作数布局非法: {e}"))?;
    b.validate_layout()
        .map_err(|e| format!("GEMM 右操作数布局非法: {e}"))?;

    let (a_block, b_block, c_block, a_len, b_len, c_len) =
        checked_batch_gemm_geometry(bt, m, k, n)?;

    let elem_bytes = std::mem::size_of::<T>();
    if elem_bytes == 0 {
        return Err("GEMM 标量类型不能是零大小类型".to_string());
    }
    let max_vec_elements = (isize::MAX as usize) / elem_bytes;
    for (name, len) in [("左操作数", a_len), ("右操作数", b_len), ("输出", c_len)] {
        if len > max_vec_elements {
            return Err(format!(
                "GEMM {name} 缓冲区 {len} 个元素（每个 {elem_bytes} 字节）超过 isize 字节可寻址上限"
            ));
        }
    }
    if a.data.len() != a_len {
        return Err(format!(
            "GEMM 左操作数长度 {} 与 batch×m×k={a_len} 不符",
            a.data.len()
        ));
    }
    if b.data.len() != b_len {
        return Err(format!(
            "GEMM 右操作数长度 {} 与 batch×k×n={b_len} 不符",
            b.data.len()
        ));
    }

    let rsa = k as isize;
    let rsb = n as isize;
    let rsc = n as isize;
    let mut c = vec![T::zero(); c_len];
    for ib in 0..bt {
        // Geometry checks prove these offsets remain inside the allocated buffers.
        let a_off = ib
            .checked_mul(a_block)
            .ok_or_else(|| "GEMM 左操作数 batch 偏移溢出 usize".to_string())?;
        let b_off = ib
            .checked_mul(b_block)
            .ok_or_else(|| "GEMM 右操作数 batch 偏移溢出 usize".to_string())?;
        let c_off = ib
            .checked_mul(c_block)
            .ok_or_else(|| "GEMM 输出 batch 偏移溢出 usize".to_string())?;
        debug_assert!(a_off < a.data.len());
        debug_assert!(b_off < b.data.len());
        debug_assert!(c_off < c.len());
        unsafe {
            T::gemm(
                m,
                k,
                n,
                T::one(),
                a.data.as_ptr().add(a_off),
                rsa,
                1,
                b.data.as_ptr().add(b_off),
                rsb,
                1,
                T::zero(),
                c.as_mut_ptr().add(c_off),
                rsc,
                1,
            );
        }
    }
    Ok(c)
}

/// Resolves repeated legs within one tensor.
///
/// A repeated internal leg is traced out; an externally referenced leg becomes
/// a diagonal axis. Multiplicity greater than two is unsupported.
fn resolve_traces<T: Scalar>(
    legs: &mut Vec<LegId>,
    tensor: DenseTensor<T>,
    external: &dyn Fn(LegId) -> bool,
) -> Result<DenseTensor<T>, String> {
    let mut t = tensor;
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
            None => return Ok(t),
            Some((i, j)) => {
                let l = legs[i];
                if legs.iter().filter(|&&x| x == l).count() > 2 {
                    return Err(format!("腿 {l} 在同一张量出现 >2 次，暂不支持"));
                }
                if external(l) {
                    // Extract the diagonal and keep one axis.
                    t = t.diag_pair(i, j);
                    legs.remove(j);
                } else {
                    // Trace the diagonal and remove both axes.
                    t = t.trace_pair(i, j);
                    legs.remove(j);
                    legs.remove(i);
                }
            }
        }
    }
}

/// Contracts a pair with duplicate-free leg lists; `keep` selects result legs.
fn pairwise<T: Scalar>(
    net: &TensorNetwork,
    la: &[LegId],
    a: DenseTensor<T>,
    lb: &[LegId],
    b: DenseTensor<T>,
    keep: &dyn Fn(LegId) -> bool,
) -> Result<(Vec<LegId>, DenseTensor<T>), String> {
    let set_a: HashSet<LegId> = la.iter().copied().collect();
    let set_b: HashSet<LegId> = lb.iter().copied().collect();

    let mut batch = Vec::new();
    let mut con = Vec::new();
    let mut free_a = Vec::new();
    let mut sum_a = Vec::new();
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
            sum_a.push(l);
        }
    }
    let mut free_b = Vec::new();
    let mut sum_b = Vec::new();
    for &l in lb {
        if !set_a.contains(&l) {
            if keep(l) {
                free_b.push(l);
            } else {
                sum_b.push(l);
            }
        }
    }

    // Sum exclusive legs that no later tensor or output needs.
    let mut la: Vec<LegId> = la.to_vec();
    let mut a = a;
    for l in sum_a {
        let ax = la.iter().position(|&x| x == l).unwrap();
        a = a.sum_axis(ax);
        la.remove(ax);
    }
    let mut lb: Vec<LegId> = lb.to_vec();
    let mut b = b;
    for l in sum_b {
        let ax = lb.iter().position(|&x| x == l).unwrap();
        b = b.sum_axis(ax);
        lb.remove(ax);
    }

    // permute：A -> [batch, free_a, con]，B -> [batch, con, free_b]
    let pos = |legs: &[LegId], l: LegId| legs.iter().position(|&x| x == l).unwrap();
    let a_axes: Vec<usize> = batch
        .iter()
        .chain(free_a.iter())
        .chain(con.iter())
        .map(|&l| pos(&la, l))
        .collect();
    let b_axes: Vec<usize> = batch
        .iter()
        .chain(con.iter())
        .chain(free_b.iter())
        .map(|&l| pos(&lb, l))
        .collect();
    let a = a.permute(&a_axes);
    let b = b.permute(&b_axes);

    let dprod = |legs: &[LegId], label: &str| -> Result<usize, String> {
        legs.iter().try_fold(1usize, |acc, &leg| {
            acc.checked_mul(net.try_dim(leg)?)
                .ok_or_else(|| format!("GEMM {label} 维度乘积溢出 usize"))
        })
    };
    let bt = dprod(&batch, "batch")?;
    let m = dprod(&free_a, "m")?;
    let k = dprod(&con, "k")?;
    let n2 = dprod(&free_b, "n")?;
    let c = checked_batch_gemm(&a, &b, bt, m, k, n2)?;

    let result_legs: Vec<LegId> = batch
        .iter()
        .chain(free_a.iter())
        .chain(free_b.iter())
        .copied()
        .collect();
    let shape: Vec<usize> = result_legs
        .iter()
        .map(|&l| net.try_dim(l))
        .collect::<Result<_, _>>()?;
    Ok((
        result_legs,
        DenseTensor::try_from_data(shape, c).map_err(|e| format!("GEMM 结果布局非法: {e}"))?,
    ))
}

/// Contracts a network along an SSA path and orders axes as `net.output`.
pub fn contract_network<T: Scalar>(
    net: &TensorNetwork,
    tensors: Vec<DenseTensor<T>>,
    path: &SsaPath,
) -> Result<DenseTensor<T>, String> {
    validate_shapes(net, &tensors)?;
    // Repeated output legs cannot be represented as a final axis permutation.
    {
        let mut od = net.output.clone();
        od.sort_unstable();
        if od.windows(2).any(|w| w[0] == w[1]) {
            return Err(format!("output 含重复腿，暂不支持: {:?}", net.output));
        }
    }
    let n = net.n_tensors();
    let in_output: HashSet<LegId> = net.output.iter().copied().collect();

    // Count distinct tensor holders for each leg.
    let mut refcount: HashMap<LegId, usize> = HashMap::new();
    for t in &net.inputs {
        for l in sorted_dedup(t) {
            *refcount.entry(l).or_insert(0) += 1;
        }
    }

    // Resolve traces before path execution.
    let mut slots: Vec<Option<(Vec<LegId>, DenseTensor<T>)>> = Vec::with_capacity(n + path.len());
    for (i, t) in tensors.into_iter().enumerate() {
        let mut legs = net.inputs[i].clone();
        let rc = &refcount;
        let io = &in_output;
        let external = move |l: LegId| rc[&l] > 1 || io.contains(&l);
        let t = resolve_traces(&mut legs, t, &external)?;
        slots.push(Some((legs, t)));
    }

    for (step, &(ia, ib)) in path.iter().enumerate() {
        if ia == ib {
            return Err(format!("第 {step} 步自收缩"));
        }
        let (la, ta) = slots
            .get_mut(ia)
            .and_then(|s| s.take())
            .ok_or(format!("第 {step} 步引用无效张量 {ia}"))?;
        let (lb, tb) = slots
            .get_mut(ib)
            .and_then(|s| s.take())
            .ok_or(format!("第 {step} 步引用无效张量 {ib}"))?;
        // Remove the contracted tensors from holder counts.
        for &l in la.iter().chain(lb.iter()) {
            *refcount.get_mut(&l).unwrap() -= 1;
        }
        let rc = refcount.clone();
        let io = in_output.clone();
        let keep = move |l: LegId| rc[&l] > 0 || io.contains(&l);
        let (rlegs, rt) = pairwise(net, &la, ta, &lb, tb, &keep)?;
        for &l in &rlegs {
            *refcount.get_mut(&l).unwrap() += 1;
        }
        slots.push(Some((rlegs, rt)));
    }

    let alive: Vec<usize> = (0..slots.len()).filter(|&i| slots[i].is_some()).collect();
    if alive.len() != 1 {
        return Err(format!("路径不完整：剩余 {} 个张量", alive.len()));
    }
    let (mut legs, mut t) = slots[alive[0]].take().unwrap();

    // A one-tensor network may still contain reduction legs.
    let mut i = 0;
    while i < legs.len() {
        if !in_output.contains(&legs[i]) {
            t = t.sum_axis(i);
            legs.remove(i);
        } else {
            i += 1;
        }
    }

    // Permute the result into output order.
    if sorted_dedup(&legs) != sorted_dedup(&net.output) {
        return Err(format!("最终腿 {legs:?} 与 output {:?} 不符", net.output));
    }
    let axes: Vec<usize> = net
        .output
        .iter()
        .map(|&l| legs.iter().position(|&x| x == l).unwrap())
        .collect();
    Ok(t.permute(&axes))
}

#[cfg(test)]
mod tests {
    use super::{checked_batch_gemm_geometry, contract_network};
    use crate::network::TensorNetwork;
    use crate::tensor::DenseTensor;

    fn matmul_net() -> TensorNetwork {
        TensorNetwork {
            name: "matmul".into(),
            inputs: vec![vec![0, 1], vec![1, 2]],
            output: vec![0, 2],
            size_dict: [(0, 2), (1, 2), (2, 2)].into_iter().collect(),
        }
    }

    #[test]
    fn malformed_layout_is_rejected_before_raw_gemm() {
        // Simulate corrupted FFI data to verify rejection before raw GEMM.
        let malformed = DenseTensor {
            shape: vec![2, 2],
            data: vec![1.0],
        };
        let rhs = DenseTensor::from_data(vec![2, 2], vec![1.0; 4]);
        let err = contract_network(&matmul_net(), vec![malformed, rhs], &vec![(0, 1)])
            .expect_err("短 data 必须在 raw GEMM 前被拒绝");
        assert!(err.contains("布局非法"), "unexpected error: {err}");
    }

    #[test]
    fn overflow_network_is_rejected_before_shape_or_gemm_work() {
        let net = TensorNetwork {
            name: "overflow".into(),
            inputs: vec![vec![0, 1], vec![1, 2]],
            output: vec![0, 2],
            size_dict: [(0, isize::MAX as usize), (1, 3), (2, 2)]
                .into_iter()
                .collect(),
        };
        let err = contract_network::<f64>(&net, vec![], &vec![(0, 1)])
            .expect_err("溢出网络必须在读取张量前被拒绝");
        assert!(err.contains("元素数溢出"), "unexpected error: {err}");
    }

    #[test]
    fn gemm_geometry_rejects_isize_pointer_offset_before_kernel() {
        // A usize product can exceed the isize range used by raw pointer strides.
        let err = checked_batch_gemm_geometry(1, isize::MAX as usize, 2, 1)
            .expect_err("超出 isize 的单 batch 偏移必须被拒绝");
        assert!(err.contains("元素偏移"), "unexpected error: {err}");
    }
}
