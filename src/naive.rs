//! Brute-force einsum over the joint index space.
//! This is a correctness reference for small networks only.

use std::collections::HashMap;

use crate::contract::validate_shapes;
use crate::network::{LegId, TensorNetwork};
use crate::tensor::{DenseTensor, Scalar};

pub fn naive_einsum<T: Scalar>(
    net: &TensorNetwork,
    tensors: &[DenseTensor<T>],
) -> Result<DenseTensor<T>, String> {
    validate_shapes(net, tensors)?;
    let mut all_legs: Vec<LegId> = net
        .inputs
        .iter()
        .flatten()
        .chain(net.output.iter())
        .copied()
        .collect();
    all_legs.sort_unstable();
    all_legs.dedup();

    let log2_space: f64 = all_legs.iter().map(|&l| net.log2_dim(l)).sum();
    if log2_space > 28.0 {
        return Err(format!(
            "联合指标空间 2^{log2_space:.1} 太大，naive 只用于小网络"
        ));
    }
    let dims: Vec<usize> = all_legs.iter().map(|&l| net.dim(l)).collect();
    let leg_pos: HashMap<LegId, usize> =
        all_legs.iter().enumerate().map(|(i, &l)| (l, i)).collect();

    // Cache each tensor axis as (global index position, local stride).
    let mut tensor_maps: Vec<Vec<(usize, usize)>> = Vec::new();
    for (i, legs) in net.inputs.iter().enumerate() {
        let strides = tensors[i].strides();
        tensor_maps.push(
            legs.iter()
                .zip(strides)
                .map(|(&l, s)| (leg_pos[&l], s))
                .collect(),
        );
    }
    let out_shape: Vec<usize> = net.output.iter().map(|&l| net.dim(l)).collect();
    let mut out = DenseTensor::<T>::zeros(out_shape.clone());
    let out_strides = out.strides();
    let out_map: Vec<(usize, usize)> = net
        .output
        .iter()
        .zip(out_strides)
        .map(|(&l, s)| (leg_pos[&l], s))
        .collect();

    let total: usize = dims.iter().product();
    let mut idx = vec![0usize; all_legs.len()];
    for _ in 0..total {
        let mut term = T::one();
        for (t, map) in tensor_maps.iter().enumerate() {
            let off: usize = map.iter().map(|&(p, s)| idx[p] * s).sum();
            term = term * tensors[t].data[off];
        }
        let out_off: usize = out_map.iter().map(|&(p, s)| idx[p] * s).sum();
        out.data[out_off] += term;
        // Advance the mixed-radix index.
        for d in (0..idx.len()).rev() {
            idx[d] += 1;
            if idx[d] < dims[d] {
                break;
            }
            idx[d] = 0;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::naive_einsum;
    use crate::{DenseTensor, TensorNetwork};

    fn matmul_net() -> TensorNetwork {
        TensorNetwork {
            name: "naive-matmul".into(),
            inputs: vec![vec![0, 1], vec![1, 2]],
            output: vec![0, 2],
            size_dict: [(0, 2), (1, 2), (2, 2)].into_iter().collect(),
        }
    }

    #[test]
    fn malformed_inputs_return_errors_instead_of_panicking_or_miscomputing() {
        let net = matmul_net();
        let one = DenseTensor::from_data(vec![2, 2], vec![1.0; 4]);

        let short = naive_einsum(&net, std::slice::from_ref(&one))
            .expect_err("short tensor list must be rejected");
        assert!(short.contains("张量个数"), "{short}");

        let wrong_shape = DenseTensor::from_data(vec![4], vec![1.0; 4]);
        let mismatch = naive_einsum(&net, &[wrong_shape, one.clone()])
            .expect_err("wrong shape must be rejected");
        assert!(mismatch.contains("shape"), "{mismatch}");

        let mut missing_dim = net;
        missing_dim.size_dict.remove(&1);
        let missing = naive_einsum(&missing_dim, &[one.clone(), one])
            .expect_err("missing size_dict entry must be rejected");
        assert!(missing.contains("缺少 size_dict"), "{missing}");
    }
}
