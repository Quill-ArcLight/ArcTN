//! Output slicing assembles disjoint blocks, while internal slicing sums them.

use arctn::tensor::Scalar;
use arctn::{contract_network_sliced, naive_einsum, DenseTensor, TensorNetwork};
use num_complex::{Complex32, Complex64};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn check_slices<T: Scalar>(net: &TensorNetwork, path: &Vec<(usize, usize)>, tolerance: f64) {
    let mut rng = ChaCha8Rng::seed_from_u64(9817);
    let tensors: Vec<DenseTensor<T>> = net
        .inputs
        .iter()
        .map(|legs| DenseTensor::random(legs.iter().map(|&leg| net.dim(leg)).collect(), &mut rng))
        .collect();
    let expected = naive_einsum(net, &tensors).unwrap();
    let mut labels: Vec<_> = net.size_dict.keys().copied().collect();
    labels.sort_unstable();
    // Every subset covers output-only, internal-only, mixed, and fully sliced
    // outputs. Reverse the label order to exercise assignment independence.
    for mask in 0..(1usize << labels.len()) {
        let mut sliced: Vec<_> = labels
            .iter()
            .enumerate()
            .filter_map(|(position, &leg)| ((mask >> position) & 1 == 1).then_some(leg))
            .collect();
        sliced.reverse();
        let actual = contract_network_sliced(net, &tensors, path, &sliced).unwrap();
        assert_eq!(actual.shape(), expected.shape(), "sliced={sliced:?}");
        let error = actual.max_abs_diff(&expected);
        assert!(error < tolerance, "sliced={sliced:?}, error={error:e}");
    }
}

#[test]
fn output_blocks_preserve_order_and_hyperedges_for_all_scalar_types() {
    let net = TensorNetwork {
        name: "shared-output-hyperedge".into(),
        inputs: vec![vec![0, 1, 4], vec![1, 2, 4], vec![1, 3]],
        output: vec![3, 1, 0, 2],
        size_dict: [(0, 2), (1, 3), (2, 2), (3, 2), (4, 5)]
            .into_iter()
            .collect(),
    };
    let path = vec![(0, 1), (3, 2)];
    check_slices::<f64>(&net, &path, 1e-11);
    check_slices::<f32>(&net, &path, 1e-4);
    check_slices::<Complex64>(&net, &path, 1e-11);
    check_slices::<Complex32>(&net, &path, 1e-4);
}

#[test]
fn unary_slices_preserve_output_permutation_and_unary_reduction() {
    let net = TensorNetwork {
        name: "unary-output-blocks".into(),
        inputs: vec![vec![0, 1, 2]],
        output: vec![2, 0],
        size_dict: [(0, 2), (1, 3), (2, 4)].into_iter().collect(),
    };
    let data = (0..2)
        .flat_map(|a| (0..3).flat_map(move |b| (0..4).map(move |c| (100 * a + 10 * b + c) as f64)))
        .collect();
    let tensors = vec![DenseTensor::from_data(vec![2, 3, 4], data)];
    let expected = vec![30.0, 330.0, 33.0, 333.0, 36.0, 336.0, 39.0, 339.0];
    for sliced in [vec![0], vec![2], vec![2, 0], vec![1, 2], vec![2, 1, 0]] {
        let actual = contract_network_sliced(&net, &tensors, &vec![], &sliced).unwrap();
        assert_eq!(actual.shape(), &[4, 2]);
        assert_eq!(actual.data(), expected, "sliced={sliced:?}");
    }
}

#[test]
fn mixed_slices_are_bit_identical_across_thread_counts() {
    let net = TensorNetwork {
        name: "mixed-ragged-chunks".into(),
        inputs: vec![vec![0, 1], vec![1, 2]],
        output: vec![2, 0],
        size_dict: [(0, 2), (1, 257), (2, 3)].into_iter().collect(),
    };
    let mut rng = ChaCha8Rng::seed_from_u64(377);
    let tensors = vec![
        DenseTensor::<f64>::random(vec![2, 257], &mut rng),
        DenseTensor::<f64>::random(vec![257, 3], &mut rng),
    ];
    let path = vec![(0, 1)];
    let expected = naive_einsum(&net, &tensors).unwrap();
    let mut first = None;
    for threads in [1, 2, 3, 4, 8] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let actual =
            pool.install(|| contract_network_sliced(&net, &tensors, &path, &[1, 0]).unwrap());
        assert!(actual.max_abs_diff(&expected) < 1e-10);
        let bits: Vec<_> = actual.data().iter().map(|value| value.to_bits()).collect();
        if let Some(ref baseline) = first {
            assert_eq!(&bits, baseline, "threads={threads}");
        } else {
            first = Some(bits);
        }
    }
}

#[test]
fn output_slicing_still_rejects_duplicate_unknown_and_trace_labels() {
    let net = TensorNetwork {
        name: "output-validation".into(),
        inputs: vec![vec![0, 1]],
        output: vec![0, 1],
        size_dict: [(0, 2), (1, 2)].into_iter().collect(),
    };
    let tensors = vec![DenseTensor::from_data(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0])];
    for sliced in [vec![0, 0], vec![0, 9]] {
        assert!(contract_network_sliced(&net, &tensors, &vec![], &sliced).is_err());
    }
    let repeated = TensorNetwork {
        name: "output-diagonal".into(),
        inputs: vec![vec![0, 0]],
        output: vec![0],
        size_dict: [(0, 2)].into_iter().collect(),
    };
    let error = contract_network_sliced(&repeated, &tensors, &vec![], &[0]).unwrap_err();
    assert!(error.contains("同一输入张量中重复"), "{error}");
}

#[test]
fn oversized_full_output_returns_an_error_before_allocation() {
    // Each input contains only two elements, but their outer product cannot
    // fit in a Vec<f64>. No huge input or attempted allocation is needed.
    let n = usize::BITS as usize - 1;
    let labels: Vec<_> = (0..n as u32).collect();
    let net = TensorNetwork {
        name: "oversized-output".into(),
        inputs: labels.iter().map(|&leg| vec![leg]).collect(),
        output: labels.clone(),
        size_dict: labels.iter().map(|&leg| (leg, 2)).collect(),
    };
    let tensors = vec![DenseTensor::from_data(vec![2], vec![1.0, 2.0]); n];
    let mut path = vec![(0, 1)];
    for input in 2..n {
        path.push((n + input - 2, input));
    }
    let error = contract_network_sliced(&net, &tensors, &path, &[0]).unwrap_err();
    assert!(error.contains("完整输出张量的字节数"), "{error}");
}
