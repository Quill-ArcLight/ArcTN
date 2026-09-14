//! Correctness tests for diagonal extraction and trace resolution.
//! `naive_einsum` is the reference because repeated axes share one global index.

use std::collections::HashMap;

use arctn::network::TensorNetwork;
use arctn::tensor::DenseTensor;
use arctn::{contract_network, naive_einsum};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn sd(pairs: &[(u32, usize)]) -> HashMap<u32, usize> {
    pairs.iter().copied().collect()
}

/// A repeated leg referenced by another tensor becomes a diagonal, not a trace.
#[test]
fn diagonal_shared_with_other_tensor() {
    let (da, db, dc) = (3usize, 2usize, 4usize);
    let net = TensorNetwork {
        name: "diag1".into(),
        inputs: vec![vec![0, 0, 1], vec![0, 2]],
        output: vec![1, 2],
        size_dict: sd(&[(0, da), (1, db), (2, dc)]),
    };
    let mut rng = ChaCha8Rng::seed_from_u64(1);
    let a = DenseTensor::<f64>::random(vec![da, da, db], &mut rng);
    let b = DenseTensor::<f64>::random(vec![da, dc], &mut rng);
    let got = contract_network(&net, vec![a.clone(), b.clone()], &vec![(0, 1)]).unwrap();
    let want = naive_einsum(&net, &[a, b]).unwrap();
    assert_eq!(got.shape(), want.shape());
    assert!(
        got.max_abs_diff(&want) < 1e-12,
        "diff={}",
        got.max_abs_diff(&want)
    );
}

/// A repeated output leg is extracted as a diagonal and remains open.
#[test]
fn diagonal_kept_as_output() {
    let (da, db) = (4usize, 3usize);
    let net = TensorNetwork {
        name: "diag2".into(),
        inputs: vec![vec![0, 0, 1], vec![1, 2]], // A[a,a,b] * B[b,x], with a exposed only by output.
        output: vec![0, 2],                      // [a,x]
        size_dict: sd(&[(0, da), (1, db), (2, 5)]),
    };
    let mut rng = ChaCha8Rng::seed_from_u64(2);
    let a = DenseTensor::<f64>::random(vec![da, da, db], &mut rng);
    let b = DenseTensor::<f64>::random(vec![db, 5], &mut rng);
    let got = contract_network(&net, vec![a.clone(), b.clone()], &vec![(0, 1)]).unwrap();
    let want = naive_einsum(&net, &[a, b]).unwrap();
    assert_eq!(got.shape(), want.shape());
    assert!(
        got.max_abs_diff(&want) < 1e-12,
        "diff={}",
        got.max_abs_diff(&want)
    );
}

/// One tensor may contain both a private trace leg and an externally referenced diagonal leg.
#[test]
fn mixed_trace_and_diagonal() {
    let (da, db, dc) = (3usize, 2usize, 4usize);
    let net = TensorNetwork {
        name: "mix".into(),
        inputs: vec![vec![0, 0, 1, 1], vec![1, 2]], // C[a,a,b,b]·D[b,c]
        output: vec![2], // [c] after tracing a and contracting diagonal b with D.
        size_dict: sd(&[(0, da), (1, db), (2, dc)]),
    };
    let mut rng = ChaCha8Rng::seed_from_u64(3);
    let c = DenseTensor::<f64>::random(vec![da, da, db, db], &mut rng);
    let dd = DenseTensor::<f64>::random(vec![db, dc], &mut rng);
    let got = contract_network(&net, vec![c.clone(), dd.clone()], &vec![(0, 1)]).unwrap();
    let want = naive_einsum(&net, &[c, dd]).unwrap();
    assert_eq!(got.shape(), want.shape());
    assert!(
        got.max_abs_diff(&want) < 1e-12,
        "diff={}",
        got.max_abs_diff(&want)
    );
}

/// Greedy planning, simulation, and execution must agree with the reference on diagonal networks.
#[test]
fn diagonal_end_to_end_via_greedy() {
    use arctn::paths::greedy::greedy;
    let (da, db, dc) = (3usize, 2usize, 4usize);
    let net = TensorNetwork {
        name: "diag_e2e".into(),
        inputs: vec![vec![0, 0, 1], vec![0, 2], vec![1, 2]], // A[a,a,b]·B[a,c]·C[b,c]
        output: vec![],
        size_dict: sd(&[(0, da), (1, db), (2, dc)]),
    };
    let (path, stats) = greedy(&net).unwrap();
    assert!(stats.log10_flops.is_finite());
    let mut rng = ChaCha8Rng::seed_from_u64(9);
    let a = DenseTensor::<f64>::random(vec![da, da, db], &mut rng);
    let b = DenseTensor::<f64>::random(vec![da, dc], &mut rng);
    let c = DenseTensor::<f64>::random(vec![db, dc], &mut rng);
    let got = contract_network(&net, vec![a.clone(), b.clone(), c.clone()], &path).unwrap();
    let want = naive_einsum(&net, &[a, b, c]).unwrap();
    assert!(
        got.max_abs_diff(&want) < 1e-12,
        "e2e diff={}",
        got.max_abs_diff(&want)
    );
}
