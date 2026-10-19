//! Running a compiled plan stage by stage must give the same bits as `execute`.

use std::collections::HashMap;

use arctn::network::TensorNetwork;
use arctn::tensor::{DenseTensor, Scalar};
use arctn::{contract_network, greedy, naive_einsum, CompiledContraction};
use num_complex::Complex64;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn sd(pairs: &[(u32, usize)]) -> HashMap<u32, usize> {
    pairs.iter().copied().collect()
}

fn random_inputs<T: Scalar>(net: &TensorNetwork, seed: u64) -> Vec<DenseTensor<T>> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    net.inputs
        .iter()
        .map(|legs| DenseTensor::random(legs.iter().map(|&l| net.dim(l)).collect(), &mut rng))
        .collect()
}

/// Run the plan stage by stage, checking slots, legs and shapes on the way.
fn replay<T: Scalar>(
    net: &TensorNetwork,
    compiled: &CompiledContraction,
    inputs: &[DenseTensor<T>],
) -> DenseTensor<T> {
    let mut slot_legs: Vec<Option<Vec<u32>>> = Vec::new();
    let mut slots: Vec<Option<DenseTensor<T>>> = Vec::new();
    for (prep, t) in compiled.input_preps().iter().zip(inputs) {
        let prepared = prep.execute(t).unwrap();
        assert_eq!(prepared.shape(), prep.shape());
        slot_legs.push(Some(prep.legs().to_vec()));
        slots.push(Some(prepared));
    }
    for (s, step) in compiled.pair_steps().iter().enumerate() {
        assert_eq!(step.result_slot(), compiled.n_inputs() + s);
        assert_eq!(slots.len(), step.result_slot());
        let left = slots[step.left_slot()].take().unwrap();
        let right = slots[step.right_slot()].take().unwrap();
        assert_eq!(
            slot_legs[step.left_slot()].take().unwrap(),
            step.left_legs()
        );
        assert_eq!(
            slot_legs[step.right_slot()].take().unwrap(),
            step.right_legs()
        );
        let dims: Vec<usize> = step.result_legs().iter().map(|&l| net.dim(l)).collect();
        assert_eq!(dims, step.result_shape());
        let out = step.execute(&left, &right).unwrap();
        assert_eq!(out.shape(), step.result_shape());
        slot_legs.push(Some(step.result_legs().to_vec()));
        slots.push(Some(out));
    }
    let fin = compiled.finalize();
    assert_eq!(slot_legs[fin.slot()].take().unwrap(), fin.input_legs());
    let last = slots[fin.slot()].take().unwrap();
    assert!(slots.iter().all(Option::is_none), "仅收尾张量应存活");
    let out = fin.execute(&last).unwrap();
    assert_eq!(out.shape(), fin.output_shape());
    out
}

/// Has a diagonal, a trace, a dangling leg, a batch leg and a hyperedge.
fn mixed_network() -> TensorNetwork {
    TensorNetwork {
        name: "pair-steps-mixed".into(),
        inputs: vec![
            vec![0, 0, 1],
            vec![0, 2, 3, 3],
            vec![1, 2, 4, 5],
            vec![4, 6, 9],
            vec![6, 8, 9],
            vec![9, 10],
        ],
        output: vec![8, 6, 10],
        size_dict: sd(&[
            (0, 3),
            (1, 2),
            (2, 3),
            (3, 2),
            (4, 2),
            (5, 3),
            (6, 2),
            (8, 3),
            (9, 2),
            (10, 3),
        ]),
    }
}

#[test]
fn replay_matches_execute_bitwise_f64_and_complex() {
    let net = mixed_network();
    let (greedy_path, _) = greedy(&net).unwrap();
    let paths = vec![greedy_path, vec![(0, 1), (2, 6), (3, 4), (8, 5), (7, 9)]];
    for path in paths {
        let compiled = CompiledContraction::compile(&net, &path).unwrap();
        assert_eq!(compiled.pair_steps().len(), path.len());
        for (step, &(l, r)) in compiled.pair_steps().iter().zip(&path) {
            assert_eq!((step.left_slot(), step.right_slot()), (l, r));
        }

        let real = random_inputs::<f64>(&net, 7);
        let want = compiled.execute(&real).unwrap();
        let got = replay(&net, &compiled, &real);
        assert_eq!(got.shape(), want.shape());
        assert_eq!(got.data(), want.data(), "逐步执行须与整网执行逐位一致");
        let reference = naive_einsum(&net, &real).unwrap();
        for (g, r) in got.data().iter().zip(reference.data()) {
            assert!((g - r).abs() < 1e-9 * (1.0 + r.abs()));
        }

        let complex = random_inputs::<Complex64>(&net, 11);
        let want = compiled.execute(&complex).unwrap();
        let got = replay(&net, &compiled, &complex);
        assert_eq!(got.data(), want.data(), "逐步执行须与整网执行逐位一致");
    }
}

#[test]
fn two_node_fixed_path_step_matches_contract_network() {
    // T0[i, k] · T1[k, j] -> [i, j]
    let net = TensorNetwork {
        name: "pair-steps-two-node".into(),
        inputs: vec![vec![0, 1], vec![1, 2]],
        output: vec![0, 2],
        size_dict: sd(&[(0, 3), (1, 4), (2, 5)]),
    };
    let path = vec![(0, 1)];
    let compiled = CompiledContraction::compile(&net, &path).unwrap();
    let step = &compiled.pair_steps()[0];
    assert_eq!(step.gemm_dims(), (1, 3, 4, 5));
    assert_eq!(step.result_legs(), &[0, 2]);
    assert!(compiled.input_preps().iter().all(|p| p.is_identity()));
    assert!(compiled.finalize().is_identity());

    let inputs = random_inputs::<f64>(&net, 3);
    let got = step.execute(&inputs[0], &inputs[1]).unwrap();
    let want = contract_network(&net, inputs, &path).unwrap();
    assert_eq!(got.data(), want.data());
}

#[test]
fn single_tensor_network_uses_only_finalize() {
    let net = TensorNetwork {
        name: "pair-steps-single".into(),
        inputs: vec![vec![0, 1, 2]],
        output: vec![2, 0],
        size_dict: sd(&[(0, 2), (1, 3), (2, 4)]),
    };
    let compiled = CompiledContraction::compile(&net, &vec![]).unwrap();
    assert!(compiled.pair_steps().is_empty());
    assert_eq!(compiled.finalize().slot(), 0);
    assert!(!compiled.finalize().is_identity());
    let inputs = random_inputs::<f64>(&net, 5);
    let got = replay(&net, &compiled, &inputs);
    assert_eq!(got.data(), compiled.execute(&inputs).unwrap().data());
}

#[test]
fn stage_execution_rejects_mismatched_shapes() {
    let net = mixed_network();
    let (path, _) = greedy(&net).unwrap();
    let compiled = CompiledContraction::compile(&net, &path).unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(1);

    let step = &compiled.pair_steps()[0];
    let mut wrong = step.left_shape().to_vec();
    wrong[0] += 1;
    let left = DenseTensor::<f64>::random(wrong, &mut rng);
    let right = DenseTensor::<f64>::random(step.right_shape().to_vec(), &mut rng);
    let err = step.execute(&left, &right).unwrap_err();
    assert!(err.contains("左操作数"), "unexpected error: {err}");

    let prep = &compiled.input_preps()[0];
    let err = prep
        .execute(&DenseTensor::<f64>::random(vec![2, 2], &mut rng))
        .unwrap_err();
    assert!(err.contains("输入张量"), "unexpected error: {err}");

    let fin = compiled.finalize();
    let err = fin
        .execute(&DenseTensor::<f64>::random(vec![7], &mut rng))
        .unwrap_err();
    assert!(err.contains("收尾张量"), "unexpected error: {err}");
}
