//! Sliced execution must reject invalid specifications that could duplicate sums
//! or silently ignore inputs.

use arctn::{contract_network, contract_network_sliced, DenseTensor, TensorNetwork};

fn dot_network() -> TensorNetwork {
    TensorNetwork {
        name: "dot".into(),
        inputs: vec![vec![0], vec![0]],
        output: vec![],
        size_dict: [(0, 2)].into_iter().collect(),
    }
}

fn dot_tensors() -> Vec<DenseTensor<f64>> {
    vec![
        DenseTensor::from_data(vec![2], vec![1.0, 2.0]),
        DenseTensor::from_data(vec![2], vec![3.0, 5.0]),
    ]
}

#[test]
fn valid_slice_still_matches_unsliced_contraction() {
    let net = dot_network();
    let tensors = dot_tensors();
    let path = vec![(0, 1)];
    let direct = contract_network(&net, tensors.clone(), &path).unwrap();
    let sliced = contract_network_sliced(&net, &tensors, &path, &[0]).unwrap();
    assert_eq!(direct.data(), &[13.0]);
    assert_eq!(sliced.data(), direct.data());
}

#[test]
fn sliced_contraction_rejects_duplicate_and_unheld_legs() {
    let net = dot_network();
    let tensors = dot_tensors();
    let path = vec![(0, 1)];

    let duplicate = contract_network_sliced(&net, &tensors, &path, &[0, 0]).unwrap_err();
    assert!(duplicate.contains("重复"), "{duplicate}");

    // Network validation rejects unused dimensions before slicing.
    let mut unheld_net = net.clone();
    unheld_net.size_dict.insert(1, 2);
    let unheld = contract_network_sliced(&unheld_net, &tensors, &path, &[1]).unwrap_err();
    assert!(unheld.contains("未被任何输入张量使用"), "{unheld}");
}

#[test]
fn sliced_contraction_rejects_bad_tensor_count_and_shape() {
    let net = dot_network();
    let path = vec![(0, 1)];
    let mut extra = dot_tensors();
    extra.push(DenseTensor::scalar(7.0));
    let count_error = contract_network_sliced(&net, &extra, &path, &[0]).unwrap_err();
    assert!(count_error.contains("张量个数"), "{count_error}");

    let bad_shape = vec![
        DenseTensor::from_data(vec![1], vec![1.0]),
        DenseTensor::from_data(vec![2], vec![3.0, 5.0]),
    ];
    let shape_error = contract_network_sliced(&net, &bad_shape, &path, &[0]).unwrap_err();
    assert!(shape_error.contains("shape"), "{shape_error}");
}

#[test]
fn sliced_contraction_rejects_trace_leg() {
    let net = TensorNetwork {
        name: "trace".into(),
        inputs: vec![vec![0, 0]],
        output: vec![],
        size_dict: [(0, 2)].into_iter().collect(),
    };
    let tensors = vec![DenseTensor::from_data(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0])];
    let err = contract_network_sliced(&net, &tensors, &vec![], &[0]).unwrap_err();
    assert!(err.contains("同一输入张量中重复"), "{err}");
}
