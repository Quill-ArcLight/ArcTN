use arctn::{contract_network, random_greedy, DenseTensor, TensorNetwork};

fn main() -> Result<(), String> {
    let net = TensorNetwork {
        name: "matrix_product".into(),
        inputs: vec![vec![0, 1], vec![1, 2]],
        output: vec![0, 2],
        size_dict: [(0, 2), (1, 3), (2, 2)].into_iter().collect(),
    };
    let (path, stats) = random_greedy(&net, 16, 0)?;
    let a = DenseTensor::from_data(vec![2, 3], (0..6).map(f64::from).collect());
    let b = DenseTensor::from_data(vec![3, 2], (0..6).map(f64::from).collect());
    let output = contract_network(&net, vec![a, b], &path)?;
    println!("path: {path:?}; log10 FLOPs: {}", stats.log10_flops);
    println!("output: {:?}", output.data());
    Ok(())
}
