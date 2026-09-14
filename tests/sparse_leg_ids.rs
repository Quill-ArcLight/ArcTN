use arctn::paths::optimal::optimal_dp;
use arctn::tree::CTree;
use arctn::{simulate_path, TensorNetwork};

fn max_leg_network() -> TensorNetwork {
    TensorNetwork {
        name: "u32-max-leg".into(),
        inputs: vec![vec![u32::MAX], vec![u32::MAX]],
        output: vec![u32::MAX],
        size_dict: [(u32::MAX, 2)].into_iter().collect(),
    }
}

#[test]
fn max_leg_id_is_safe_in_simulation_tree_and_optimal_dp() {
    let net = max_leg_network();
    net.validate().unwrap();

    let path = vec![(0, 1)];
    let simulated = simulate_path(&net, &path).unwrap();
    let tree = CTree::from_path(&net, &path).unwrap();
    assert_eq!(tree.legs[tree.root], vec![u32::MAX]);
    assert_eq!(tree.total_log10_flops(&net), simulated.log10_flops);

    let (optimal_path, optimal_stats) = optimal_dp(&net, 8).unwrap();
    assert_eq!(optimal_path, path);
    assert_eq!(optimal_stats.log10_flops, simulated.log10_flops);
    let optimal_tree = CTree::from_path(&net, &optimal_path).unwrap();
    assert_eq!(optimal_tree.legs.last().unwrap(), &[u32::MAX]);
}

#[test]
fn unused_max_leg_dimension_is_rejected() {
    let net = TensorNetwork {
        name: "unused-u32-max-leg".into(),
        inputs: vec![vec![0]],
        output: vec![0],
        size_dict: [(0, 2), (u32::MAX, 2)].into_iter().collect(),
    };
    let err = net.validate().unwrap_err();
    assert!(err.contains("未被任何输入张量使用"), "{err}");
}
