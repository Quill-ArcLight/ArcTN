use arctn::paths::greedy::{greedy, random_greedy};
use arctn::simplify::random_greedy_simplified;
use arctn::{simulate_path, TensorNetwork};

fn valid_network() -> TensorNetwork {
    TensorNetwork {
        name: "pathfinding-result-valid".into(),
        inputs: vec![vec![0, 1], vec![1, 2], vec![2, 3], vec![0, 3]],
        output: Vec::new(),
        size_dict: [(0, 2), (1, 3), (2, 2), (3, 4)].into_iter().collect(),
    }
}

fn missing_dim_network() -> TensorNetwork {
    TensorNetwork {
        name: "pathfinding-result-missing-dim".into(),
        inputs: vec![vec![0, 1], vec![1, 2]],
        output: vec![0, 2],
        size_dict: [(0, 2), (2, 2)].into_iter().collect(),
    }
}

#[test]
fn high_level_pathfinding_entries_return_legal_candidates() {
    let net = valid_network();
    let candidates = [
        greedy(&net).unwrap(),
        random_greedy(&net, 8, 7).unwrap(),
        random_greedy_simplified(&net, 8, 11).unwrap(),
    ];

    for (path, expected) in candidates {
        let actual = simulate_path(&net, &path).unwrap();
        assert_eq!(actual.log10_flops.to_bits(), expected.log10_flops.to_bits());
        assert_eq!(
            actual.log2_max_size.to_bits(),
            expected.log2_max_size.to_bits()
        );
        assert_eq!(
            actual.log2_max_contraction_size.to_bits(),
            expected.log2_max_contraction_size.to_bits()
        );
        assert_eq!(
            actual.log2_total_size.to_bits(),
            expected.log2_total_size.to_bits()
        );
        assert_eq!(
            actual.log2_read_write.to_bits(),
            expected.log2_read_write.to_bits()
        );
        assert_eq!(
            actual.log2_peak_size.to_bits(),
            expected.log2_peak_size.to_bits()
        );
    }
}

#[test]
fn high_level_pathfinding_entries_reject_a_malformed_network() {
    let net = missing_dim_network();

    assert!(greedy(&net).is_err());
    assert!(random_greedy(&net, 2, 0).is_err());
    assert!(random_greedy_simplified(&net, 2, 0).is_err());
}

#[test]
fn single_random_greedy_trial_matches_greedy() {
    let net = valid_network();
    let expected = greedy(&net).unwrap();

    for seed in [0, 7, u64::MAX] {
        let actual = random_greedy(&net, 1, seed).unwrap();
        assert_eq!(actual.0, expected.0);
        assert_eq!(
            actual.1.log10_flops.to_bits(),
            expected.1.log10_flops.to_bits()
        );
    }
}

#[test]
fn random_greedy_rejects_zero_trials() {
    let net = valid_network();

    assert!(random_greedy(&net, 0, 7).is_err());
}

#[test]
fn simplified_entries_preserve_their_legacy_zero_behavior() {
    let net = valid_network();

    assert_eq!(
        random_greedy_simplified(&net, 0, 11).unwrap().0,
        random_greedy_simplified(&net, 1, 11).unwrap().0
    );
}
