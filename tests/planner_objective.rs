use arctn::{PathStats, PlannerObjective, FLOPS_WEIGHT, READ_WRITE_WEIGHT};

fn comparison_net() -> arctn::network::TensorNetwork {
    arctn::network::TensorNetwork {
        name: "fixed-objective-comparison".into(),
        inputs: vec![vec![0, 1, 2, 3], vec![0, 1, 4], vec![2], vec![3, 4]],
        output: vec![],
        size_dict: [(0, 5), (1, 4), (2, 2), (3, 4), (4, 4)]
            .into_iter()
            .collect(),
    }
}

fn stats(log2_flops: f64, log2_read_write: f64) -> PathStats {
    PathStats {
        log10_flops: log2_flops * std::f64::consts::LOG10_2,
        log2_max_size: 0.0,
        log2_max_contraction_size: 0.0,
        log2_total_size: 0.0,
        log2_read_write,
        log2_peak_size: 0.0,
    }
}

#[test]
fn fixed_identifier_and_weights_are_stable() {
    assert_eq!(PlannerObjective::default(), PlannerObjective::FIXED);
    assert_eq!(PlannerObjective::FIXED.as_str(), "flops_read_write");
    assert_eq!(PlannerObjective::FIXED.flops_weight(), 1.0);
    assert_eq!(PlannerObjective::FIXED.read_write_weight(), 64.0);
    assert_eq!(FLOPS_WEIGHT, 1.0);
    assert_eq!(READ_WRITE_WEIGHT, 64.0);
}

#[test]
fn fixed_score_matches_an_exact_small_value() {
    // F = 2^10 and RWC = 2^5. The expected value uses the default weights.
    let path = stats(10.0, 5.0);
    let expected = (2.0f64.powi(10) * FLOPS_WEIGHT + 2.0f64.powi(5) * READ_WRITE_WEIGHT).log2();
    let score = PlannerObjective::FIXED.score_path_log2(&path);
    assert!((score - expected).abs() < 1e-12);
}

#[test]
fn fixed_score_uses_full_read_write_complexity_not_total_output_size() {
    let mut path = stats(10.0, 5.0);
    let expected = PlannerObjective::FIXED.score_path_log2(&path);
    path.log2_total_size = 1_000.0;
    assert_eq!(
        PlannerObjective::FIXED.score_path_log2(&path).to_bits(),
        expected.to_bits()
    );
}

#[test]
fn fixed_score_remains_finite_beyond_the_linear_f64_range() {
    let path = stats(2_000.0, 1_995.0);
    let expected = 2_000.0 + 3.0f64.log2();
    let score = PlannerObjective::FIXED.score_path_log2(&path);
    assert!(score.is_finite());
    assert!((score - expected).abs() < 1e-12);
}

#[test]
fn sliced_score_multiplies_the_entire_fixed_score() {
    let path = stats(40.0, 35.0);
    let base = PlannerObjective::FIXED.score_path_log2(&path);
    let sliced = PlannerObjective::FIXED.score_sliced_log2(&path, 7.25);
    assert!((sliced - (base + 7.25)).abs() < 1e-12);
}

#[test]
fn fixed_score_can_prefer_higher_flops_when_read_write_is_lower() {
    let low_flops_high_read_write = stats(10.0, 20.0);
    let high_flops_low_read_write = stats(12.0, 0.0);

    assert!(high_flops_low_read_write.log10_flops > low_flops_high_read_write.log10_flops);
    assert!(
        PlannerObjective::FIXED.score_path_log2(&high_flops_low_read_write)
            < PlannerObjective::FIXED.score_path_log2(&low_flops_high_read_write)
    );
}

#[test]
fn fixed_score_replays_real_sliced_contraction_candidates() {
    use arctn::simulate_path;

    let net = comparison_net();
    let candidates = [
        (vec![(1, 3), (0, 4), (2, 5)], vec![0, 1]),
        (vec![(0, 1), (3, 4), (2, 5)], vec![2, 3]),
    ];

    let evaluate = |path: &Vec<(usize, usize)>, sliced: &Vec<u32>| {
        let mut per_slice_net = net.clone();
        for leg in sliced {
            per_slice_net.size_dict.insert(*leg, 1);
        }
        let per_slice = simulate_path(&per_slice_net, path).expect("candidate must be valid");
        let log2_n_slices: f64 = sliced.iter().map(|leg| net.log2_dim(*leg)).sum();
        (per_slice, log2_n_slices)
    };
    let a = evaluate(&candidates[0].0, &candidates[0].1);
    let b = evaluate(&candidates[1].0, &candidates[1].1);

    let linear = |value: f64| value.exp2();
    let close = |actual: f64, expected: f64| {
        assert!(
            (actual - expected).abs() < 1e-8,
            "actual={actual}, expected={expected}"
        );
    };

    let a_flops = linear(a.0.log10_flops / std::f64::consts::LOG10_2 + a.1);
    let b_flops = linear(b.0.log10_flops / std::f64::consts::LOG10_2 + b.1);
    let a_read_write = linear(a.0.log2_read_write + a.1);
    let b_read_write = linear(b.0.log2_read_write + b.1);
    let a_score = linear(PlannerObjective::FIXED.score_sliced_log2(&a.0, a.1));
    let b_score = linear(PlannerObjective::FIXED.score_sliced_log2(&b.0, b.1));

    close(a_flops, 520.0);
    close(a_read_write, 860.0);
    close(
        a_score,
        a_flops * FLOPS_WEIGHT + a_read_write * READ_WRITE_WEIGHT,
    );
    close(b_flops, 680.0);
    close(b_read_write, 928.0);
    close(
        b_score,
        b_flops * FLOPS_WEIGHT + b_read_write * READ_WRITE_WEIGHT,
    );
    assert!(a_score < b_score);
    assert!(a.0.log2_max_size <= 2.0 && b.0.log2_max_size <= 2.0);
}
