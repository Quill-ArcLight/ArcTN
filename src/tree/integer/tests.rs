use super::*;
use crate::tree::SsaPath;
#[cfg(feature = "integer-tree-cost")]
use crate::tree::{rotatable_nodes, RotScratch};
use std::collections::{BTreeMap, BTreeSet};

// Deliberately independent set-based replay: no planner/cache/cost helpers.
fn replay(net: &TensorNetwork, path: &SsaPath) -> (u128, u128) {
    let n = net.inputs.len();
    let mut live: BTreeMap<usize, (BTreeSet<LegId>, u128)> = net
        .inputs
        .iter()
        .enumerate()
        .map(|(id, legs)| {
            (
                id,
                (
                    legs.iter().copied().collect(),
                    legs.iter().map(|leg| net.size_dict[leg] as u128).product(),
                ),
            )
        })
        .collect();
    let output: BTreeSet<_> = net.output.iter().copied().collect();
    let (mut f, mut r) = (0, 0);
    for (step, &(a, b)) in path.iter().enumerate() {
        let (aa, ar) = live.remove(&a).unwrap();
        let (bb, br) = live.remove(&b).unwrap();
        let union: BTreeSet<_> = aa.union(&bb).copied().collect();
        let kept: BTreeSet<_> = union
            .iter()
            .copied()
            .filter(|l| output.contains(l) || live.values().any(|(axes, _)| axes.contains(l)))
            .collect();
        let work: u128 = union.iter().map(|leg| net.size_dict[leg] as u128).product();
        let write: u128 = kept.iter().map(|leg| net.size_dict[leg] as u128).product();
        f += work;
        r += ar + br + write;
        live.insert(n + step, (kept, write));
    }
    assert_eq!(live.len(), 1);
    assert_eq!(live.first_key_value().unwrap().1 .0, output);
    (f, r)
}

fn mixed_net() -> (TensorNetwork, SsaPath) {
    (
        TensorNetwork {
            name: "binary-hyperedge-repeat-output".into(),
            inputs: vec![
                vec![0, 1, 1],
                vec![0, 2],
                vec![2, 3],
                vec![3, 4],
                vec![0, 4],
            ],
            output: vec![0],
            size_dict: (0..5).map(|l| (l, 2)).collect(),
        },
        vec![(0, 1), (2, 5), (3, 6), (4, 7)],
    )
}

#[cfg(feature = "integer-tree-cost")]
fn assert_cache<const F: bool, const R: bool>(tree: &CTreeCore<F, R>, net: &TensorNetwork) {
    let exact = tree.integer_costs.as_ref().expect("small actual costs fit");
    let (f, r) = replay(net, &tree.to_path());
    assert_eq!(exact.flops, if F { f } else { 0 });
    assert_eq!(exact.read_write, if R { r } else { 0 });
    let expected = match tree.objective.kind() {
        ObjectiveKind::TotalFlops => f,
        ObjectiveKind::TotalReadWrite => r,
        ObjectiveKind::Weighted => f + 64 * r,
    };
    assert_eq!(exact.score, expected);
    for node in 0..tree.legs.len() {
        if F && tree.alive[node] && !tree.is_leaf(node) {
            let union: BTreeSet<_> = tree.legs[tree.left[node]]
                .iter()
                .chain(&tree.legs[tree.right[node]])
                .copied()
                .collect();
            assert_eq!(
                exact.steps[node],
                union.iter().map(|l| net.size_dict[l] as u128).product()
            );
        }
    }
}

#[cfg(feature = "integer-tree-cost")]
fn exercise<const F: bool, const R: bool>(objective: PlannerObjective) {
    let (net, path) = mixed_net();
    let mut tree = CTreeCore::<F, R>::from_path_with_objective(&net, &path, objective).unwrap();
    let mut scratch = RotScratch::new();
    for iteration in 0..96 {
        assert_cache(&tree, &net);
        let nodes = rotatable_nodes(&tree);
        let b = nodes[iteration % nodes.len()];
        let promote = iteration % 3 == 0;
        let before = tree.to_path();
        let proposal = tree
            .rotate_delta(&net, b, promote, &mut scratch)
            .unwrap()
            .unwrap();
        assert_eq!(before, tree.to_path(), "evaluation preserves tree topology");
        assert!(proposal.integer.is_some());
        tree.rotate_apply(b, promote, proposal);
        assert_cache(&tree, &net);
        if iteration % 11 == 0 {
            tree.rebuild_cost_sums(&net).unwrap();
            assert_cache(&tree, &net);
            tree.reconfigure_node(&net, tree.root, 4).unwrap();
            assert_cache(&tree, &net);
        }
    }
}

#[test]
#[cfg(feature = "integer-tree-cost")]
fn continuous_rotations_and_reconfiguration_match_integer_replay() {
    exercise::<true, true>(PlannerObjective::FIXED);
    exercise::<true, false>(PlannerObjective::new(3.5, 0.0).unwrap());
    exercise::<false, true>(PlannerObjective::new(0.0, 2.5).unwrap());
}

#[test]
#[cfg(feature = "integer-tree-cost")]
fn nonroot_reconfiguration_preserves_the_external_read_and_all_caches() {
    let net = TensorNetwork {
        name: "nonroot-binary-cycle".into(),
        inputs: vec![vec![0, 3], vec![0, 1], vec![1, 2], vec![2, 3], vec![4]],
        output: vec![4],
        size_dict: (0..5).map(|leg| (leg, 2)).collect(),
    };
    // The first four leaves form a non-root subtree. Start with the expensive
    // outer product of opposite vertices; the fifth leaf stays outside it.
    let path = vec![(0, 2), (1, 5), (3, 6), (4, 7)];
    let mut tree = CTreeCore::<true, true>::from_path(&net, &path).unwrap();
    let root = tree.root;
    let root_size = tree.integer_costs.as_ref().unwrap().tensor_sizes[root];
    let before = tree.integer_costs.as_ref().unwrap().score;
    assert_ne!(7, root);
    assert_cache(&tree, &net);
    let outcome = tree.reconfigure_node(&net, 7, 4).unwrap();
    assert!(outcome.applied);
    assert!(!outcome.fell_back);
    assert_eq!(tree.root, root);
    assert_cache(&tree, &net);
    let costs = tree.integer_costs.as_ref().unwrap();
    assert!(costs.score < before);
    assert_eq!(costs.tensor_sizes[root], root_size);
    assert_eq!(costs.tensor_sizes[7], 1);
    tree.rebuild_cost_sums(&net).unwrap();
    assert_cache(&tree, &net);
}

#[test]
#[cfg(feature = "integer-tree-cost")]
fn read_write_counts_raw_repeated_leaf_and_internal_multiplicity() {
    let net = TensorNetwork {
        name: "multiplicity".into(),
        inputs: vec![vec![0, 0, 1], vec![1, 2], vec![2]],
        output: vec![],
        size_dict: (0..3).map(|l| (l, 2)).collect(),
    };
    let path = vec![(0, 1), (2, 3)];
    let tree = CTreeCore::<true, true>::from_path(&net, &path).unwrap();
    // Reads/writes: (8+4+2) + (2+2+1) = 19; F=8+2=10.
    let c = tree.integer_costs.as_ref().unwrap();
    assert_eq!((c.flops, c.read_write, c.score), (10, 19, 1226));
    assert_eq!(c.tensor_sizes[0], 8);
    assert_eq!(c.tensor_sizes[3], 2);
    assert_cache(&tree, &net);
}

#[test]
#[cfg(feature = "integer-tree-cost")]
fn large_global_label_count_does_not_disable_small_tree_costs() {
    let n = 160usize;
    let net = TensorNetwork {
        name: "long-chain".into(),
        inputs: (0..n)
            .map(|i| {
                let mut a = vec![];
                if i > 0 {
                    a.push((i - 1) as u32);
                }
                if i + 1 < n {
                    a.push(i as u32);
                }
                a
            })
            .collect(),
        output: vec![],
        size_dict: (0..n - 1).map(|l| (l as u32, 2)).collect(),
    };
    let mut path = vec![(0, 1)];
    for i in 2..n {
        path.push((i, n + i - 2));
    }
    let tree = CTreeCore::<true, true>::from_path(&net, &path).unwrap();
    assert!(net.size_dict.len() > 128);
    assert_cache(&tree, &net);
}

#[cfg(feature = "integer-tree-cost")]
fn cycle() -> (TensorNetwork, SsaPath) {
    let edges: Vec<Vec<u32>> = (0..4)
        .map(|edge| (edge * 31..(edge + 1) * 31).collect())
        .collect();
    (
        TensorNetwork {
            name: "weighted-overflow-rotation".into(),
            inputs: vec![
                edges[0].iter().chain(&edges[3]).copied().collect(),
                edges[0].iter().chain(&edges[1]).copied().collect(),
                edges[1].iter().chain(&edges[2]).copied().collect(),
                edges[2].iter().chain(&edges[3]).copied().collect(),
            ],
            output: vec![],
            size_dict: (0..124).map(|l| (l, 2)).collect(),
        },
        vec![(0, 1), (2, 4), (3, 5)],
    )
}

#[test]
#[cfg(feature = "integer-tree-cost")]
fn overflowing_rotation_rebuilds_legacy_and_evaluates_same_candidate() {
    let (net, path) = cycle();
    let mut tree = CTreeCore::<true, true>::from_path(&net, &path).unwrap();
    assert!(tree.integer_costs.is_some());
    let mut legacy = tree.clone();
    legacy.integer_enabled = false;
    legacy.rebuild_cost_sums(&net).unwrap();
    let mut scratch = RotScratch::new();
    let change = tree
        .rotate_delta(&net, 4, false, &mut scratch)
        .unwrap()
        .unwrap();
    assert!(change.fell_back && change.integer.is_none());
    assert!(tree.integer_costs.is_none() && !tree.integer_enabled);
    assert_eq!(tree.to_path(), path);
    let old = legacy
        .rotate_delta(&net, 4, false, &mut scratch)
        .unwrap()
        .unwrap();
    assert_eq!(
        change.next_flops_log2.to_bits(),
        old.next_flops_log2.to_bits()
    );
    assert_eq!(
        change.next_read_write_log2.to_bits(),
        old.next_read_write_log2.to_bits()
    );
    assert_eq!(
        tree.search_score_log2().to_bits(),
        legacy.search_score_log2().to_bits()
    );
    tree.rotate_apply(4, false, change);
    legacy.rotate_apply(4, false, old);
    assert_eq!(tree.to_path(), legacy.to_path());
    assert_eq!(
        tree.search_score_log2().to_bits(),
        legacy.search_score_log2().to_bits()
    );
    tree.rebuild_cost_sums(&net).unwrap();
    assert!(
        tree.integer_costs.is_none(),
        "fallback is permanent for this tree"
    );
}

#[test]
#[cfg(feature = "integer-tree-cost")]
fn rejected_reconfiguration_fallback_refreshes_replica_before_next_move() {
    use crate::paths::optimal::ReconfigurationPlan;
    use crate::tree::{run_segment, Replica};
    use rand::SeedableRng;
    let (net, path) = cycle();
    let tree = CTreeCore::<true, true>::from_path(&net, &path).unwrap();
    let mut rep = Replica {
        score_log2: tree.search_score_log2(),
        flops_linear: 0.0,
        best_total: tree.search_score_log2(),
        best_integer: tree.integer_score(),
        best_path: path.clone(),
        rotatable: rotatable_nodes(&tree),
        tree,
        rng: rand_chacha::ChaCha8Rng::seed_from_u64(17),
    };
    // This legal alternative exceeds the weighted integer bound and is worse;
    // fallback must complete, even though its topology is never committed.
    let alternate = vec![(0, 2), (1, 4), (3, 5)];
    let (log2_flops, step_legs) =
        crate::path::simulate_path_flops_log2_and_legs(&net, &alternate).unwrap();
    let outcome = rep
        .tree
        .reconfigure_candidate(
            &net,
            6,
            vec![0, 1, 2, 3],
            vec![4, 5, 6],
            ReconfigurationPlan {
                path: alternate,
                log2_flops,
                step_legs,
            },
        )
        .unwrap();
    assert!(!outcome.applied && outcome.fell_back);
    assert_eq!(rep.tree.to_path(), path);
    if outcome.applied || outcome.fell_back {
        rep.refresh_current_costs();
    }
    assert_eq!(
        rep.score_log2.to_bits(),
        rep.tree.search_score_log2().to_bits()
    );
    run_segment(&mut rep, &net, 0.1, 1, 0, 4, None).unwrap();
    assert_eq!(
        rep.score_log2.to_bits(),
        rep.tree.search_score_log2().to_bits()
    );
}

#[test]
#[cfg(feature = "integer-tree-cost")]
fn fallback_acceptance_keeps_legacy_probability_and_random_draws() {
    use rand::{Rng, RngCore, SeedableRng};
    let (net, path) = cycle();
    let mut tree = CTreeCore::<true, true>::from_path(&net, &path).unwrap();
    let change = tree
        .rotate_delta(&net, 4, false, &mut RotScratch::new())
        .unwrap()
        .unwrap();
    assert!(change.fell_back);
    let current = tree.search_score_log2();
    let next = change.score_log2(PlannerObjective::FIXED);
    for seed in 0..16 {
        let mut actual = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let mut expected = actual.clone();
        let relative = ((next - current) * std::f64::consts::LN_2).exp_m1();
        let accepted = next <= current || expected.gen::<f64>() < (-relative / 1e12).exp();
        assert_eq!(
            tree.accept_rotation_relative(change, 1e12, &mut actual),
            accepted
        );
        assert_eq!(actual.next_u64(), expected.next_u64());
        let mut actual = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let mut expected = actual.clone();
        let accepted = next <= current || expected.gen::<f64>() < (-0.02 * (next - current)).exp();
        assert_eq!(
            tree.accept_rotation_log(change, 0.02, &mut actual),
            accepted
        );
        assert_eq!(actual.next_u64(), expected.next_u64());
    }
}

#[test]
fn binary_search_entry_points_return_independently_replayable_costs() {
    use crate::tree::{
        anneal_path_with_objective, temper_path_with_objective, treesa_path_with_objective,
    };
    let (net, path) = mixed_net();
    for objective in [
        PlannerObjective::FIXED,
        PlannerObjective::new(3.5, 0.0).unwrap(),
        PlannerObjective::new(0.0, 2.5).unwrap(),
    ] {
        let results = [
            anneal_path_with_objective(&net, &path, 48, 0, 0.2, 0.01, objective).unwrap(),
            temper_path_with_objective(&net, &path, 2, 2, 24, 0.01, 0.2, 7, 4, 0, objective)
                .unwrap(),
            treesa_path_with_objective(&net, &path, 2, 0.1, 2.0, 3, 2, 7, 4, 0, objective).unwrap(),
        ];
        for (result, stats) in results {
            let (f, r) = replay(&net, &result);
            assert!((stats.log10_flops - (f as f64).log10()).abs() < 1e-12);
            assert!((stats.log2_read_write - (r as f64).log2()).abs() < 1e-12);
        }
    }
}

#[cfg(feature = "integer-tree-cost")]
fn outer_prefix(labels: usize) -> (TensorNetwork, SsaPath) {
    let blocks: Vec<Vec<u32>> = (0..4)
        .map(|i| {
            (i * labels / 4..(i + 1) * labels / 4)
                .map(|j| j as u32)
                .collect()
        })
        .collect();
    let inputs = blocks.iter().chain(&blocks).cloned().collect();
    (
        TensorNetwork {
            name: "actual-total-boundary".into(),
            inputs,
            output: vec![],
            size_dict: (0..labels).map(|l| (l as u32, 2)).collect(),
        },
        vec![(0, 1), (2, 8), (3, 9), (7, 10), (6, 11), (5, 12), (4, 13)],
    )
}

#[test]
#[cfg(feature = "integer-tree-cost")]
fn actual_flops_sum_not_just_largest_power_controls_fallback() {
    let objective = PlannerObjective::new(1.0, 0.0).unwrap();
    for (labels, fits) in [(126, true), (127, false), (128, false)] {
        let (net, path) = outer_prefix(labels);
        let tree =
            CTreeCore::<true, false>::from_path_with_objective(&net, &path, objective).unwrap();
        assert_eq!(tree.integer_costs.is_some(), fits);
        assert!(tree.total_cost_log2(&net).is_finite());
        if fits {
            assert_cache(&tree, &net);
        }
    }
}

#[test]
fn integer_tree_feature_controls_cache_selection_and_diagnostics() {
    fn check<const F: bool, const R: bool>(objective: PlannerObjective) {
        let (net, path) = mixed_net();
        assert!(crate::integer_cost::supports(&net, objective));
        let tree = CTreeCore::<F, R>::from_path_with_objective(&net, &path, objective).unwrap();
        let enabled = cfg!(feature = "integer-tree-cost");
        assert_eq!(tree.integer_enabled, enabled);
        assert_eq!(tree.integer_costs.is_some(), enabled);
        assert_eq!(tree.binary_dimensions, enabled);
    }
    check::<true, false>(PlannerObjective::new(1.0, 0.0).unwrap());
    check::<false, true>(PlannerObjective::new(0.0, 1.0).unwrap());
    check::<true, true>(PlannerObjective::FIXED);
    let expected = if cfg!(feature = "integer-tree-cost") {
        (
            "checked-u128-with-linear-f64-fallback",
            "fallback-to-linear-f64-then-reject-nonfinite",
        )
    } else {
        ("linear-f64-total-flops", "reject-cell")
    };
    assert_eq!(
        (
            crate::tree::PURE_FLOPS_NUMERIC_MODE,
            crate::tree::PURE_FLOPS_OVERFLOW_POLICY,
        ),
        expected
    );
}

#[test]
fn unsupported_dimensions_and_weights_keep_legacy_caches() {
    let (mut net, path) = mixed_net();
    let custom = PlannerObjective::new(1.0, 63.0).unwrap();
    assert!(
        CTreeCore::<true, true>::from_path_with_objective(&net, &path, custom)
            .unwrap()
            .integer_costs
            .is_none()
    );
    net.size_dict.insert(0, 3);
    assert!(CTreeCore::<true, true>::from_path(&net, &path)
        .unwrap()
        .integer_costs
        .is_none());
}

#[test]
#[cfg(feature = "integer-tree-cost")]
fn exact_ordering_retains_low_bits_and_probabilities_keep_original_domains() {
    let high = 1u128 << 100;
    assert!(improves(high, high + 1, 0.0));
    assert!(!improves(high + 1, high, 0.0));
    assert!(relative_change(high, high + 1) > 0.0);
    assert_eq!(relative_change(100, 120), 0.2);
    assert_eq!(relative_change(100, 80), -0.2);
    let (net, path) = mixed_net();
    let mut tree = CTreeCore::<true, true>::from_path(&net, &path).unwrap();
    let change = tree
        .rotate_delta(&net, 5, false, &mut RotScratch::new())
        .unwrap()
        .unwrap();
    let old = tree.integer_score().unwrap() as f64;
    let next = change.integer.unwrap().score as f64;
    assert!(
        (tree.rotation_relative_change(change, || panic!("integer branch must be lazy"))
            - (next - old) / old)
            .abs()
            < 1e-15
    );
    assert!(
        (tree.rotation_log_change(change, || panic!("integer branch must be lazy"))
            - (next.log2() - old.log2()))
        .abs()
            < 1e-14
    );
}
