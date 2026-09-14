//! Correctness tests for execution, path search, cost accounting, traces, and JSON loading.

use num_complex::Complex64;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use arctn::network::TensorNetwork;
use arctn::path::{simulate_path, sorted_dedup, PathStats, SsaPath};
use arctn::paths::greedy::{greedy, random_greedy};
use arctn::paths::optimal::optimal_dp;
use arctn::tensor::{DenseTensor, Scalar};
use arctn::{contract_network, naive_einsum, PlannerObjective};

fn planner_score(stats: &PathStats) -> f64 {
    PlannerObjective::FIXED.score_path_log2(stats)
}

fn random_tensors<T: Scalar>(net: &TensorNetwork, seed: u64) -> Vec<DenseTensor<T>> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    net.inputs
        .iter()
        .map(|legs| {
            let shape: Vec<usize> = legs.iter().map(|&l| net.dim(l)).collect();
            DenseTensor::random(shape, &mut rng)
        })
        .collect()
}

fn check_path_matches_naive<T: Scalar>(net: &TensorNetwork, path: &SsaPath, seed: u64, tol: f64) {
    let tensors = random_tensors::<T>(net, seed);
    let expect = naive_einsum(net, &tensors).expect("naive 失败");
    let got = contract_network(net, tensors, path).expect("执行失败");
    let diff = expect.max_abs_diff(&got);
    assert!(
        diff < tol,
        "网络 {} 结果不符，max diff = {diff:e}",
        net.name
    );
}

// A compiled plan must match `contract_network` bit for bit.
// Cover f64, Complex64, exponent stripping, single tensors, and traces.
#[test]
fn compiled_matches_contract_network_bitwise() {
    use arctn::CompiledContraction;
    for seed in 0..25u64 {
        let mut rng = ChaCha8Rng::seed_from_u64(seed + 4200);
        let n = 3 + (seed as usize % 5);
        let net = TensorNetwork::random_connected(n, 2, &[2, 3, 4], (seed % 3) as usize, &mut rng);
        for (pi, path) in [
            greedy(&net).unwrap().0,
            random_greedy(&net, 8, seed).unwrap().0,
        ]
        .iter()
        .enumerate()
        {
            let cc = CompiledContraction::compile(&net, path).expect("编译失败");
            // Exact f64 agreement.
            let ts = random_tensors::<f64>(&net, seed * 11 + pi as u64);
            let base = contract_network(&net, ts.clone(), path).expect("基线执行失败");
            let got = cc.execute(&ts).expect("编译执行失败");
            assert_eq!(base.shape(), got.shape(), "seed={seed} 形状不符");
            for (x, y) in base.data().iter().zip(got.data()) {
                assert_eq!(x.to_bits(), y.to_bits(), "seed={seed} f64 非逐位一致");
            }
            // Exact Complex64 agreement.
            let tc = random_tensors::<Complex64>(&net, seed * 13 + pi as u64);
            let bac = contract_network(&net, tc.clone(), path).expect("c64 基线失败");
            let goc = cc.execute(&tc).expect("c64 编译执行失败");
            for (x, y) in bac.data().iter().zip(goc.data()) {
                assert_eq!(x.re.to_bits(), y.re.to_bits(), "seed={seed} c64 re 非逐位");
                assert_eq!(x.im.to_bits(), y.im.to_bits(), "seed={seed} c64 im 非逐位");
            }
            // The stripped f64 mantissa and exponent must reconstruct the result.
            let (mant, exp) = cc.execute_stripped(&ts).expect("strip 执行失败");
            let scale = 10f64.powf(exp);
            let mut maxrel = 0.0f64;
            for (x, y) in base.data().iter().zip(mant.data()) {
                let rebuilt = y * scale;
                let d = (x - rebuilt).abs();
                let r = d / x.abs().max(1e-300);
                maxrel = maxrel.max(r);
            }
            assert!(
                maxrel < 1e-9,
                "seed={seed} strip_exponent(f64) 重建误差 {maxrel:e}"
            );
            // The stripped complex mantissa and real exponent must reconstruct the result.
            let (mantc, expc) = cc.execute_stripped(&tc).expect("c64 strip 失败");
            let scalec = 10f64.powf(expc);
            let mut maxrelc = 0.0f64;
            for (x, y) in bac.data().iter().zip(mantc.data()) {
                let rebuilt = Complex64::new(y.re * scalec, y.im * scalec);
                let r = (x - rebuilt).norm() / x.norm().max(1e-300);
                maxrelc = maxrelc.max(r);
            }
            assert!(
                maxrelc < 1e-9,
                "seed={seed} strip_exponent(c64) 重建误差 {maxrelc:e}"
            );
        }
    }
}

#[test]
fn compiled_execute_rejects_wrong_shaped_input() {
    // Reject input-count and shape mismatches before entering unsafe GEMM code.
    use arctn::CompiledContraction;
    let mut sd = std::collections::HashMap::new();
    sd.insert(0u32, 2);
    sd.insert(1u32, 3);
    sd.insert(2u32, 4);
    let net = TensorNetwork {
        name: "ws".into(),
        inputs: vec![vec![0, 1], vec![1, 2]],
        output: vec![0, 2],
        size_dict: sd,
    };
    let path = vec![(0usize, 1usize)];
    let cc = CompiledContraction::compile(&net, &path).expect("编译");
    // Matching inputs are accepted.
    let good = random_tensors::<f64>(&net, 1);
    assert!(cc.execute(&good).is_ok());
    // Tensor 0 keeps rank two but changes its second dimension from 3 to 2.
    let mut bad = good.clone();
    bad[0] = DenseTensor::from_data(vec![2, 2], vec![0.0; 4]);
    assert!(cc.execute(&bad).is_err(), "错维输入必须被 execute 拒绝");
    assert!(
        cc.execute_stripped(&bad).is_err(),
        "错维输入必须被 execute_stripped 拒绝"
    );
    // The compiled and direct execution paths use the same validation contract.
    assert!(contract_network(&net, bad, &path).is_err());
}

#[test]
fn compiled_handles_single_tensor_and_trace_nets() {
    use arctn::CompiledContraction;
    // A single tensor uses an empty path and reduces residual legs.
    let mut sd = std::collections::HashMap::new();
    for l in 0..3u32 {
        sd.insert(l, 2);
    }
    let net1 = TensorNetwork {
        name: "single".into(),
        inputs: vec![vec![0, 1, 2]],
        output: vec![0, 2],
        size_dict: sd.clone(),
    };
    let cc1 = CompiledContraction::compile(&net1, &vec![]).expect("单张量编译");
    let ts1 = random_tensors::<f64>(&net1, 7);
    let base1 = contract_network(&net1, ts1.clone(), &vec![]).expect("单张量基线");
    let got1 = cc1.execute(&ts1).expect("单张量执行");
    for (x, y) in base1.data().iter().zip(got1.data()) {
        assert_eq!(x.to_bits(), y.to_bits());
    }
    // Repeated leg 3 in T0 is private and absent from the output, so it is traced.
    let mut sd2 = std::collections::HashMap::new();
    for l in [0, 1, 3] {
        sd2.insert(l, 2);
    }
    let net2 = TensorNetwork {
        name: "trace".into(),
        inputs: vec![vec![0, 3, 3], vec![0, 1]],
        output: vec![1],
        size_dict: sd2,
    };
    let path2 = vec![(0usize, 1usize)];
    let cc2 = CompiledContraction::compile(&net2, &path2).expect("trace 编译");
    let ts2 = random_tensors::<f64>(&net2, 9);
    let base2 = contract_network(&net2, ts2.clone(), &path2).expect("trace 基线");
    let got2 = cc2.execute(&ts2).expect("trace 执行");
    assert_eq!(base2.shape(), got2.shape());
    for (x, y) in base2.data().iter().zip(got2.data()) {
        assert_eq!(x.to_bits(), y.to_bits());
    }
}

#[test]
fn engine_matches_naive_random_nets() {
    for seed in 0..20u64 {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let n = 3 + (seed as usize % 4); // Three to six tensors.
        let net = TensorNetwork::random_connected(n, 2, &[2, 3], (seed % 3) as usize, &mut rng);
        let (gpath, _) = greedy(&net).unwrap();
        check_path_matches_naive::<f64>(&net, &gpath, seed * 7 + 1, 1e-9);
        check_path_matches_naive::<Complex64>(&net, &gpath, seed * 7 + 2, 1e-9);
        let (rpath, _) = random_greedy(&net, 8, seed).unwrap();
        check_path_matches_naive::<f64>(&net, &rpath, seed * 7 + 3, 1e-9);
        let (opath, _) = optimal_dp(&net, 26).expect("optimal 失败");
        check_path_matches_naive::<f64>(&net, &opath, seed * 7 + 4, 1e-9);
        check_path_matches_naive::<Complex64>(&net, &opath, seed * 7 + 5, 1e-9);
    }
}

#[test]
fn engine_matches_naive_grid() {
    let net = TensorNetwork::grid_2d(3, 3, 2);
    let (path, _) = greedy(&net).unwrap();
    check_path_matches_naive::<f64>(&net, &path, 11, 1e-8);
    check_path_matches_naive::<Complex64>(&net, &path, 12, 1e-8);
}

/// Exhaustive search over the same contraction space as the dynamic program.
fn brute_force_best(net: &TensorNetwork) -> f64 {
    fn legs_of(net: &TensorNetwork) -> Vec<Vec<u32>> {
        net.inputs.iter().map(|t| sorted_dedup(t)).collect()
    }
    fn share(a: &[u32], b: &[u32]) -> bool {
        a.iter().any(|l| b.binary_search(l).is_ok())
    }
    fn rec(
        net: &TensorNetwork,
        alive: &mut [(usize, Vec<u32>)],
        path: &mut SsaPath,
        next_id: usize,
        best: &mut f64,
    ) {
        if alive.len() == 1 {
            let stats = simulate_path(net, path).expect("枚举路径非法");
            let score = planner_score(&stats);
            if score < *best {
                *best = score;
            }
            return;
        }
        let mut pairs: Vec<(usize, usize)> = Vec::new();
        for i in 0..alive.len() {
            for j in (i + 1)..alive.len() {
                if share(&alive[i].1, &alive[j].1) {
                    pairs.push((i, j));
                }
            }
        }
        if pairs.is_empty() {
            for i in 0..alive.len() {
                for j in (i + 1)..alive.len() {
                    pairs.push((i, j));
                }
            }
        }
        for (i, j) in pairs {
            let (id_i, legs_i) = alive[i].clone();
            let (id_j, legs_j) = alive[j].clone();
            // The simulator performs the exact reference-count accounting.
            let merged = {
                let mut m = legs_i.clone();
                m.extend_from_slice(&legs_j);
                sorted_dedup(&m)
            };
            let mut next_alive: Vec<(usize, Vec<u32>)> = alive
                .iter()
                .enumerate()
                .filter(|(k, _)| *k != i && *k != j)
                .map(|(_, v)| v.clone())
                .collect();
            next_alive.push((next_id, merged));
            path.push((id_i.min(id_j), id_i.max(id_j)));
            rec(net, &mut next_alive, path, next_id + 1, best);
            path.pop();
        }
    }
    let mut alive: Vec<(usize, Vec<u32>)> = legs_of(net).into_iter().enumerate().collect();
    let mut best = f64::INFINITY;
    rec(net, &mut alive, &mut Vec::new(), net.n_tensors(), &mut best);
    best
}

#[test]
fn dp_is_optimal_on_tiny_nets() {
    for seed in 0..15u64 {
        let mut rng = ChaCha8Rng::seed_from_u64(seed + 100);
        let n = 3 + (seed as usize % 3); // 3..5
        let net = TensorNetwork::random_connected(n, 2, &[2, 3, 4], (seed % 2) as usize, &mut rng);
        let (_, stats) = optimal_dp(&net, 26).expect("optimal 失败");
        let best = brute_force_best(&net);
        assert!(
            (planner_score(&stats) - best).abs() < 1e-9,
            "网络 seed={seed}: DP={} 暴力={}",
            planner_score(&stats),
            best
        );
    }
}

#[test]
fn fixed_cost_model_hand_example() {
    // For A(2x4), B(4x8), C(8x3), the fixed objective selects A(BC).
    // FLOPs = 120, read/write complexity = 94, score = 120 + 64 * 94 = 6136.
    let net = TensorNetwork {
        name: "chain".into(),
        inputs: vec![vec![0, 1], vec![1, 2], vec![2, 3]],
        output: vec![0, 3],
        size_dict: [(0u32, 2usize), (1, 4), (2, 8), (3, 3)]
            .into_iter()
            .collect(),
    };
    let (path, stats) = optimal_dp(&net, 26).unwrap();
    assert!((planner_score(&stats) - 6136f64.log2()).abs() < 1e-12);
    assert!((planner_score(&stats) - brute_force_best(&net)).abs() < 1e-12);
    assert!((stats.log10_flops - 120f64.log10()).abs() < 1e-12);
    assert!((stats.log2_read_write - 94f64.log2()).abs() < 1e-12);
    // The selected path must also execute correctly.
    check_path_matches_naive::<f64>(&net, &path, 5, 1e-10);
}

#[test]
fn trace_legs() {
    // T0[a,a,b] is traced before contraction with T1[b].
    let net = TensorNetwork {
        name: "trace".into(),
        inputs: vec![vec![0, 0, 1], vec![1]],
        output: vec![],
        size_dict: [(0u32, 3usize), (1, 4)].into_iter().collect(),
    };
    let path = vec![(0, 1)];
    check_path_matches_naive::<f64>(&net, &path, 21, 1e-10);
    check_path_matches_naive::<Complex64>(&net, &path, 22, 1e-10);
}

#[test]
fn single_tensor_network() {
    // Sum a residual leg from T[a,b] to produce output [a].
    let net = TensorNetwork {
        name: "single".into(),
        inputs: vec![vec![0, 1]],
        output: vec![0],
        size_dict: [(0u32, 3usize), (1, 5)].into_iter().collect(),
    };
    check_path_matches_naive::<f64>(&net, &vec![], 31, 1e-10);
}

#[test]
fn hyperedge_batch() {
    // Leg 0 is a hyperedge shared by T0[a,b], T1[a,c], and T2[a].
    // Contracting T0 with T1 must retain it as a batch leg for T2.
    let net = TensorNetwork {
        name: "hyper".into(),
        inputs: vec![vec![0, 1], vec![0, 2], vec![0]],
        output: vec![],
        size_dict: [(0u32, 3usize), (1, 2), (2, 4)].into_iter().collect(),
    };
    for path in [vec![(0, 1), (2, 3)], vec![(1, 2), (0, 3)]] {
        check_path_matches_naive::<f64>(&net, &path, 41, 1e-10);
        check_path_matches_naive::<Complex64>(&net, &path, 42, 1e-10);
    }
}

#[test]
fn sliced_contraction_matches_naive() {
    use arctn::slice::contract_network_sliced;
    for seed in 0..10u64 {
        let mut rng = ChaCha8Rng::seed_from_u64(seed + 200);
        let net = TensorNetwork::random_connected(5, 3, &[2, 3], 1, &mut rng);
        let (path, _) = greedy(&net).unwrap();
        // Slice up to two contracted legs.
        let mut cand: Vec<u32> = net
            .size_dict
            .keys()
            .copied()
            .filter(|l| !net.output.contains(l))
            .collect();
        cand.sort_unstable();
        let sliced = &cand[..2.min(cand.len())];
        let tensors = random_tensors::<f64>(&net, seed * 13 + 1);
        let expect = naive_einsum(&net, &tensors).unwrap();
        let got = contract_network_sliced(&net, &tensors, &path, sliced).unwrap();
        let diff = expect.max_abs_diff(&got);
        assert!(diff < 1e-9, "seed={seed} 切片收缩不符 diff={diff:e}");
    }
}

#[test]
fn find_slices_reduces_peak() {
    use arctn::slice::{contract_network_sliced, find_slices};
    let net = TensorNetwork::grid_2d(3, 4, 2);
    let (path, stats) = greedy(&net).unwrap();
    let target = stats.log2_max_size - 1.0;
    let sr = find_slices(&net, &path, target).expect("找不到切片方案");
    assert!(!sr.legs.is_empty());
    assert!(sr.per_slice.log2_max_size <= target + 1e-9);
    // The reported total sliced FLOPs must be finite.
    assert!(sr.log10_flops_total.is_finite());
    // Sliced execution must preserve the result.
    let tensors = random_tensors::<f64>(&net, 77);
    let expect = naive_einsum(&net, &tensors).unwrap();
    let got = contract_network_sliced(&net, &tensors, &path, &sr.legs).unwrap();
    assert!(expect.max_abs_diff(&got) < 1e-9);
}

#[test]
fn reconf_preserves_result_and_never_worsens() {
    use arctn::tree::reconfigure_path;
    for seed in 0..12u64 {
        let mut rng = ChaCha8Rng::seed_from_u64(seed + 300);
        let n = 4 + (seed as usize % 3); // 4..6
        let net = TensorNetwork::random_connected(n, 3, &[2, 3], (seed % 3) as usize, &mut rng);
        let (path, before) = greedy(&net).unwrap();
        let (new_path, after) = reconfigure_path(&net, &path, 8, 10).expect("reconf 失败");
        assert!(planner_score(&after) <= planner_score(&before) + 1e-9);
        // Reconfiguration must preserve the contraction result.
        check_path_matches_naive::<f64>(&net, &new_path, seed * 17 + 1, 1e-9);
        check_path_matches_naive::<Complex64>(&net, &new_path, seed * 17 + 2, 1e-9);
    }
    // Reconfiguration reaches the exact optimum on this small network.
    let mut rng = ChaCha8Rng::seed_from_u64(999);
    let net = TensorNetwork::random_connected(6, 2, &[2, 3, 4], 1, &mut rng);
    let (gpath, _) = greedy(&net).unwrap();
    let (_, rstats) = reconfigure_path(&net, &gpath, 8, 10).unwrap();
    let (_, ostats) = optimal_dp(&net, 26).unwrap();
    assert!((planner_score(&rstats) - planner_score(&ostats)).abs() < 1e-9);
}

#[test]
fn bisect_produces_valid_paths() {
    use arctn::paths::bisect::bisect;
    for seed in 0..8u64 {
        let mut rng = ChaCha8Rng::seed_from_u64(seed + 400);
        let net = TensorNetwork::random_connected(6, 3, &[2, 3], (seed % 2) as usize, &mut rng);
        let (path, _) = bisect(&net, 8, seed, 3).expect("bisect 失败");
        check_path_matches_naive::<f64>(&net, &path, seed * 19 + 1, 1e-9);
    }
    // On a larger network, verify path length and finite simulated cost.
    let mut rng = ChaCha8Rng::seed_from_u64(888);
    let net = TensorNetwork::rand_regular(60, 3, 2, &mut rng);
    let (path, stats) = bisect(&net, 16, 5, 12).unwrap();
    assert_eq!(path.len(), net.n_tensors() - 1);
    assert!(stats.log10_flops.is_finite());
}

#[test]
fn slice_and_reconf_matches_naive() {
    use arctn::slice::{contract_network_sliced, slice_and_reconf};
    let net = TensorNetwork::grid_2d(3, 4, 2);
    let (path, stats) = greedy(&net).unwrap();
    let target = stats.log2_max_size - 1.0;
    let (new_path, sr) = slice_and_reconf(&net, &path, target, 3, 8).expect("失败");
    assert!(sr.per_slice.log2_max_size <= target + 1e-9);
    let tensors = random_tensors::<f64>(&net, 555);
    let expect = naive_einsum(&net, &tensors).unwrap();
    let got = contract_network_sliced(&net, &tensors, &new_path, &sr.legs).unwrap();
    assert!(expect.max_abs_diff(&got) < 1e-9);
}

#[test]
fn treesa_preserves_result_never_worsens_and_deterministic() {
    use arctn::tree::treesa_path;
    for seed in 0..4u64 {
        let mut rng = ChaCha8Rng::seed_from_u64(seed + 1700);
        let n = 5 + (seed as usize % 3);
        let net = TensorNetwork::random_connected(n, 3, &[2, 3], (seed % 2) as usize, &mut rng);
        let (path, before) = greedy(&net).unwrap();
        let (tp, after) = treesa_path(&net, &path, 4, 0.01, 15.0, 16, 4, 100, 6, seed).unwrap();
        assert!(planner_score(&after) <= planner_score(&before) + 1e-9);
        check_path_matches_naive::<f64>(&net, &tp, seed * 59 + 1, 1e-9);
        check_path_matches_naive::<Complex64>(&net, &tp, seed * 59 + 2, 1e-9);
        let (tp2, _) = treesa_path(&net, &path, 4, 0.01, 15.0, 16, 4, 100, 6, seed).unwrap();
        assert_eq!(tp, tp2, "seed={seed} treesa 不确定");
    }
}

#[test]
fn slice_temper_reaches_target_and_matches_naive() {
    use arctn::slice::{contract_network_sliced, slice_temper};
    let net = TensorNetwork::grid_2d(3, 4, 2);
    let (path, stats) = greedy(&net).unwrap();
    let target = stats.log2_max_size - 1.0;
    let (new_path, sr) =
        slice_temper(&net, &path, target, 7, 4, 4, 1_500, 6).expect("slice_temper 失败");
    assert!(sr.per_slice.log2_max_size <= target + 1e-9);
    let tensors = random_tensors::<f64>(&net, 777);
    let expect = naive_einsum(&net, &tensors).unwrap();
    let got = contract_network_sliced(&net, &tensors, &new_path, &sr.legs).unwrap();
    assert!(expect.max_abs_diff(&got) < 1e-9);
    // The same seed must reproduce the same result.
    let (p2, sr2) = slice_temper(&net, &path, target, 7, 4, 4, 1_500, 6).unwrap();
    assert_eq!(new_path, p2);
    assert_eq!(sr.legs, sr2.legs);
}

#[test]
fn budgeted_respects_budget_and_matches_naive() {
    use arctn::paths::budgeted::budgeted_random_greedy;
    use arctn::slice::contract_network_sliced;
    let net = TensorNetwork::grid_2d(3, 4, 2);
    // Establish the unconstrained greedy peak.
    let (_, free_stats) = greedy(&net).unwrap();
    // Tighten the budget enough to require slicing.
    let budget = free_stats.log2_max_size - 2.0;
    let r = budgeted_random_greedy(&net, budget, 16, 9).expect("budgeted 失败");
    assert!(
        !r.sliced.is_empty(),
        "预算 {budget} 应当强制切片（自由峰值 {}）",
        free_stats.log2_max_size
    );
    assert!(
        r.per_slice.log2_max_size <= budget + 1e-9,
        "单片峰值 {} 超预算 {budget}",
        r.per_slice.log2_max_size
    );
    let tensors = random_tensors::<f64>(&net, 666);
    let expect = naive_einsum(&net, &tensors).unwrap();
    let got = contract_network_sliced(&net, &tensors, &r.path, &r.sliced).unwrap();
    assert!(expect.max_abs_diff(&got) < 1e-9);

    // Exercise additional random networks.
    for seed in 0..6u64 {
        let mut rng = ChaCha8Rng::seed_from_u64(seed + 500);
        let net = TensorNetwork::random_connected(6, 4, &[2, 3], 0, &mut rng);
        let (_, fs) = greedy(&net).unwrap();
        if fs.log2_max_size < 3.0 {
            continue;
        }
        let budget = fs.log2_max_size - 1.0;
        let Ok(r) = budgeted_random_greedy(&net, budget, 8, seed) else {
            continue;
        };
        assert!(r.per_slice.log2_max_size <= budget + 1e-9);
        let tensors = random_tensors::<f64>(&net, seed * 23 + 7);
        let expect = naive_einsum(&net, &tensors).unwrap();
        let got = contract_network_sliced(&net, &tensors, &r.path, &r.sliced).unwrap();
        assert!(expect.max_abs_diff(&got) < 1e-9, "seed={seed}");
    }
}

#[test]
fn simplify_preserves_result() {
    use arctn::simplify::{random_greedy_simplified, simplify};
    // Simplified paths on random small networks must match naive contraction.
    for seed in 0..12u64 {
        let mut rng = ChaCha8Rng::seed_from_u64(seed + 700);
        let n = 4 + (seed as usize % 3);
        let net = TensorNetwork::random_connected(n, 2, &[2, 3], (seed % 3) as usize, &mut rng);
        let (path, _) = random_greedy_simplified(&net, 8, seed).unwrap();
        check_path_matches_naive::<f64>(&net, &path, seed * 29 + 1, 1e-9);
        check_path_matches_naive::<Complex64>(&net, &path, seed * 29 + 2, 1e-9);
    }
    // Cover scalar, vector, and matrix absorption with output [d].
    let net = TensorNetwork {
        name: "ranks".into(),
        inputs: vec![vec![], vec![0], vec![0, 1], vec![1, 2, 3], vec![2]],
        output: vec![3],
        size_dict: [(0u32, 2usize), (1, 3), (2, 2), (3, 2)]
            .into_iter()
            .collect(),
    };
    let s = simplify(&net);
    assert!(
        s.reduced.n_tensors() <= 2,
        "rank0/1/2 应被吸收，剩 {} 个",
        s.reduced.n_tensors()
    );
    let (path, _) = random_greedy_simplified(&net, 4, 3).unwrap();
    check_path_matches_naive::<f64>(&net, &path, 91, 1e-9);
    // The repository fixture must produce a valid simplified path.
    let p =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/demo_tiny6.net.json");
    let net = TensorNetwork::load_json(&p).unwrap();
    let (_, stats) = random_greedy_simplified(&net, 16, 7).unwrap();
    assert!(stats.log10_flops.is_finite());
}

#[test]
fn anneal_preserves_result_and_never_worsens() {
    use arctn::anneal_paths;
    for seed in 0..10u64 {
        let mut rng = ChaCha8Rng::seed_from_u64(seed + 800);
        let n = 5 + (seed as usize % 3);
        let net = TensorNetwork::random_connected(n, 3, &[2, 3], (seed % 2) as usize, &mut rng);
        let (path, before) = greedy(&net).unwrap();
        let (ap, after) = anneal_paths(&net, &path, 4, 20_000, seed);
        assert!(planner_score(&after) <= planner_score(&before) + 1e-9);
        // Annealing must preserve the contraction result.
        check_path_matches_naive::<f64>(&net, &ap, seed * 31 + 1, 1e-9);
        check_path_matches_naive::<Complex64>(&net, &ap, seed * 31 + 2, 1e-9);
    }
}

#[test]
fn temper_simplified_valid_and_deterministic() {
    use arctn::simplify::temper_simplified;
    for seed in 0..4u64 {
        let mut rng = ChaCha8Rng::seed_from_u64(seed + 1500);
        let n = 6 + (seed as usize % 3);
        let net = TensorNetwork::random_connected(n, 3, &[2, 3], (seed % 2) as usize, &mut rng);
        let (p, s) = temper_simplified(&net, 4, seed, 4, 4, 1_500, 1e-3, 0.2, 300, 6).unwrap();
        // Tempering must not worsen its simplified random-greedy starting point.
        let (_, rs) = arctn::simplify::random_greedy_simplified(&net, 4, seed).unwrap();
        assert!(planner_score(&s) <= planner_score(&rs) + 1e-9);
        check_path_matches_naive::<f64>(&net, &p, seed * 53 + 1, 1e-9);
        check_path_matches_naive::<Complex64>(&net, &p, seed * 53 + 2, 1e-9);
        let (p2, _) = temper_simplified(&net, 4, seed, 4, 4, 1_500, 1e-3, 0.2, 300, 6).unwrap();
        assert_eq!(p, p2, "seed={seed} s-temper 不确定");
    }
}

#[test]
fn order_dp_monotone_with_dangling_legs() {
    // Regression: leaf costs must include single-owner non-output legs.
    use arctn::paths::ordertree::{leaf_order_of_path, order_dp};
    let mut size_dict = std::collections::HashMap::new();
    for (l, d) in [(0u32, 4usize), (1, 2), (2, 4), (3, 3), (4, 2), (5, 4)] {
        size_dict.insert(l, d);
    }
    // Tensors 0 and 1 carry dangling non-output legs 3 and 5, respectively.
    let net = TensorNetwork {
        name: "dangling".into(),
        inputs: vec![vec![0, 1, 3], vec![1, 2, 5], vec![2, 4], vec![4, 0]],
        output: vec![],
        size_dict,
    };
    let (gpath, gs) = greedy(&net).unwrap();
    let order = leaf_order_of_path(&net, &gpath).unwrap();
    let (dpath, ds) = order_dp(&net, &order).unwrap();
    assert!(planner_score(&ds) <= planner_score(&gs) + 1e-9);
    check_path_matches_naive::<f64>(&net, &dpath, 17, 1e-9);
    check_path_matches_naive::<Complex64>(&net, &dpath, 18, 1e-9);
}

#[test]
fn temper_paths_diverse_init_never_worse_than_best_init() {
    use arctn::paths::greedy::random_greedy;
    use arctn::tree::temper_paths;
    for seed in 0..4u64 {
        let mut rng = ChaCha8Rng::seed_from_u64(seed + 1300);
        let net = TensorNetwork::random_connected(6, 3, &[2, 3], 1, &mut rng);
        // Use greedy and two independently seeded random-greedy starts.
        let inits = vec![
            greedy(&net).unwrap().0,
            random_greedy(&net, 4, seed).unwrap().0,
            random_greedy(&net, 4, seed + 99).unwrap().0,
        ];
        let best_init = inits
            .iter()
            .map(|p| planner_score(&arctn::path::simulate_path(&net, p).unwrap()))
            .fold(f64::INFINITY, f64::min);
        let (tp, ts) = temper_paths(&net, &inits, 4, 4, 1_500, 1e-3, 0.2, 300, 6, seed, 0).unwrap();
        assert!(planner_score(&ts) <= best_init + 1e-9);
        check_path_matches_naive::<f64>(&net, &tp, seed * 47 + 1, 1e-9);
        // The same seed must reproduce the same path.
        let (tp2, _) = temper_paths(&net, &inits, 4, 4, 1_500, 1e-3, 0.2, 300, 6, seed, 0).unwrap();
        assert_eq!(tp, tp2);
    }
}

#[test]
fn order_dp_optimal_within_order_and_matches_naive() {
    use arctn::paths::optimal::optimal_dp;
    use arctn::paths::ordertree::{leaf_order_of_path, order_dp, rcm_order};
    for seed in 0..10u64 {
        let mut rng = ChaCha8Rng::seed_from_u64(seed + 1100);
        let n = 4 + (seed as usize % 4);
        let net = TensorNetwork::random_connected(n, 3, &[2, 3], (seed % 3) as usize, &mut rng);
        let (gpath, gs) = greedy(&net).unwrap();
        // The greedy tree is feasible within DP constrained to its leaf order.
        let order = leaf_order_of_path(&net, &gpath).unwrap();
        let (dpath, ds) = order_dp(&net, &order).unwrap();
        assert!(planner_score(&ds) <= planner_score(&gs) + 1e-9);
        check_path_matches_naive::<f64>(&net, &dpath, seed * 43 + 1, 1e-9);
        check_path_matches_naive::<Complex64>(&net, &dpath, seed * 43 + 2, 1e-9);
        // A fixed-order optimum cannot beat the global optimum beyond tolerance.
        if let Ok((_, os)) = optimal_dp(&net, 16) {
            assert!(
                planner_score(&ds) >= planner_score(&os) - 1e-9,
                "seed={seed} orderdp({:.4}) 竟低于全局 optimal({:.4})",
                planner_score(&ds),
                planner_score(&os)
            );
        }
        // Reverse Cuthill-McKee ordering must also produce a valid path.
        let (rpath, _) = order_dp(&net, &rcm_order(&net)).unwrap();
        check_path_matches_naive::<f64>(&net, &rpath, seed * 43 + 3, 1e-9);
    }
}

#[test]
fn temper_preserves_result_never_worsens_and_is_deterministic() {
    use arctn::tree::temper_path;
    for seed in 0..6u64 {
        let mut rng = ChaCha8Rng::seed_from_u64(seed + 900);
        let n = 5 + (seed as usize % 3);
        let net = TensorNetwork::random_connected(n, 3, &[2, 3], (seed % 2) as usize, &mut rng);
        let (path, before) = greedy(&net).unwrap();
        // These parameters exercise both ordinary and reconfiguration moves.
        let (tp, after) = temper_path(&net, &path, 4, 6, 2_000, 1e-3, 0.2, 300, 6, seed).unwrap();
        assert!(planner_score(&after) <= planner_score(&before) + 1e-9);
        // Tempering must preserve the contraction result.
        check_path_matches_naive::<f64>(&net, &tp, seed * 37 + 1, 1e-9);
        check_path_matches_naive::<Complex64>(&net, &tp, seed * 37 + 2, 1e-9);
        // Rayon scheduling must not affect same-seed reproducibility.
        let (tp2, after2) = temper_path(&net, &path, 4, 6, 2_000, 1e-3, 0.2, 300, 6, seed).unwrap();
        assert_eq!(tp, tp2, "seed={seed} 回火不确定");
        assert_eq!(after.log10_flops.to_bits(), after2.log10_flops.to_bits());
    }
}

#[test]
fn load_fixture_and_find_paths() {
    let p =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/demo_tiny6.net.json");
    let net = TensorNetwork::load_json(&p).unwrap();
    assert_eq!(net.n_tensors(), 6);
    let (_, gs) = greedy(&net).unwrap();
    let (_, rs) = random_greedy(&net, 16, 7).unwrap();
    let (_, os) = optimal_dp(&net, 26).unwrap();
    // The exact fixed-objective result must not be worse than either heuristic.
    assert!(planner_score(&os) <= planner_score(&gs) + 1e-9);
    assert!(planner_score(&os) <= planner_score(&rs) + 1e-9);
    assert!(os.log10_flops.is_finite());
}

// Simulation and execution must agree on network validity.

#[test]
fn simulate_and_contract_support_external_diagonal_and_trace() {
    // Repeated leg 0 in T0 is also held by T1, so it forms an external diagonal.
    // Simulation and execution must both accept it and match naive contraction.
    let diag = TensorNetwork {
        name: "ext_diag".into(),
        inputs: vec![vec![0, 0], vec![0]],
        output: vec![],
        size_dict: [(0u32, 3usize)].into_iter().collect(),
    };
    assert!(
        simulate_path(&diag, &vec![(0, 1)]).is_ok(),
        "外部对角应被 simulate 接受"
    );
    assert!(
        contract_network::<f64>(&diag, random_tensors(&diag, 1), &vec![(0, 1)]).is_ok(),
        "外部对角执行器应可执行"
    );
    check_path_matches_naive::<f64>(&diag, &vec![(0, 1)], 71, 1e-10);
    // A private repeated non-output leg remains a valid ordinary trace.
    let ok = TensorNetwork {
        name: "plain_trace".into(),
        inputs: vec![vec![0, 0, 1], vec![1]],
        output: vec![],
        size_dict: [(0u32, 3usize), (1, 4)].into_iter().collect(),
    };
    assert!(
        simulate_path(&ok, &vec![(0, 1)]).is_ok(),
        "普通 trace 应被 simulate 接受"
    );
    check_path_matches_naive::<f64>(&ok, &vec![(0, 1)], 71, 1e-10);
    // More than two occurrences remain unsupported.
    let bad = TensorNetwork {
        name: "triple".into(),
        inputs: vec![vec![0, 0, 0], vec![0]],
        output: vec![],
        size_dict: [(0u32, 3usize)].into_iter().collect(),
    };
    assert!(
        simulate_path(&bad, &vec![(0, 1)]).is_err(),
        "同腿出现 >2 次仍应被拒"
    );
}

#[test]
fn simulate_and_contract_reject_duplicate_output() {
    // Simulation and execution must reject duplicate output legs.
    let net = TensorNetwork {
        name: "dup_out".into(),
        inputs: vec![vec![0], vec![1, 0]],
        output: vec![0, 0],
        size_dict: [(0u32, 4usize), (1, 3)].into_iter().collect(),
    };
    assert!(
        simulate_path(&net, &vec![(0, 1)]).is_err(),
        "重复 output 应被 simulate 拒绝"
    );
    // Execution returns an error rather than panicking.
    let r = contract_network::<f64>(&net, random_tensors(&net, 2), &vec![(0, 1)]);
    assert!(r.is_err(), "重复 output 执行器应 Err（不得 panic）");
}

#[test]
fn simulate_rejects_empty_path_with_orphan_output() {
    // An empty path must reject output legs absent from the single input tensor.
    let net = TensorNetwork {
        name: "orphan_out".into(),
        inputs: vec![vec![0]],
        output: vec![0, 1],
        size_dict: [(0u32, 3usize), (1, 4)].into_iter().collect(),
    };
    assert!(
        simulate_path(&net, &vec![]).is_err(),
        "孤儿 output 腿应被空路径校验拒绝"
    );
    // A valid single tensor may reduce legs not present in the output.
    let ok = TensorNetwork {
        name: "ok_single".into(),
        inputs: vec![vec![0, 1]],
        output: vec![0],
        size_dict: [(0u32, 3usize), (1, 5)].into_iter().collect(),
    };
    assert!(simulate_path(&ok, &vec![]).is_ok());
}

#[test]
fn trace_input_peak_counts_full_dense_residency() {
    // Trace inputs count their full dense storage rather than deduplicated legs.
    // T0[a,a,b] has 3*3*4 = 36 elements, T1[b] has 4, and the scalar result has 1.
    let net = TensorNetwork {
        name: "trace_peak".into(),
        inputs: vec![vec![0, 0, 1], vec![1]],
        output: vec![],
        size_dict: [(0u32, 3usize), (1, 4)].into_iter().collect(),
    };
    let stats = simulate_path(&net, &vec![(0, 1)]).expect("合法 trace");
    assert!(
        (stats.log2_peak_size - 41f64.log2()).abs() < 1e-9,
        "trace 峰值应含完整 d² 驻留：得 {} 期望 {}",
        stats.log2_peak_size,
        41f64.log2()
    );
}

#[test]
#[should_panic(expected = "合法排列")]
fn permute_rejects_non_permutation_axes() {
    let t = DenseTensor::<f64>::from_data(vec![2, 3], (0..6).map(|x| x as f64).collect());
    let _ = t.permute(&[0, 0]); // Duplicate axes trigger the debug assertion.
}

#[test]
fn bisect_handles_star_graph() {
    // FM refinement must balance this rank-nine star and return an executable path.
    use arctn::paths::bisect::bisect;
    let k = 9usize;
    let hub: Vec<u32> = (0..k as u32).collect();
    let mut inputs = vec![hub];
    for i in 0..k as u32 {
        inputs.push(vec![i]);
    }
    let size_dict = (0..k as u32).map(|l| (l, 2usize)).collect();
    let net = TensorNetwork {
        name: "star".into(),
        inputs,
        output: vec![],
        size_dict,
    };
    let (path, _stats) = bisect(&net, 8, 7, 12).expect("bisect 星图应产出路径");
    // The path must match naive contraction.
    check_path_matches_naive::<f64>(&net, &path, 91, 1e-9);
}
