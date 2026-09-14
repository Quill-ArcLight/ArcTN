//! Contracts for parallel sliced execution:
//! 1. Results are bit-identical across thread counts because chunking depends only
//!    on slice and output sizes.
//! 2. Parallel execution matches the reference einsum semantics.
//!
//! `to_bits()` detects changes in floating-point reduction order that tolerance
//! comparisons would hide.

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use arctn::naive_einsum;
use arctn::network::TensorNetwork;
use arctn::paths::greedy::greedy;
use arctn::slice::{contract_network_sliced, find_slices};
use arctn::tensor::{DenseTensor, Scalar};

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

/// Runs sliced contraction in a fixed-size Rayon pool and returns result bits.
fn run_bits(
    threads: usize,
    net: &TensorNetwork,
    tensors: &[DenseTensor<f64>],
    path: &arctn::path::SsaPath,
    sliced: &[u32],
) -> Vec<u64> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("建线程池失败");
    let out = pool.install(|| contract_network_sliced(net, tensors, path, sliced).unwrap());
    out.data().iter().map(|x| x.to_bits()).collect()
}

#[test]
fn sliced_exec_is_bit_identical_across_thread_counts() {
    // Use enough slices to exercise chunked parallel execution.
    let net = TensorNetwork::grid_2d(4, 5, 2);
    let (path, stats) = greedy(&net).unwrap();
    let sr = find_slices(&net, &path, stats.log2_max_size - 4.0).expect("切不出方案");
    let n_slices = (2f64).powf(sr.log2_n_slices).round() as usize;
    assert!(n_slices >= 8, "切片数 {n_slices} 太少，测不到并行分支");

    let tensors = random_tensors::<f64>(&net, 4242);
    let base = run_bits(1, &net, &tensors, &path, &sr.legs);
    for threads in [2usize, 3, 4, 8] {
        let got = run_bits(threads, &net, &tensors, &path, &sr.legs);
        assert_eq!(
            base, got,
            "线程数 {threads} 下结果位模式与单线程不同（n_slices={n_slices}）——\
             分块数一定是漏了线程无关性"
        );
    }
}

#[test]
fn sliced_exec_parallel_still_matches_naive() {
    for seed in 0..6u64 {
        let mut rng = ChaCha8Rng::seed_from_u64(seed + 900);
        let net = TensorNetwork::random_connected(6, 3, &[2, 3], 1, &mut rng);
        let (path, _) = greedy(&net).unwrap();
        let mut cand: Vec<u32> = net
            .size_dict
            .keys()
            .copied()
            .filter(|l| !net.output.contains(l))
            .collect();
        cand.sort_unstable();
        let sliced = &cand[..3.min(cand.len())];
        let tensors = random_tensors::<f64>(&net, seed * 31 + 7);
        let expect = naive_einsum(&net, &tensors).unwrap();

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .build()
            .unwrap();
        let got = pool.install(|| contract_network_sliced(&net, &tensors, &path, sliced).unwrap());
        let diff = expect.max_abs_diff(&got);
        assert!(diff < 1e-9, "seed={seed} 并行切片收缩不符 diff={diff:e}");
    }
}

/// Chunks must be non-empty, disjoint, complete, and cover a ragged tail.
///
/// Small or power-of-two slice counts do not exercise a ragged tail, so this test
/// covers both the maximum-chunk and memory-budget gates explicitly.
#[test]
fn chunking_is_exact_and_covers_ragged() {
    use arctn::slice::slice_parallel_chunks;

    let check = |n_slices: usize, out_numel: usize| -> usize {
        let n_chunks = slice_parallel_chunks(n_slices, out_numel);
        assert!(n_chunks >= 1);
        let chunk = n_slices.div_ceil(n_chunks);
        let mut covered = 0usize;
        for c in 0..n_chunks {
            let lo = c * chunk;
            let hi = ((c + 1) * chunk).min(n_slices);
            assert!(
                lo < hi,
                "空块 c={c} n_slices={n_slices} n_chunks={n_chunks}"
            );
            assert_eq!(lo, covered, "块不连续 c={c} n_slices={n_slices}");
            covered = hi;
        }
        assert_eq!(covered, n_slices, "覆盖不全 n_slices={n_slices}");
        n_chunks
    };

    // 257 is the smallest ragged case for the 256-chunk cap and unit output.
    let n = check(257, 1);
    assert_ne!(257 % 257usize.div_ceil(n), 0, "257 应当是 ragged 的");
    // The memory gate can also make three slices ragged into two chunks.
    check(3, 1 << 21);
    // Cover combinations of both gates.
    for &out_numel in &[1usize, 1 << 10, 1 << 20, 1 << 21, 1 << 22, 1 << 24] {
        for n_slices in 1..600 {
            check(n_slices, out_numel);
        }
        for n_slices in [1024, 1152, 3456, 4096, 65536] {
            check(n_slices, out_numel);
        }
    }
    // Extreme output sizes must never produce zero chunks.
    assert_eq!(check(1000, usize::MAX), 1);
}

/// Runs an end-to-end contraction with a verified ragged tail.
#[test]
fn sliced_exec_handles_ragged_last_chunk() {
    use arctn::slice::slice_parallel_chunks;
    let mut rng = ChaCha8Rng::seed_from_u64(31337);
    let net = TensorNetwork::random_connected(8, 3, &[5, 5], 1, &mut rng);
    let (path, _) = greedy(&net).unwrap();
    let mut cand: Vec<u32> = net
        .size_dict
        .keys()
        .copied()
        .filter(|l| !net.output.contains(l))
        .collect();
    cand.sort_unstable();
    let sliced = &cand[..4.min(cand.len())];
    let n_slices: usize = sliced.iter().map(|&l| net.dim(l)).product();
    let out_numel: usize = net
        .output
        .iter()
        .map(|&l| net.dim(l))
        .product::<usize>()
        .max(1);
    let n_chunks = slice_parallel_chunks(n_slices, out_numel);
    assert!(n_chunks > 1, "n_slices={n_slices} 没走到并行分支");
    assert_ne!(
        n_slices % n_slices.div_ceil(n_chunks),
        0,
        "n_slices={n_slices} n_chunks={n_chunks} 不 ragged，这个测试没测到目标分支"
    );

    let tensors = random_tensors::<f64>(&net, 555);
    let expect = naive_einsum(&net, &tensors).unwrap();
    let got = contract_network_sliced(&net, &tensors, &path, sliced).unwrap();
    assert!(expect.max_abs_diff(&got) < 1e-9, "尾块处理错误");
}
