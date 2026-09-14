//! Cross-process determinism tests for recursive bisection.
//!
//! Child processes have different HashMap seeds; explicit search seeds must still reproduce paths.

use std::collections::{HashMap, HashSet};
use std::process::Command;

use arctn::network::{LegId, TensorNetwork};
use arctn::paths::bisect::bisect;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Marks a child process that computes once and prints its fingerprint.
const CHILD: &str = "ARCTN_DETERMINISM_CHILD";
const REPEATS: usize = 8;

/// Builds a synthetic network with multi-holder hyperedges.
fn hyper_net(seed: u64, n_tensors: usize, n_legs: usize) -> TensorNetwork {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut inputs = Vec::with_capacity(n_tensors);
    for _ in 0..n_tensors {
        let k = rng.gen_range(2..=4usize);
        let mut legs: Vec<LegId> = (0..n_legs as LegId).collect();
        legs.shuffle(&mut rng);
        legs.truncate(k);
        legs.sort_unstable();
        inputs.push(legs);
    }
    let size_dict: HashMap<LegId, usize> = (0..n_legs as LegId).map(|l| (l, 2)).collect();
    TensorNetwork {
        name: format!("hyper_n{n_tensors}"),
        inputs,
        output: vec![],
        size_dict,
    }
}

/// Runs a test in multiple child processes and collects fingerprints.
fn collect_across_processes(test_name: &str) -> HashSet<String> {
    let exe = std::env::current_exe().expect("拿不到测试二进制路径");
    let mut seen = HashSet::new();
    for i in 0..REPEATS {
        let out = Command::new(&exe)
            .args(["--exact", test_name, "--nocapture"])
            .env(CHILD, "1")
            .output()
            .expect("拉起子进程失败");
        assert!(
            out.status.success(),
            "子进程 #{i} 失败：{}",
            String::from_utf8_lossy(&out.stderr)
        );
        let s = String::from_utf8_lossy(&out.stdout);
        let line = s
            .lines()
            .find(|l| l.starts_with("RESULT "))
            .unwrap_or_else(|| panic!("子进程 #{i} 没打印 RESULT：\n{s}"))
            .to_string();
        seen.insert(line);
    }
    seen
}

/// Checks cross-process determinism of bisection.
#[test]
fn bisect_is_deterministic_across_processes() {
    if std::env::var(CHILD).is_ok() {
        let mut bits = Vec::new();
        for k in 0..3u64 {
            let net = hyper_net(700 + k, 110, 28);
            let (path, st) = bisect(&net, 16, 3, 12).expect("bisect 失败");
            // Include both path and FLOPs so equal-cost path drift is visible.
            bits.push(format!(
                "{:016x}/{}",
                st.log10_flops.to_bits(),
                path.iter().map(|(a, b)| a * 1_000_003 + b).sum::<usize>()
            ));
        }
        println!("RESULT {}", bits.join(","));
        return;
    }
    let seen = collect_across_processes("bisect_is_deterministic_across_processes");
    assert_eq!(
        seen.len(),
        1,
        "bisect 在 {REPEATS} 个独立进程里给出了 {} 种不同结果 —— \
         HashMap 迭代序又漏进结果了（见本文件头）：\n{:#?}",
        seen.len(),
        seen
    );
}
