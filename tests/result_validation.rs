//! Public fallible pathfinding APIs must return `Err` for a malformed network rather than
//! reaching an infallible dimension lookup and panicking.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::time::{Duration, Instant};

use arctn::network::TensorNetwork;
use arctn::path::simulate_path;
use arctn::paths::bisect::bisect;
use arctn::paths::budgeted::{
    budgeted_portfolio, budgeted_portfolio_until, budgeted_random_greedy,
    budgeted_random_greedy_seeded, budgeted_random_greedy_seeded_until,
};
use arctn::paths::ordertree::{order_dp, order_dp_until, MAX_ORDER_DP_N};
use arctn::simplify::{bisect_simplified, temper_simplified, temper_simplified_until};
use arctn::tree::{reconfigure_path, reconfigure_path_until};
use arctn::SsaPath;

fn missing_dim_network() -> TensorNetwork {
    TensorNetwork {
        name: "missing-dim".into(),
        inputs: vec![vec![0, 1], vec![1, 2]],
        output: vec![0, 2],
        size_dict: [(0, 2), (2, 2)].into_iter().collect(),
    }
}

fn overflowing_shape_network() -> TensorNetwork {
    TensorNetwork {
        name: "overflowing-shape".into(),
        inputs: vec![vec![0, 1], vec![1, 2]],
        output: vec![0, 2],
        // Each individual dimension is representable by GEMM strides, but the first dense
        // input shape is not. This exercises the checked-product branch of `validate`.
        size_dict: [(0, isize::MAX as usize), (1, 3), (2, 2)]
            .into_iter()
            .collect(),
    }
}

fn err_without_panic<T>(label: &str, f: impl FnOnce() -> Result<T, String>) -> String {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Err(error)) => error,
        Ok(Ok(_)) => panic!("{label} unexpectedly accepted a malformed network"),
        Err(_) => panic!("{label} panicked instead of returning Err for a malformed network"),
    }
}

fn assert_all_fallible_entrances_reject(net: &TensorNetwork) {
    let empty_path = SsaPath::new();

    err_without_panic("bisect", || bisect(net, 1, 0, 2));
    err_without_panic("order_dp", || order_dp(net, &[]));
    err_without_panic("order_dp_until", || order_dp_until(net, &[], None));
    err_without_panic("reconfigure_path", || {
        reconfigure_path(net, &empty_path, 8, 1)
    });
    err_without_panic("reconfigure_path_until", || {
        reconfigure_path_until(net, &empty_path, 8, 1, None)
    });

    err_without_panic("temper_simplified", || {
        temper_simplified(net, 1, 0, 1, 1, 1, 1e-4, 0.15, 1, 1)
    });
    err_without_panic("temper_simplified_until", || {
        temper_simplified_until(
            net,
            1,
            0,
            1,
            1,
            1,
            1e-4,
            0.15,
            1,
            1,
            Some(Instant::now() + Duration::from_secs(1)),
        )
    });
    err_without_panic("bisect_simplified", || bisect_simplified(net, 1, 0, 2));

    err_without_panic("budgeted_random_greedy", || {
        budgeted_random_greedy(net, 8.0, 1, 0)
    });
    err_without_panic("budgeted_random_greedy_seeded", || {
        budgeted_random_greedy_seeded(net, 8.0, 1, 0, None)
    });
    err_without_panic("budgeted_random_greedy_seeded_until", || {
        budgeted_random_greedy_seeded_until(net, 8.0, 1, 0, None, None)
    });
    err_without_panic("budgeted_portfolio", || {
        budgeted_portfolio(net, 8.0, 1, 0, None)
    });
    err_without_panic("budgeted_portfolio_until", || {
        budgeted_portfolio_until(net, 8.0, 1, 0, None, None)
    });
}

#[test]
fn fallible_pathfinding_apis_reject_malformed_networks_without_panicking() {
    assert_all_fallible_entrances_reject(&missing_dim_network());
    assert_all_fallible_entrances_reject(&overflowing_shape_network());
}

#[test]
fn order_dp_rejects_its_public_size_limit_before_quadratic_allocation() {
    let n = MAX_ORDER_DP_N + 1;
    let net = TensorNetwork {
        name: "order-dp-too-large".into(),
        inputs: (0..n).map(|_| vec![0]).collect(),
        output: Vec::new(),
        size_dict: [(0, 2)].into_iter().collect(),
    };

    let error = err_without_panic("order_dp size guard", || order_dp(&net, &[]));
    assert!(error.contains("n≤"), "unexpected error: {error}");
    let error = err_without_panic("order_dp_until size guard", || {
        order_dp_until(&net, &[], None)
    });
    assert!(error.contains("n≤"), "unexpected error: {error}");
}

#[test]
fn timed_s_temper_returns_a_complete_incumbent_after_its_search_deadline() {
    // Rank-3 tensors prevent immediate simplification. Large work limits ensure
    // the cooperative deadline, rather than natural completion, stops the search.
    let n = 32usize;
    let net = TensorNetwork {
        name: "timed-s-temper-incumbent".into(),
        inputs: (0..n)
            .map(|i| vec![i as u32, ((i + 1) % n) as u32, (n + i) as u32])
            .collect(),
        output: Vec::new(),
        size_dict: (0..(2 * n)).map(|i| (i as u32, 2usize)).collect(),
    };
    let started = Instant::now();
    let deadline = started + Duration::from_millis(25);
    let (path, stats) = temper_simplified_until(
        &net,
        1,
        7,
        1,
        10_000,
        250_000,
        1e-4,
        0.15,
        0,
        8,
        Some(deadline),
    )
    .expect("a completed pre-deadline incumbent must not be discarded");

    assert!(
        Instant::now() >= deadline,
        "fixture did not exercise the deadline path"
    );
    assert_eq!(path.len(), n - 1);
    let authoritative = simulate_path(&net, &path).expect("returned SSA path must be valid");
    assert_eq!(stats.log10_flops, authoritative.log10_flops);
    assert_eq!(stats.log2_max_size, authoritative.log2_max_size);
    assert_eq!(
        stats.log2_max_contraction_size,
        authoritative.log2_max_contraction_size
    );
    assert_eq!(stats.log2_total_size, authoritative.log2_total_size);
    assert_eq!(stats.log2_peak_size, authoritative.log2_peak_size);
    assert!(
        started.elapsed() < Duration::from_secs(2),
        "cooperative deadline failed to stop the oversized search"
    );
}
