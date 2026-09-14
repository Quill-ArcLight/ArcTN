//! `slice_temper_until` checks its deadline before starting a slicing or tempering arm.

use std::time::{Duration, Instant};

use arctn::slice::slice_temper_until;
use arctn::TensorNetwork;

#[test]
fn expired_deadline_stops_before_first_slice_temper_step() {
    let net = TensorNetwork {
        name: "deadline".into(),
        inputs: vec![vec![0], vec![0]],
        output: vec![],
        size_dict: [(0, 2)].into_iter().collect(),
    };
    // An empty path exposes a missed deadline check by failing if simulation starts.
    let expired = Instant::now() - Duration::from_millis(1);
    assert!(slice_temper_until(&net, &vec![], 0.0, 1, 1, 1, 1, 1, Some(expired)).is_none());
}
