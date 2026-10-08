//! Cost arithmetic shared by the connected and outer-product subset DPs.

use super::{union_log2_sum, LegMetadata};
use crate::network::{LegId, TensorNetwork};
use crate::objective::{ObjectiveKind, PlannerObjective};
use crate::path::logaddexp2;

pub(super) trait CostArithmetic {
    type Value: Copy + PartialOrd;
    const ZERO: Self::Value;
    const USE_LOGS: bool;

    fn size(legs: &[LegId], metadata: &LegMetadata) -> Self::Value;
    fn union_size(a: &[LegId], b: &[LegId], metadata: &LegMetadata) -> Self::Value;
    fn add(a: Self::Value, b: Self::Value) -> Self::Value;
    fn score(
        flops: Self::Value,
        read_write: Self::Value,
        objective: PlannerObjective,
    ) -> Self::Value;
    fn component_zero<const FLOPS: bool, const READ_WRITE: bool>() -> Self::Value;
    fn component_total<const FLOPS: bool, const READ_WRITE: bool>(
        left: Self::Value,
        right: Self::Value,
        flops: Self::Value,
        read_write: Self::Value,
        objective: PlannerObjective,
    ) -> Result<Self::Value, String>;
}

/// Preserve the original arithmetic, including linear connected pure-FLOPs DP.
pub(super) struct LogCosts;

impl CostArithmetic for LogCosts {
    type Value = f64;
    const ZERO: f64 = f64::NEG_INFINITY;
    const USE_LOGS: bool = true;

    #[inline]
    fn size(legs: &[LegId], metadata: &LegMetadata) -> f64 {
        legs.iter().map(|&leg| metadata.log2_dim(leg)).sum()
    }

    #[inline]
    fn union_size(a: &[LegId], b: &[LegId], metadata: &LegMetadata) -> f64 {
        union_log2_sum(a, b, metadata)
    }

    #[inline]
    fn add(a: f64, b: f64) -> f64 {
        logaddexp2(a, b)
    }

    #[inline]
    fn score(flops: f64, read_write: f64, objective: PlannerObjective) -> f64 {
        objective.score_terms_log2(flops, read_write)
    }

    #[inline]
    fn component_zero<const FLOPS: bool, const READ_WRITE: bool>() -> f64 {
        if FLOPS && !READ_WRITE {
            0.0
        } else {
            Self::ZERO
        }
    }

    #[inline]
    fn component_total<const FLOPS: bool, const READ_WRITE: bool>(
        left: f64,
        right: f64,
        flops: f64,
        read_write: f64,
        objective: PlannerObjective,
    ) -> Result<f64, String> {
        if FLOPS && !READ_WRITE {
            let step = flops.exp2();
            let total = left + right + step;
            if !step.is_finite() || !total.is_finite() {
                return Err("pure-FLOPs linear f64 overflow in local optimal DP".into());
            }
            Ok(total)
        } else {
            Ok(Self::add(
                Self::add(left, right),
                Self::score(flops, read_write, objective),
            ))
        }
    }
}

/// Exact integer costs for binary dimensions and the supported objectives.
/// Only instantiate this arithmetic after `supports` proves all sums fit u128.
pub(super) struct BinaryCosts;

impl BinaryCosts {
    pub(super) fn supports(net: &TensorNetwork, objective: PlannerObjective) -> bool {
        if !net.size_dict.values().all(|&dim| dim == 2) {
            return false;
        }
        // A positive factor can be omitted for a single-term objective.
        let coefficient = match objective.kind() {
            ObjectiveKind::TotalFlops => 1u128,
            ObjectiveKind::TotalReadWrite => 3,
            ObjectiveKind::Weighted if objective == PlannerObjective::FIXED => 1 + 3 * 64,
            ObjectiveKind::Weighted => return false,
        };
        // Every intermediate/union uses at most all distinct network legs.
        // Raw input sizes must also count repeated axes, before a trace is taken.
        let exponent = net
            .size_dict
            .len()
            .max(net.inputs.iter().map(Vec::len).max().unwrap_or(0));
        let Ok(exponent) = u32::try_from(exponent) else {
            return false;
        };
        // Bound the cost of any complete path, including all read/write terms.
        // max(1) also bounds sizes when the network has only one input.
        1u128
            .checked_shl(exponent)
            .and_then(|size| size.checked_mul(coefficient))
            .and_then(|step| step.checked_mul(net.n_tensors().saturating_sub(1).max(1) as u128))
            .is_some()
    }
}

impl CostArithmetic for BinaryCosts {
    type Value = u128;
    const ZERO: u128 = 0;
    const USE_LOGS: bool = false;

    #[inline]
    fn size(legs: &[LegId], _metadata: &LegMetadata) -> u128 {
        1u128 << legs.len()
    }

    #[inline]
    fn union_size(a: &[LegId], b: &[LegId], _metadata: &LegMetadata) -> u128 {
        let (mut i, mut j, mut shared) = (0, 0, 0);
        while i < a.len() && j < b.len() {
            match a[i].cmp(&b[j]) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    shared += 1;
                    i += 1;
                    j += 1;
                }
            }
        }
        1u128 << (a.len() + b.len() - shared)
    }

    #[inline]
    fn add(a: u128, b: u128) -> u128 {
        a + b
    }

    #[inline]
    fn score(flops: u128, read_write: u128, objective: PlannerObjective) -> u128 {
        match objective.kind() {
            ObjectiveKind::TotalFlops => flops,
            ObjectiveKind::TotalReadWrite => read_write,
            ObjectiveKind::Weighted => flops + (read_write << 6),
        }
    }

    #[inline]
    fn component_zero<const FLOPS: bool, const READ_WRITE: bool>() -> u128 {
        0
    }

    #[inline]
    fn component_total<const FLOPS: bool, const READ_WRITE: bool>(
        left: u128,
        right: u128,
        flops: u128,
        read_write: u128,
        objective: PlannerObjective,
    ) -> Result<u128, String> {
        Ok(left + right + Self::score(flops, read_write, objective))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn binary_net(distinct_legs: usize, n_inputs: usize) -> TensorNetwork {
        let mut inputs = vec![Vec::new(); n_inputs];
        for leg in 0..distinct_legs {
            inputs[leg % n_inputs].push(leg as LegId);
        }
        TensorNetwork {
            name: "binary-cost-arithmetic".into(),
            inputs,
            output: vec![],
            size_dict: (0..distinct_legs).map(|leg| (leg as LegId, 2)).collect(),
        }
    }

    #[test]
    fn support_bound_accepts_the_last_safe_exponent_for_each_objective() {
        // Three inputs imply two contractions. Distribute axes so each input
        // shape is also representable by the network's usize element count.
        let cases = [
            (PlannerObjective::new(1.0, 0.0).unwrap(), 126),
            (PlannerObjective::new(0.0, 1.0).unwrap(), 125),
            (PlannerObjective::FIXED, 119),
        ];
        for (objective, last_safe_exponent) in cases {
            let safe = binary_net(last_safe_exponent, 3);
            let unsafe_bound = binary_net(last_safe_exponent + 1, 3);
            safe.validate().unwrap();
            unsafe_bound.validate().unwrap();
            assert!(BinaryCosts::supports(&safe, objective));
            assert!(!BinaryCosts::supports(&unsafe_bound, objective));
        }
        assert!(!BinaryCosts::supports(
            &binary_net(128, 3),
            PlannerObjective::new(1.0, 0.0).unwrap(),
        ));
    }

    #[test]
    fn support_bound_accounts_for_the_number_of_contractions() {
        let objective = PlannerObjective::new(1.0, 0.0).unwrap();
        // 3 * 2^126 fits u128; 4 * 2^126 does not.
        assert!(BinaryCosts::supports(&binary_net(126, 4), objective));
        assert!(!BinaryCosts::supports(&binary_net(126, 5), objective));
    }

    #[test]
    fn single_term_weights_are_omitted_but_custom_mixed_weights_fall_back() {
        let net = binary_net(6, 3);
        for weight in [f64::MIN_POSITIVE, 0.125, 2.5, f64::MAX] {
            let flops = PlannerObjective::new(weight, 0.0).unwrap();
            let read_write = PlannerObjective::new(0.0, weight).unwrap();
            assert!(BinaryCosts::supports(&net, flops));
            assert!(BinaryCosts::supports(&net, read_write));
            assert_eq!(BinaryCosts::score(128, 17, flops), 128);
            assert_eq!(BinaryCosts::score(128, 17, read_write), 17);
        }
        for (flops_weight, read_write_weight) in [(1.0, 63.0), (2.0, 128.0), (0.5, 64.0)] {
            let objective = PlannerObjective::new(flops_weight, read_write_weight).unwrap();
            assert!(!BinaryCosts::supports(&net, objective));
        }
    }

    #[test]
    fn nonbinary_dimensions_fall_back_for_every_objective() {
        for dimension in [1, 3, 4] {
            let mut net = binary_net(3, 3);
            net.size_dict.insert(0, dimension);
            for objective in [
                PlannerObjective::new(1.0, 0.0).unwrap(),
                PlannerObjective::new(0.0, 1.0).unwrap(),
                PlannerObjective::FIXED,
            ] {
                assert!(!BinaryCosts::supports(&net, objective));
            }
        }
    }

    #[test]
    fn sizes_preserve_repeated_input_axes_and_count_scalar_writes() {
        let mut net = binary_net(2, 2);
        net.inputs = vec![vec![0, 0, 1], vec![1]];
        net.validate().unwrap();
        assert!(BinaryCosts::supports(&net, PlannerObjective::FIXED));
        let metadata = LegMetadata::new(&net, false);
        assert!(metadata.log2_dims.is_empty());
        assert_eq!(BinaryCosts::size(&net.inputs[0], &metadata), 8);
        assert_eq!(BinaryCosts::size(&[], &metadata), 1);
        // FLOPs count each effective leg once; raw reads count both copies of 0.
        assert_eq!(BinaryCosts::union_size(&[0, 1], &[1], &metadata), 4);
        let read_write = BinaryCosts::add(BinaryCosts::add(8, 2), 1);
        assert_eq!(
            BinaryCosts::score(4, read_write, PlannerObjective::FIXED),
            708
        );
    }

    #[test]
    fn arithmetic_guard_also_bounds_raw_repeated_axis_counts() {
        // Test the arithmetic guard directly with shape-only inputs: validation
        // would reject these arrays earlier because their numel exceeds usize.
        let mut net = binary_net(1, 2);
        net.inputs = vec![vec![0; 120], vec![0]];
        assert!(BinaryCosts::supports(&net, PlannerObjective::FIXED));
        net.inputs[0].push(0);
        assert!(!BinaryCosts::supports(&net, PlannerObjective::FIXED));
    }

    #[test]
    fn shifts_and_addition_preserve_cost_bits_beyond_f64_precision() {
        let net = binary_net(90, 3);
        assert!(BinaryCosts::supports(&net, PlannerObjective::FIXED));
        let metadata = LegMetadata::new(&net, false);
        let left = (0..60).collect::<Vec<LegId>>();
        let right = (35..90).collect::<Vec<LegId>>();
        let flops = BinaryCosts::union_size(&left, &right, &metadata);
        assert_eq!(flops, 1u128 << 90);
        let read_write = BinaryCosts::add(
            BinaryCosts::add(
                BinaryCosts::size(&left, &metadata),
                BinaryCosts::size(&right, &metadata),
            ),
            BinaryCosts::size(&[], &metadata),
        );
        let score = BinaryCosts::score(flops, read_write, PlannerObjective::FIXED);
        assert_eq!(score, (1u128 << 90) + (1u128 << 66) + (1u128 << 61) + 64);
        assert_eq!(score as f64, (score - 64) as f64);

        let pure_flops = PlannerObjective::new(1.0, 0.0).unwrap();
        let total =
            BinaryCosts::component_total::<true, false>(1u128 << 90, 1u128 << 60, 1, 0, pure_flops)
                .unwrap();
        assert_eq!(total, (1u128 << 90) + (1u128 << 60) + 1);
        assert_eq!(total as f64, (total - 1) as f64);
    }
}
