//! Arithmetic for fixed-leaf-order DP; traversal and reconstruction are shared.

use crate::integer_cost;
use crate::network::{LegId, TensorNetwork};
use crate::objective::PlannerObjective;
use crate::path::logaddexp2;
use std::iter::Sum;
use std::ops::{Add, AddAssign, Sub, SubAssign};

pub(super) trait IntervalCosts {
    type Exponent: Copy
        + Send
        + Sync
        + Add<Output = Self::Exponent>
        + Sub<Output = Self::Exponent>
        + AddAssign
        + SubAssign
        + Sum;
    type Cost: Copy + Send + Sync;
    const ZERO_EXPONENT: Self::Exponent;

    fn dimension(net: &TensorNetwork, leg: LegId) -> Self::Exponent;
    fn leaf_cost<const FLOPS: bool, const READ_WRITE: bool>() -> Self::Cost;
    fn unreachable() -> Self::Cost;
    fn better(candidate: Self::Cost, incumbent: Self::Cost) -> bool;
    fn complete<const FLOPS: bool, const READ_WRITE: bool>(cost: Self::Cost) -> bool;
    #[allow(clippy::too_many_arguments)]
    fn candidate<const FLOPS: bool, const READ_WRITE: bool>(
        left: Self::Cost,
        right: Self::Cost,
        step: Self::Exponent,
        left_read: Self::Exponent,
        right_read: Self::Exponent,
        result: Self::Exponent,
        objective: PlannerObjective,
    ) -> Result<Self::Cost, String>;
}

pub(super) struct LogCosts;

impl IntervalCosts for LogCosts {
    type Exponent = f64;
    type Cost = f64;
    const ZERO_EXPONENT: f64 = 0.0;

    #[inline]
    fn dimension(net: &TensorNetwork, leg: LegId) -> f64 {
        net.log2_dim(leg)
    }

    fn leaf_cost<const FLOPS: bool, const READ_WRITE: bool>() -> f64 {
        if FLOPS && !READ_WRITE {
            0.0
        } else {
            f64::NEG_INFINITY
        }
    }

    fn unreachable() -> f64 {
        f64::INFINITY
    }

    #[inline]
    fn better(candidate: f64, incumbent: f64) -> bool {
        candidate < incumbent
    }

    fn complete<const FLOPS: bool, const READ_WRITE: bool>(cost: f64) -> bool {
        !FLOPS || READ_WRITE || cost.is_finite()
    }

    #[inline]
    fn candidate<const FLOPS: bool, const READ_WRITE: bool>(
        left: f64,
        right: f64,
        step: f64,
        left_read: f64,
        right_read: f64,
        result: f64,
        objective: PlannerObjective,
    ) -> Result<f64, String> {
        if FLOPS && !READ_WRITE {
            let step = step.exp2();
            let total = left + right + step;
            if step.is_nan() || total.is_nan() {
                return Err("pure-FLOPs non-finite cost in leaf-order DP".into());
            }
            // Infinity cannot improve the incumbent, exactly as in the old DP.
            Ok(total)
        } else {
            let flops = if FLOPS { step } else { f64::NEG_INFINITY };
            let read_write = if READ_WRITE {
                logaddexp2(logaddexp2(left_read, right_read), result)
            } else {
                f64::NEG_INFINITY
            };
            let step_score = objective.score_terms_log2(flops, read_write);
            Ok(logaddexp2(logaddexp2(left, right), step_score))
        }
    }
}

pub(super) struct BinaryCosts;

impl IntervalCosts for BinaryCosts {
    // Signed counts permit the existing prefix-sum inclusion/exclusion order.
    // Validation bounds each binary input to fewer than usize::BITS axes and
    // order DP bounds n to 5000, so even temporary prefix sums fit easily.
    type Exponent = i64;
    // None means every tree for this interval exceeds u128, not cost zero.
    // In particular u128::MAX remains a valid, comparable cost.
    type Cost = Option<u128>;
    const ZERO_EXPONENT: i64 = 0;

    #[inline]
    fn dimension(_net: &TensorNetwork, _leg: LegId) -> i64 {
        1
    }

    fn leaf_cost<const FLOPS: bool, const READ_WRITE: bool>() -> Option<u128> {
        Some(0)
    }

    fn unreachable() -> Option<u128> {
        None
    }

    #[inline]
    fn better(candidate: Option<u128>, incumbent: Option<u128>) -> bool {
        match (candidate, incumbent) {
            (Some(candidate), Some(incumbent)) => candidate < incumbent,
            (Some(_), None) => true,
            _ => false,
        }
    }

    fn complete<const FLOPS: bool, const READ_WRITE: bool>(cost: Option<u128>) -> bool {
        cost.is_some()
    }

    #[inline]
    fn candidate<const FLOPS: bool, const READ_WRITE: bool>(
        left: Option<u128>,
        right: Option<u128>,
        step: i64,
        left_read: i64,
        right_read: i64,
        result: i64,
        objective: PlannerObjective,
    ) -> Result<Option<u128>, String> {
        let power = |exponent: i64| integer_cost::pow2(usize::try_from(exponent).ok()?);
        let evaluate = || -> Option<u128> {
            let children = left?.checked_add(right?)?;
            let flops = if FLOPS { power(step)? } else { 0 };
            let read_write = if READ_WRITE {
                power(left_read)?
                    .checked_add(power(right_read)?)?
                    .checked_add(power(result)?)?
            } else {
                0
            };
            children.checked_add(integer_cost::score(flops, read_write, objective)?)
        };
        // All active objective terms are nonnegative. An overflowing candidate
        // cannot beat a representable candidate, including in a parent interval.
        Ok(evaluate())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_u128_range_is_distinct_from_an_unreachable_candidate() {
        let objective = PlannerObjective::new(1.0, 0.0).unwrap();
        let maximum = BinaryCosts::candidate::<true, false>(
            Some(u128::MAX - 1),
            Some(0),
            0,
            0,
            0,
            0,
            objective,
        )
        .unwrap();
        assert_eq!(maximum, Some(u128::MAX));
        assert!(BinaryCosts::better(maximum, None));
        let overflow =
            BinaryCosts::candidate::<true, false>(maximum, Some(0), 0, 0, 0, 0, objective).unwrap();
        assert_eq!(overflow, None);
        assert!(!BinaryCosts::better(overflow, maximum));
    }

    #[test]
    fn integer_candidates_keep_small_terms_beyond_f64_precision() {
        let objective = PlannerObjective::new(1.0, 0.0).unwrap();
        let exact = BinaryCosts::candidate::<true, false>(
            Some(1u128 << 90),
            Some(1),
            0,
            0,
            0,
            0,
            objective,
        )
        .unwrap()
        .unwrap();
        assert_eq!(exact, (1u128 << 90) + 2);
        assert_eq!(exact as f64, (1u128 << 90) as f64);
    }
}
