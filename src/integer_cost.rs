//! Checked cost arithmetic for networks whose index dimensions are all two.
//!
//! Eligibility is separate from range checking: a large network can have small
//! intermediates, but every generated size and accumulated cost must still fit.

use crate::network::TensorNetwork;
use crate::objective::{ObjectiveKind, PlannerObjective};

pub(crate) fn supports(net: &TensorNetwork, objective: PlannerObjective) -> bool {
    net.size_dict.values().all(|&dimension| dimension == 2)
        && (objective.kind() != ObjectiveKind::Weighted || objective == PlannerObjective::FIXED)
}

#[inline]
pub(crate) fn pow2(exponent: usize) -> Option<u128> {
    1u128.checked_shl(u32::try_from(exponent).ok()?)
}

/// A positive single-term weight does not affect ordering or relative changes.
/// Mixed objectives are supported only for the exact default weights (1, 64).
#[inline]
pub(crate) fn score(flops: u128, read_write: u128, objective: PlannerObjective) -> Option<u128> {
    match objective.kind() {
        ObjectiveKind::TotalFlops => Some(flops),
        ObjectiveKind::TotalReadWrite => Some(read_write),
        ObjectiveKind::Weighted if objective == PlannerObjective::FIXED => {
            // checked_shl checks the shift distance, not discarded high bits.
            flops.checked_add(read_write.checked_mul(64)?)
        }
        ObjectiveKind::Weighted => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shift_and_weighted_score_check_actual_integer_range() {
        assert_eq!(pow2(127), Some(1u128 << 127));
        assert_eq!(pow2(128), None);
        assert_eq!(pow2(usize::MAX), None);
        assert_eq!(score(8, 12, PlannerObjective::FIXED), Some(776));
        assert_eq!(score(0, 1u128 << 122, PlannerObjective::FIXED), None);
        assert_eq!(score(u128::MAX, 1, PlannerObjective::FIXED), None);
        assert_eq!(
            score(63, u128::MAX / 64, PlannerObjective::FIXED),
            Some(u128::MAX)
        );
    }

    #[test]
    fn unsupported_weights_are_not_silently_rounded() {
        let objective = PlannerObjective::new(1.0, 63.5).unwrap();
        assert_eq!(score(8, 12, objective), None);
        assert_eq!(
            score(8, 12, PlannerObjective::new(2.5, 0.0).unwrap()),
            Some(8)
        );
        assert_eq!(
            score(8, 12, PlannerObjective::new(0.0, 2.5).unwrap()),
            Some(12)
        );
    }
}
