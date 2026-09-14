//! Planning objectives for contraction paths.
//!
//! Scores use log2 space to avoid overflow. Sliced scores include the slice
//! count because each slice repeats the FLOPs and read/write work.

use crate::path::{logaddexp2, PathStats};

/// Default FLOPs weight.
pub const FLOPS_WEIGHT: f64 = 1.0;
/// Default read/write-complexity weight.
pub const READ_WRITE_WEIGHT: f64 = 64.0;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ObjectiveKind {
    TotalFlops,
    TotalReadWrite,
    Weighted,
}

/// Objective used by one planning call.
///
/// The default is `FLOPs + 64 * read/write complexity`. Rust and Python callers
/// may provide other non-negative weights. At least one weight must be positive.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlannerObjective {
    flops_weight: f64,
    read_write_weight: f64,
}

impl Default for PlannerObjective {
    fn default() -> Self {
        Self::FIXED
    }
}

impl PlannerObjective {
    /// Default planning objective: `FLOPs + 64 * read/write complexity`.
    pub const FIXED: Self = Self {
        flops_weight: FLOPS_WEIGHT,
        read_write_weight: READ_WRITE_WEIGHT,
    };

    pub fn new(flops_weight: f64, read_write_weight: f64) -> Result<Self, String> {
        if !flops_weight.is_finite() || flops_weight < 0.0 {
            return Err("flops_weight must be finite and non-negative".into());
        }
        if !read_write_weight.is_finite() || read_write_weight < 0.0 {
            return Err("read_write_weight must be finite and non-negative".into());
        }
        if flops_weight == 0.0 && read_write_weight == 0.0 {
            return Err("flops_weight and read_write_weight cannot both be zero".into());
        }
        Ok(Self {
            flops_weight: flops_weight + 0.0,
            read_write_weight: read_write_weight + 0.0,
        })
    }

    pub(crate) fn kind(self) -> ObjectiveKind {
        match (self.flops_weight > 0.0, self.read_write_weight > 0.0) {
            (true, false) => ObjectiveKind::TotalFlops,
            (false, true) => ObjectiveKind::TotalReadWrite,
            (true, true) => ObjectiveKind::Weighted,
            (false, false) => unreachable!("validated planner objective has no active term"),
        }
    }

    /// Stable identifier used by command-line output and result metadata.
    pub const fn as_str(self) -> &'static str {
        match (self.flops_weight > 0.0, self.read_write_weight > 0.0) {
            (true, false) => "total_flops",
            (false, true) => "total_read_write",
            (true, true) => "flops_read_write",
            (false, false) => "invalid",
        }
    }

    pub const fn flops_weight(self) -> f64 {
        self.flops_weight
    }

    pub const fn read_write_weight(self) -> f64 {
        self.read_write_weight
    }

    #[inline]
    pub(crate) fn score_terms_log2(self, log2_flops: f64, log2_read_write: f64) -> f64 {
        match self.kind() {
            ObjectiveKind::TotalFlops => self.flops_weight.log2() + log2_flops,
            ObjectiveKind::TotalReadWrite => self.read_write_weight.log2() + log2_read_write,
            ObjectiveKind::Weighted => logaddexp2(
                self.flops_weight.log2() + log2_flops,
                self.read_write_weight.log2() + log2_read_write,
            ),
        }
    }

    /// Returns the log2 score of an unsliced path.
    pub fn score_path_log2(self, stats: &PathStats) -> f64 {
        let log2_flops = stats.log10_flops / std::f64::consts::LOG10_2;
        self.score_terms_log2(log2_flops, stats.log2_read_write)
    }

    /// Returns the log2 score of a sliced path.
    ///
    /// `stats` describes one slice and `log2_n_slices` is the log2 slice count.
    pub fn score_sliced_log2(self, stats: &PathStats, log2_n_slices: f64) -> f64 {
        self.score_path_log2(stats) + log2_n_slices
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_weights_and_label_are_stable() {
        assert_eq!(FLOPS_WEIGHT, 1.0);
        assert_eq!(READ_WRITE_WEIGHT, 64.0);
        assert_eq!(PlannerObjective::FIXED.as_str(), "flops_read_write");
        assert_eq!(PlannerObjective::FIXED.flops_weight(), 1.0);
        assert_eq!(PlannerObjective::FIXED.read_write_weight(), 64.0);
    }

    #[test]
    fn path_and_slice_scores_use_the_fixed_formula() {
        let stats = PathStats {
            log10_flops: 10.0f64.log10(),
            log2_max_size: 0.0,
            log2_max_contraction_size: 0.0,
            log2_total_size: 1.0,
            log2_read_write: 1.0,
            log2_peak_size: 0.0,
        };
        let path_score = PlannerObjective::FIXED.score_path_log2(&stats);
        assert!((path_score.exp2() - 138.0).abs() < 1e-12);
        assert!(
            (PlannerObjective::FIXED
                .score_sliced_log2(&stats, 2.0)
                .exp2()
                - 552.0)
                .abs()
                < 1e-12
        );
    }

    #[test]
    fn runtime_weights_are_validated_and_normalized() {
        for (flops, read_write) in [
            (-1.0, 1.0),
            (1.0, -1.0),
            (f64::NAN, 1.0),
            (1.0, f64::NAN),
            (f64::INFINITY, 1.0),
            (1.0, f64::INFINITY),
            (0.0, 0.0),
        ] {
            assert!(PlannerObjective::new(flops, read_write).is_err());
        }

        let pure_flops = PlannerObjective::new(1.0, -0.0).unwrap();
        assert_eq!(pure_flops.as_str(), "total_flops");
        assert_eq!(pure_flops.read_write_weight().to_bits(), 0.0f64.to_bits());

        let pure_read_write = PlannerObjective::new(-0.0, 3.0).unwrap();
        assert_eq!(pure_read_write.as_str(), "total_read_write");
        assert_eq!(pure_read_write.flops_weight().to_bits(), 0.0f64.to_bits());

        let weighted = PlannerObjective::new(2.5, 7.0).unwrap();
        assert_eq!(weighted.as_str(), "flops_read_write");
        assert_eq!(weighted.flops_weight(), 2.5);
        assert_eq!(weighted.read_write_weight(), 7.0);
    }

    #[test]
    fn runtime_scores_ignore_disabled_terms_and_preserve_weights() {
        let pure_flops = PlannerObjective::new(2.0, 0.0).unwrap();
        assert_eq!(pure_flops.score_terms_log2(5.0, f64::INFINITY), 6.0);

        let pure_read_write = PlannerObjective::new(0.0, 4.0).unwrap();
        assert_eq!(pure_read_write.score_terms_log2(f64::INFINITY, 3.0), 5.0);

        let weighted = PlannerObjective::new(2.0, 4.0).unwrap();
        assert_eq!(weighted.score_terms_log2(5.0, 3.0), 96.0f64.log2());
    }
}
