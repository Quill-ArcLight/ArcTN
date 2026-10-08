//! Checked costs for the actual contraction tree, without a global-leg bound.

use super::{CTreeCore, LegId, ObjectiveKind, PlannerObjective, TensorNetwork};
use crate::integer_cost::{pow2, score};

#[cfg(test)]
mod tests;

pub(super) fn improves(candidate: u128, incumbent: u128, minimum_gain: f64) -> bool {
    candidate < incumbent && (incumbent - candidate) as f64 / incumbent as f64 > minimum_gain
}

pub(super) fn relative_change(current: u128, next: u128) -> f64 {
    if next >= current {
        (next - current) as f64 / current as f64
    } else {
        -((current - next) as f64 / current as f64)
    }
}

pub(super) fn score_log2(score: u128, objective: PlannerObjective) -> f64 {
    let offset = match objective.kind() {
        ObjectiveKind::TotalFlops => objective.flops_weight().log2(),
        ObjectiveKind::TotalReadWrite => objective.read_write_weight().log2(),
        ObjectiveKind::Weighted => 0.0,
    };
    (score as f64).log2() + offset
}

pub(super) fn union_len(a: &[LegId], b: &[LegId]) -> usize {
    let (mut i, mut j, mut shared) = (0, 0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                i += 1;
                j += 1;
                shared += 1;
            }
        }
    }
    a.len() + b.len() - shared
}

#[derive(Clone)]
pub(super) struct IntegerTreeCosts {
    pub steps: Vec<u128>,
    pub tensor_sizes: Vec<u128>,
    pub flops: u128,
    pub read_write: u128,
    pub score: u128,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct IntegerRotation {
    pub step_b: u128,
    pub step_v: u128,
    pub tensor_b: u128,
    pub flops: u128,
    pub read_write: u128,
    pub score: u128,
}

impl IntegerTreeCosts {
    /// Replace only local nodes. Dead historical nodes never require a scan.
    pub fn reconfigure<const F: bool, const R: bool>(
        &mut self,
        tree: &CTreeCore<F, R>,
        old: &[usize],
        new: &[usize],
    ) -> Option<()> {
        let (mut flops, mut read_write) = (self.flops, self.read_write);
        for &node in old {
            if F {
                flops = flops.checked_sub(self.steps[node])?;
            }
            if R {
                let times = if node == tree.root { 1 } else { 2 };
                read_write = read_write.checked_sub(self.tensor_sizes[node].checked_mul(times)?)?;
            }
        }
        let mut replacements = Vec::with_capacity(new.len());
        for &node in new {
            let step = if F {
                pow2(union_len(
                    &tree.legs[tree.left[node]],
                    &tree.legs[tree.right[node]],
                ))?
            } else {
                0
            };
            let size = if R { pow2(tree.legs[node].len())? } else { 0 };
            if F {
                flops = flops.checked_add(step)?;
            }
            if R {
                read_write = read_write.checked_add(size.checked_mul(if node == tree.root {
                    1
                } else {
                    2
                })?)?;
            }
            replacements.push((node, step, size));
        }
        let next_score = score(flops, read_write, tree.objective)?;
        self.steps.resize(tree.legs.len(), 0);
        if R {
            self.tensor_sizes.resize(tree.legs.len(), 0);
        }
        for &node in old {
            self.steps[node] = 0;
            if R {
                self.tensor_sizes[node] = 0;
            }
        }
        for (node, step, size) in replacements {
            self.steps[node] = step;
            if R {
                self.tensor_sizes[node] = size;
            }
        }
        self.flops = flops;
        self.read_write = read_write;
        self.score = next_score;
        Some(())
    }

    pub fn build<const F: bool, const R: bool>(
        tree: &CTreeCore<F, R>,
        net: &TensorNetwork,
    ) -> Option<Self> {
        let mut costs = Self {
            steps: vec![0; tree.legs.len()],
            tensor_sizes: if R {
                vec![0; tree.legs.len()]
            } else {
                Vec::new()
            },
            flops: 0,
            read_write: 0,
            score: 0,
        };
        if tree.n_leaves == 1 {
            return Some(costs);
        }
        for node in 0..tree.legs.len() {
            if !tree.alive[node] {
                continue;
            }
            let leaf = tree.is_leaf(node);
            if F && !leaf {
                let step = pow2(union_len(
                    &tree.legs[tree.left[node]],
                    &tree.legs[tree.right[node]],
                ))?;
                costs.steps[node] = step;
                costs.flops = costs.flops.checked_add(step)?;
            }
            if R {
                // Original axes count raw input reads, including repeated axes.
                let size = pow2(if leaf {
                    net.inputs[node].len()
                } else {
                    tree.legs[node].len()
                })?;
                costs.tensor_sizes[node] = size;
                // Every non-root intermediate is written once and read once.
                let term = size.checked_mul(if !leaf && node != tree.root { 2 } else { 1 })?;
                costs.read_write = costs.read_write.checked_add(term)?;
            }
        }
        costs.score = score(costs.flops, costs.read_write, tree.objective)?;
        Some(costs)
    }

    pub fn rotation<const F: bool, const R: bool>(
        &self,
        tree: &CTreeCore<F, R>,
        b: usize,
        a: usize,
        promoted: usize,
        kept: usize,
        new_legs: &[LegId],
    ) -> Option<IntegerRotation> {
        let v = tree.parent[b];
        let (step_b, step_v, flops) = if F {
            let nb = pow2(union_len(&tree.legs[a], &tree.legs[kept]))?;
            let nv = pow2(union_len(&tree.legs[promoted], new_legs))?;
            // Remove first, then add; a representable next total must not fail
            // merely because old and replacement terms coexist temporarily.
            let next = self
                .flops
                .checked_sub(self.steps[b])?
                .checked_sub(self.steps[v])?
                .checked_add(nb)?
                .checked_add(nv)?;
            (nb, nv, next)
        } else {
            (0, 0, 0)
        };
        let (tensor_b, read_write) = if R {
            let tensor = pow2(new_legs.len())?;
            let next = self
                .read_write
                .checked_sub(self.tensor_sizes[b].checked_mul(2)?)?
                .checked_add(tensor.checked_mul(2)?)?;
            (tensor, next)
        } else {
            (0, 0)
        };
        Some(IntegerRotation {
            step_b,
            step_v,
            tensor_b,
            flops,
            read_write,
            score: score(flops, read_write, tree.objective)?,
        })
    }

    /// Exact local objective, with the frontier's true input-read multiplicity.
    pub fn replacement_scores<const F: bool, const R: bool>(
        &self,
        tree: &CTreeCore<F, R>,
        frontier: &[usize],
        internals: &[usize],
        path: &[(usize, usize)],
        step_legs: &[Vec<LegId>],
        objective: PlannerObjective,
    ) -> Option<(u128, u128)> {
        let (mut current_f, mut current_r, mut candidate_f, mut candidate_r) =
            (0u128, 0u128, 0u128, 0u128);
        for &node in internals {
            if F {
                current_f = current_f.checked_add(self.steps[node])?;
            }
            if R {
                for term in [tree.left[node], tree.right[node], node] {
                    current_r = current_r.checked_add(self.tensor_sizes[term])?;
                }
            }
        }
        if R {
            for &node in frontier {
                candidate_r = candidate_r.checked_add(self.tensor_sizes[node])?;
            }
        }
        for (step, &(a, b)) in path.iter().enumerate() {
            if F {
                let axes = |id: usize| {
                    if id < frontier.len() {
                        &tree.legs[frontier[id]]
                    } else {
                        &step_legs[id - frontier.len()]
                    }
                };
                candidate_f = candidate_f.checked_add(pow2(union_len(axes(a), axes(b)))?)?;
            }
            if R {
                let multiplicity = if step + 1 == path.len() { 1 } else { 2 };
                candidate_r = candidate_r
                    .checked_add(pow2(step_legs[step].len())?.checked_mul(multiplicity)?)?;
            }
        }
        Some((
            score(current_f, current_r, objective)?,
            score(candidate_f, candidate_r, objective)?,
        ))
    }
}
