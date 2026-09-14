//! Contraction trees and subtree reconfiguration.
//!
//! Leaves represent input tensors and internal nodes represent pairwise
//! contractions. Reconfiguration optimizes a bounded subtree with bitmask
//! dynamic programming and commits it only when the objective improves.

use std::collections::HashMap;
use std::time::Instant;

use crate::network::{LegId, LegIndex, TensorNetwork};
use crate::objective::{ObjectiveKind, PlannerObjective};
use crate::path::{
    legs_union, logaddexp2, simulate_path_flops_log2_and_legs, simulate_path_legs, sorted_dedup,
    PathStats, SsaPath,
};
use crate::paths::optimal::optimal_dp_reconfiguration_core;

const NONE: usize = usize::MAX;

pub const PURE_FLOPS_NUMERIC_MODE: &str = "linear-f64-total-flops";
pub const PURE_FLOPS_OVERFLOW_POLICY: &str = "reject-cell";

#[inline]
fn checked_linear_flops(log2_value: f64, context: &str) -> Result<f64, String> {
    let value = log2_value.exp2();
    if value.is_finite() {
        Ok(value)
    } else {
        Err(format!(
            "pure-FLOPs linear f64 overflow in {context}: log2={log2_value}"
        ))
    }
}

/// Bit set of input-tensor leaves.
type Bits = Vec<u64>;

#[derive(Clone)]
struct LogSumTree {
    leaf_count: usize,
    nodes: Vec<f64>,
}

impl LogSumTree {
    fn empty() -> Self {
        Self {
            leaf_count: 1,
            nodes: vec![f64::NEG_INFINITY; 2],
        }
    }

    fn from_values(values: &[f64]) -> Self {
        let leaf_count = values.len().max(1).next_power_of_two();
        let mut nodes = vec![f64::NEG_INFINITY; 2 * leaf_count];
        nodes[leaf_count..leaf_count + values.len()].copy_from_slice(values);
        for node in (1..leaf_count).rev() {
            nodes[node] = logaddexp2(nodes[2 * node], nodes[2 * node + 1]);
        }
        Self { leaf_count, nodes }
    }

    fn total(&self) -> f64 {
        self.nodes[1]
    }

    fn update(&mut self, index: usize, value: f64) {
        debug_assert!(index < self.leaf_count);
        let mut node = self.leaf_count + index;
        self.nodes[node] = value;
        while node > 1 {
            node /= 2;
            self.nodes[node] = logaddexp2(self.nodes[2 * node], self.nodes[2 * node + 1]);
        }
    }

    fn total_replacing(&self, first: (usize, f64), second: Option<(usize, f64)>) -> f64 {
        debug_assert!(first.0 < self.leaf_count);
        debug_assert!(second
            .map(|replacement| replacement.0 < self.leaf_count)
            .unwrap_or(true));
        self.total_replacing_range(1, 0, self.leaf_count, first, second)
    }

    fn total_replacing_range(
        &self,
        node: usize,
        start: usize,
        end: usize,
        first: (usize, f64),
        second: Option<(usize, f64)>,
    ) -> f64 {
        let contains_first = (start..end).contains(&first.0);
        let contains_second =
            second.is_some_and(|replacement| (start..end).contains(&replacement.0));
        if !contains_first && !contains_second {
            return self.nodes[node];
        }
        if end - start == 1 {
            if start == first.0 {
                return first.1;
            }
            if let Some((index, value)) = second {
                if start == index {
                    return value;
                }
            }
            return self.nodes[node];
        }
        let middle = start + (end - start) / 2;
        logaddexp2(
            self.total_replacing_range(2 * node, start, middle, first, second),
            self.total_replacing_range(2 * node + 1, middle, end, first, second),
        )
    }
}

// Store arbitrary leaf sets in 64-bit words.
fn bits_new(n: usize) -> Bits {
    vec![0u64; n.div_ceil(64)]
}
fn bits_set(b: &mut Bits, i: usize) {
    b[i / 64] |= 1u64 << (i % 64);
}
fn bits_or(a: &Bits, b: &Bits) -> Bits {
    a.iter().zip(b).map(|(x, y)| x | y).collect()
}
/// Return whether `a` contains any leaf absent from `b`.
fn bits_any_outside(a: &Bits, b: &Bits) -> bool {
    a.iter().zip(b).any(|(x, y)| x & !y != 0)
}
fn bits_count(a: &Bits) -> usize {
    a.iter().map(|x| x.count_ones() as usize).sum()
}

#[derive(Clone)]
pub struct CTreeCore<const TRACK_FLOPS: bool, const TRACK_READ_WRITE: bool> {
    objective: PlannerObjective,
    pub n_leaves: usize,
    left: Vec<usize>,
    right: Vec<usize>,
    parent: Vec<usize>,
    /// Sorted, deduplicated result legs for every node.
    pub legs: Vec<Vec<LegId>>,
    leafset: Vec<Bits>,
    alive: Vec<bool>,
    pub root: usize,
    /// Map external leg labels to compact cache slots.
    leg_index: LegIndex,
    /// Leaf holders for each leg cache slot.
    holders: Vec<Bits>,
    /// Whether each leg cache slot is an output leg.
    is_output: Vec<bool>,
    /// Cached base-two dimension logarithm for each leg slot.
    log2_dims: Vec<f64>,
    /// Base-two FLOPs logarithm for each live internal node.
    ///
    /// Construction, rotations, and reconfiguration keep this cache current.
    step_log2_cache: Vec<f64>,
    step_log2_sum: LogSumTree,
    /// Linear FLOPs cache used when read/write tracking is disabled.
    step_linear_cache: Vec<f64>,
    step_linear_total: f64,
    read_write_log2_sum: LogSumTree,
}

/// Evidence from a production-style online stopping rule applied at complete
/// whole-tree sweep boundaries. The report keeps scalar evidence rather than
/// every intermediate path and never executes a sweep after the stopping
/// condition has fired.
#[derive(Clone, Debug)]
pub struct ObjectiveReconfigurationEarlyStopReport {
    pub sweeps_requested: usize,
    pub sweeps_completed: usize,
    pub minimum_sweeps: usize,
    pub patience: usize,
    pub relative_gain_threshold: f64,
    pub stop_reason: &'static str,
    pub final_low_gain_streak: usize,
    pub score_replay_wall_s: f64,
    pub objective_log2_by_sweep: Vec<f64>,
    pub relative_gain_by_sweep: Vec<f64>,
}

/// Reusable zero-allocation scratch space for rotation evaluation.
pub struct RotScratch {
    legs_b: Vec<LegId>,
    ls_b: Bits,
}

#[derive(Clone, Copy, Debug, Default)]
struct ReconfigureNodeOutcome {
    applied: bool,
}

#[derive(Clone, Copy, Debug)]
struct RotationCostChange {
    next_flops_log2: f64,
    next_read_write_log2: f64,
    new_step_b_log2: f64,
    new_step_v_log2: f64,
    new_step_b_linear: f64,
    new_step_v_linear: f64,
    next_flops_linear: f64,
    new_tensor_b_log2: f64,
}
impl RotScratch {
    pub fn new() -> Self {
        RotScratch {
            legs_b: Vec::new(),
            ls_b: Vec::new(),
        }
    }
}
impl Default for RotScratch {
    fn default() -> Self {
        Self::new()
    }
}

/// Store `a | b` in `out`, reusing its allocation.
fn bits_or_into(a: &Bits, b: &Bits, out: &mut Bits) {
    out.clear();
    out.extend(a.iter().zip(b).map(|(x, y)| x | y));
}

pub type CTree = CTreeCore<true, true>;

impl<const TRACK_FLOPS: bool, const TRACK_READ_WRITE: bool>
    CTreeCore<TRACK_FLOPS, TRACK_READ_WRITE>
{
    /// Build a contraction tree from a complete SSA path.
    pub fn from_path(net: &TensorNetwork, path: &SsaPath) -> Result<Self, String> {
        Self::from_path_with_objective(net, path, PlannerObjective::FIXED)
    }

    pub(crate) fn from_path_with_objective(
        net: &TensorNetwork,
        path: &SsaPath,
        objective: PlannerObjective,
    ) -> Result<Self, String> {
        let tracking_matches = matches!(
            (TRACK_FLOPS, TRACK_READ_WRITE, objective.kind()),
            (true, false, ObjectiveKind::TotalFlops)
                | (false, true, ObjectiveKind::TotalReadWrite)
                | (true, true, ObjectiveKind::Weighted)
        );
        if !tracking_matches {
            return Err("planner objective does not match contraction-tree specialization".into());
        }
        let n = net.n_tensors();
        let step_legs = if TRACK_READ_WRITE && TRACK_FLOPS {
            crate::path::simulate_path_full(net, path)?.1
        } else if TRACK_FLOPS {
            simulate_path_flops_log2_and_legs(net, path)?.1
        } else {
            simulate_path_legs(net, path)?
        };
        let n_nodes = n + path.len();
        // Dense labels index directly; sparse labels use compact slots.
        let leg_index = LegIndex::new(net.size_dict.keys().copied());
        let mut holders: Vec<Bits> = vec![Vec::new(); leg_index.len()];
        for (i, t) in net.inputs.iter().enumerate() {
            for l in sorted_dedup(t) {
                let slot = &mut holders[leg_index.slot(l)];
                if slot.is_empty() {
                    *slot = bits_new(n);
                }
                bits_set(slot, i);
            }
        }
        let mut is_output = vec![false; leg_index.len()];
        for &l in &net.output {
            is_output[leg_index.slot(l)] = true;
        }
        let mut log2_dims = vec![0f64; leg_index.len()];
        for (&l, &d) in &net.size_dict {
            log2_dims[leg_index.slot(l)] = (d as f64).log2();
        }
        let mut t = Self {
            objective,
            n_leaves: n,
            left: vec![NONE; n_nodes],
            right: vec![NONE; n_nodes],
            parent: vec![NONE; n_nodes],
            legs: Vec::with_capacity(n_nodes),
            leafset: Vec::with_capacity(n_nodes),
            alive: vec![true; n_nodes],
            root: n_nodes - 1,
            leg_index,
            holders,
            is_output,
            log2_dims,
            step_log2_cache: vec![0.0; n_nodes],
            step_log2_sum: LogSumTree::empty(),
            step_linear_cache: if TRACK_FLOPS && !TRACK_READ_WRITE {
                vec![0.0; n_nodes]
            } else {
                Vec::new()
            },
            step_linear_total: 0.0,
            read_write_log2_sum: LogSumTree::empty(),
        };
        for (i, inp) in net.inputs.iter().enumerate() {
            t.legs.push(sorted_dedup(inp));
            let mut b = bits_new(n);
            bits_set(&mut b, i);
            t.leafset.push(b);
        }
        for (k, &(a, b)) in path.iter().enumerate() {
            let id = n + k;
            t.left[id] = a;
            t.right[id] = b;
            t.parent[a] = id;
            t.parent[b] = id;
            t.legs.push(step_legs[k].clone());
            t.leafset.push(bits_or(&t.leafset[a], &t.leafset[b]));
        }
        if n == 1 {
            t.root = 0;
        }
        if TRACK_FLOPS {
            for v in n..n_nodes {
                t.step_log2_cache[v] = t.union_log2(&t.legs[t.left[v]], &t.legs[t.right[v]]);
            }
        }
        t.rebuild_cost_sums(net)?;
        Ok(t)
    }

    fn is_leaf(&self, v: usize) -> bool {
        self.left[v] == NONE
    }

    #[inline]
    fn leg_slot(&self, leg: LegId) -> usize {
        self.leg_index.slot(leg)
    }

    /// Sum log2 dimensions over the union of two sorted leg lists.
    fn union_log2(&self, x: &[LegId], y: &[LegId]) -> f64 {
        let (mut i, mut j, mut s) = (0usize, 0usize, 0f64);
        while i < x.len() && j < y.len() {
            match x[i].cmp(&y[j]) {
                std::cmp::Ordering::Less => {
                    s += self.log2_dims[self.leg_slot(x[i])];
                    i += 1;
                }
                std::cmp::Ordering::Greater => {
                    s += self.log2_dims[self.leg_slot(y[j])];
                    j += 1;
                }
                std::cmp::Ordering::Equal => {
                    s += self.log2_dims[self.leg_slot(x[i])];
                    i += 1;
                    j += 1;
                }
            }
        }
        for &l in &x[i..] {
            s += self.log2_dims[self.leg_slot(l)];
        }
        for &l in &y[j..] {
            s += self.log2_dims[self.leg_slot(l)];
        }
        s
    }

    /// Base-two logarithm of one contraction step's FLOPs.
    fn step_log2_cost(&self, _net: &TensorNetwork, v: usize) -> f64 {
        debug_assert!(!self.is_leaf(v));
        debug_assert!(
            self.step_log2_cache[v]
                == self.union_log2(&self.legs[self.left[v]], &self.legs[self.right[v]]),
            "step_log2_cache 失效于节点 {v}"
        );
        self.step_log2_cache[v]
    }

    /// Base-two logarithm of a node tensor's element count.
    ///
    /// Leaves use original input axes, including repeated legs. Internal nodes
    /// use result legs, matching the accounting in `PathStats::log2_read_write`.
    #[inline]
    fn tensor_log2_size(&self, net: &TensorNetwork, v: usize) -> f64 {
        if self.is_leaf(v) {
            net.inputs[v].iter().map(|&l| net.log2_dim(l)).sum()
        } else {
            self.union_log2(&self.legs[v], &[])
        }
    }

    fn read_write_term_log2(&self, net: &TensorNetwork, v: usize) -> f64 {
        if self.n_leaves == 1 {
            return f64::NEG_INFINITY;
        }
        let multiplicity_log2 = if !self.is_leaf(v) && v != self.root {
            1.0
        } else {
            0.0
        };
        self.tensor_log2_size(net, v) + multiplicity_log2
    }

    #[inline]
    fn local_priority_log2(&self, net: &TensorNetwork, v: usize) -> f64 {
        if TRACK_FLOPS {
            return self.step_log2_cost(net, v);
        }
        if TRACK_READ_WRITE {
            logaddexp2(
                logaddexp2(
                    self.tensor_log2_size(net, self.left[v]),
                    self.tensor_log2_size(net, self.right[v]),
                ),
                self.tensor_log2_size(net, v),
            )
        } else {
            unreachable!("planner objective must contain at least one term")
        }
    }

    #[inline]
    fn score_log2(&self, flops_log2: f64, read_write_log2: f64) -> f64 {
        self.objective.score_terms_log2(flops_log2, read_write_log2)
    }

    fn rebuild_cost_sums(&mut self, net: &TensorNetwork) -> Result<(), String> {
        if TRACK_FLOPS && !TRACK_READ_WRITE {
            self.step_linear_cache.resize(self.legs.len(), 0.0);
            let mut total = 0.0;
            for v in 0..self.legs.len() {
                let value = if self.alive[v] && !self.is_leaf(v) {
                    checked_linear_flops(self.step_log2_cache[v], "tree rebuild step")?
                } else {
                    0.0
                };
                self.step_linear_cache[v] = value;
                total += value;
                if !total.is_finite() {
                    return Err("pure-FLOPs linear f64 overflow in tree rebuild total".into());
                }
            }
            self.step_linear_total = total;
            self.step_log2_sum = LogSumTree::empty();
        } else {
            self.step_log2_sum = if TRACK_FLOPS {
                let values = (0..self.legs.len())
                    .map(|v| {
                        if self.alive[v] && !self.is_leaf(v) {
                            self.step_log2_cache[v]
                        } else {
                            f64::NEG_INFINITY
                        }
                    })
                    .collect::<Vec<_>>();
                LogSumTree::from_values(&values)
            } else {
                LogSumTree::empty()
            };
        }
        self.read_write_log2_sum = if TRACK_READ_WRITE {
            let values = (0..self.legs.len())
                .map(|v| {
                    if self.alive[v] {
                        self.read_write_term_log2(net, v)
                    } else {
                        f64::NEG_INFINITY
                    }
                })
                .collect::<Vec<_>>();
            LogSumTree::from_values(&values)
        } else {
            LogSumTree::empty()
        };
        Ok(())
    }

    fn refresh_linear_total(&mut self) -> Result<(), String> {
        if !TRACK_FLOPS || TRACK_READ_WRITE {
            return Ok(());
        }
        let mut total = 0.0;
        for v in 0..self.legs.len() {
            if self.alive[v] && !self.is_leaf(v) {
                let step = self.step_linear_cache[v];
                if !step.is_finite() {
                    return Err("pure-FLOPs linear f64 non-finite cached step".into());
                }
                total += step;
                if !total.is_finite() {
                    return Err("pure-FLOPs linear f64 overflow in refreshed total".into());
                }
            }
        }
        self.step_linear_total = total;
        Ok(())
    }

    /// Base-ten logarithm of total tree FLOPs.
    pub fn total_log10_flops(&self, net: &TensorNetwork) -> f64 {
        self.total_cost_log2(net) * std::f64::consts::LOG10_2
    }

    /// Left-to-right leaf order used by order-constrained tree optimization.
    pub fn leaf_inorder(&self) -> Vec<usize> {
        let mut out = Vec::with_capacity(self.n_leaves);
        let mut stack = vec![self.root];
        while let Some(v) = stack.pop() {
            if self.is_leaf(v) {
                out.push(v);
                continue;
            }
            stack.push(self.right[v]);
            stack.push(self.left[v]);
        }
        out
    }

    /// Export the tree as a postorder SSA path.
    pub fn to_path(&self) -> SsaPath {
        let mut path = SsaPath::new();
        let mut ssa: HashMap<usize, usize> = HashMap::new();
        // Use iterative postorder traversal to support deep trees.
        let mut stack = vec![(self.root, false)];
        while let Some((v, expanded)) = stack.pop() {
            if self.is_leaf(v) {
                ssa.insert(v, v);
                continue;
            }
            if !expanded {
                stack.push((v, true));
                stack.push((self.right[v], false));
                stack.push((self.left[v], false));
            } else {
                let a = ssa[&self.left[v]];
                let b = ssa[&self.right[v]];
                let (x, y) = if a < b { (a, b) } else { (b, a) };
                path.push((x, y));
                ssa.insert(v, self.n_leaves + path.len() - 1);
            }
        }
        path
    }

    /// Filter result legs for a merged leaf set.
    ///
    /// A leg survives if it is an output or if any holder lies outside the
    /// merged set. Otherwise this contraction eliminates it.
    fn kept_legs(&self, union_legs: &[LegId], leafset: &Bits) -> Vec<LegId> {
        union_legs
            .iter()
            .copied()
            .filter(|l| {
                let slot = self.leg_slot(*l);
                self.is_output[slot] || bits_any_outside(&self.holders[slot], leafset)
            })
            .collect()
    }

    /// Expand the subtree at `v` to at most `k` frontier nodes.
    ///
    /// Returns `(frontier_nodes, internal_nodes)`.
    fn expand_subtree(&self, v: usize, k: usize) -> (Vec<usize>, Vec<usize>) {
        let mut frontier = vec![self.left[v], self.right[v]];
        let mut internals = vec![v];
        while frontier.len() < k {
            // Expand the internal frontier node with the most leaves.
            let best = frontier
                .iter()
                .enumerate()
                .filter(|(_, &f)| !self.is_leaf(f))
                .max_by_key(|(_, &f)| bits_count(&self.leafset[f]))
                .map(|(i, _)| i);
            match best {
                None => break,
                Some(i) => {
                    let f = frontier.swap_remove(i);
                    internals.push(f);
                    frontier.push(self.left[f]);
                    frontier.push(self.right[f]);
                }
            }
        }
        (frontier, internals)
    }

    /// Reoptimize the local subtree at `v` with dynamic programming.
    fn reconfigure_node(
        &mut self,
        net: &TensorNetwork,
        v: usize,
        subtree_size: usize,
    ) -> Result<ReconfigureNodeOutcome, String> {
        let (frontier, internals) = self.expand_subtree(v, subtree_size);
        if frontier.len() < 3 {
            return Ok(ReconfigureNodeOutcome::default());
        }
        let cur_flops_linear = if TRACK_FLOPS && !TRACK_READ_WRITE {
            let mut total = 0.0;
            for &u in &internals {
                total += self.step_linear_cache[u];
                if !total.is_finite() {
                    return Err("pure-FLOPs linear f64 overflow in local incumbent total".into());
                }
            }
            total
        } else {
            0.0
        };
        let cur_flops_log2 = if TRACK_FLOPS && !TRACK_READ_WRITE {
            cur_flops_linear.log2()
        } else if TRACK_FLOPS {
            internals.iter().fold(f64::NEG_INFINITY, |acc, &u| {
                logaddexp2(acc, self.step_log2_cost(net, u))
            })
        } else {
            f64::NEG_INFINITY
        };
        let cur_read_write_log2 = if TRACK_READ_WRITE {
            let read_write = internals.iter().fold(f64::NEG_INFINITY, |acc, &u| {
                [self.left[u], self.right[u], u]
                    .into_iter()
                    .fold(acc, |sum, node| {
                        logaddexp2(sum, self.tensor_log2_size(net, node))
                    })
            });
            read_write
        } else {
            f64::NEG_INFINITY
        };

        // Treat frontier nodes as inputs and the root legs as output.
        let mut size_dict = HashMap::new();
        for &f in &frontier {
            for &l in &self.legs[f] {
                size_dict.insert(l, net.dim(l));
            }
        }
        let mini = TensorNetwork {
            name: String::new(),
            inputs: frontier.iter().map(|&f| self.legs[f].clone()).collect(),
            output: self.legs[v].clone(),
            size_dict,
        };
        let Ok(candidate) = optimal_dp_reconfiguration_core::<TRACK_FLOPS, TRACK_READ_WRITE>(
            &mini,
            subtree_size.max(16),
            self.objective,
        ) else {
            return Ok(ReconfigureNodeOutcome::default());
        };
        let mini_path = candidate.path;
        let mut mini_read_write_log2 = f64::NEG_INFINITY;
        if TRACK_READ_WRITE {
            mini_read_write_log2 = frontier.iter().fold(f64::NEG_INFINITY, |acc, &f| {
                logaddexp2(acc, self.tensor_log2_size(net, f))
            });
            for (step, legs) in candidate.step_legs.iter().enumerate() {
                let result_log2 = self.union_log2(legs, &[]);
                mini_read_write_log2 = logaddexp2(mini_read_write_log2, result_log2);
                if step + 1 != candidate.step_legs.len() {
                    mini_read_write_log2 = logaddexp2(mini_read_write_log2, result_log2);
                }
            }
        }
        let improved = if TRACK_FLOPS && !TRACK_READ_WRITE {
            let candidate_linear = checked_linear_flops(
                path_stats_flops_roundtrip_log2(candidate.log2_flops),
                "local reconfiguration candidate",
            )?;
            candidate_linear < cur_flops_linear * (1.0 - 1e-12)
        } else {
            self.score_log2(
                path_stats_flops_roundtrip_log2(candidate.log2_flops),
                mini_read_write_log2,
            ) < self.score_log2(
                path_stats_flops_roundtrip_log2(cur_flops_log2),
                cur_read_write_log2,
            ) - 1e-12
        };
        if !improved {
            return Ok(ReconfigureNodeOutcome::default());
        }

        // Map the optimized local SSA path back into the global tree.
        let k = frontier.len();
        let mut map: Vec<usize> = frontier.clone(); // Local SSA ID to global node ID.
        for &u in &internals {
            if u != v {
                self.alive[u] = false;
            }
        }
        for (si, &(a, b)) in mini_path.iter().enumerate() {
            let (ga, gb) = (map[a], map[b]);
            let gid = if si == mini_path.len() - 1 {
                v // Reuse the local root node.
            } else {
                let gid = self.legs.len();
                self.left.push(NONE);
                self.right.push(NONE);
                self.parent.push(NONE);
                let ls = bits_or(&self.leafset[ga], &self.leafset[gb]);
                let legs = self.kept_legs(&legs_union(&self.legs[ga], &self.legs[gb]), &ls);
                self.legs.push(legs);
                self.leafset.push(ls);
                self.alive.push(true);
                self.step_log2_cache.push(0.0);
                if TRACK_FLOPS && !TRACK_READ_WRITE {
                    self.step_linear_cache.push(0.0);
                }
                gid
            };
            self.left[gid] = ga;
            self.right[gid] = gb;
            self.parent[ga] = gid;
            self.parent[gb] = gid;
            debug_assert_eq!(map.len(), k + si);
            map.push(gid);
        }
        // Refresh step caches for every node in the replacement subtree.
        if TRACK_FLOPS {
            for &gid in &map[k..] {
                self.step_log2_cache[gid] =
                    self.union_log2(&self.legs[self.left[gid]], &self.legs[self.right[gid]]);
            }
        }
        if TRACK_FLOPS && !TRACK_READ_WRITE {
            let mut replacement = 0.0;
            for &gid in &map[k..] {
                let step = checked_linear_flops(
                    self.step_log2_cache[gid],
                    "local reconfiguration replacement step",
                )?;
                self.step_linear_cache[gid] = step;
                replacement += step;
                if !replacement.is_finite() {
                    return Err("pure-FLOPs linear f64 overflow in local replacement total".into());
                }
            }
            let next_total = self.step_linear_total - cur_flops_linear + replacement;
            if !next_total.is_finite() || next_total <= 0.0 {
                return Err("pure-FLOPs linear f64 overflow in reconfigured tree total".into());
            }
            self.step_linear_total = next_total;
        } else {
            self.rebuild_cost_sums(net)?;
        }
        Ok(ReconfigureNodeOutcome { applied: true })
    }

    /// Reconfigure the full tree in descending local-priority order.
    ///
    /// Stops at convergence or after `max_sweeps` and returns completed sweeps.
    pub fn reconfigure(
        &mut self,
        net: &TensorNetwork,
        subtree_size: usize,
        max_sweeps: usize,
    ) -> Result<usize, String> {
        for sweep in 0..max_sweeps {
            let mut nodes: Vec<usize> = (0..self.legs.len())
                .filter(|&u| self.alive[u] && !self.is_leaf(u))
                .collect();
            nodes.sort_by(|&a, &b| {
                self.local_priority_log2(net, b)
                    .total_cmp(&self.local_priority_log2(net, a))
            });
            let mut improved = false;
            for u in nodes {
                if u < self.alive.len() && self.alive[u] && !self.is_leaf(u) {
                    improved |= self.reconfigure_node(net, u, subtree_size)?.applied;
                }
            }
            if !improved {
                return Ok(sweep + 1);
            }
        }
        Ok(max_sweeps)
    }

    /// Run objective-aware whole-tree reconfiguration with a predeclared
    /// online stopping rule.  Stopping is checked only after a complete sweep:
    /// after `minimum_sweeps`, stop once `patience` consecutive relative score
    /// gains are below `relative_gain_threshold`.
    ///
    /// The score is replayed on the original network after every sweep.  This
    /// replay is part of the online algorithm and its time is reported rather
    /// than subtracted.  The caller still owns the entering incumbent and must
    /// retain it on equality or regression.
    #[allow(clippy::too_many_arguments)]
    pub fn reconfigure_early_stop(
        &mut self,
        net: &TensorNetwork,
        subtree_size: usize,
        max_sweeps: usize,
        minimum_sweeps: usize,
        patience: usize,
        relative_gain_threshold: f64,
        deadline: Option<Instant>,
    ) -> Result<ObjectiveReconfigurationEarlyStopReport, String> {
        if minimum_sweeps == 0 || patience == 0 || minimum_sweeps < patience {
            return Err("invalid online early-stop sweep policy".into());
        }
        if !relative_gain_threshold.is_finite() || relative_gain_threshold < 0.0 {
            return Err("invalid online early-stop relative gain threshold".into());
        }

        let initial_path = self.to_path();
        let mut previous_score = replay_objective_score(net, &initial_path, self.objective)?;
        let mut scores = Vec::new();
        let mut gains = Vec::new();
        let mut low_gain_streak = 0usize;
        let mut score_replay_wall_s = 0.0;
        let mut stop_reason = "maximum-sweeps";

        for sweep in 0..max_sweeps {
            if deadline_reached(deadline) {
                stop_reason = "deadline-before-sweep";
                break;
            }
            // A deadline may arrive between local subtrees.  Keep a checkpoint
            // only for deadline-constrained calls so a partial sweep is never
            // delivered as if it were a completed stopping boundary.
            let sweep_checkpoint = deadline.map(|_| self.clone());
            let mut nodes: Vec<usize> = (0..self.legs.len())
                .filter(|&u| self.alive[u] && !self.is_leaf(u))
                .collect();
            nodes.sort_by(|&a, &b| {
                self.local_priority_log2(net, b)
                    .total_cmp(&self.local_priority_log2(net, a))
            });
            let mut accepted = 0usize;
            for u in nodes {
                if deadline_reached(deadline) {
                    stop_reason = "deadline-during-sweep";
                    break;
                }
                if u < self.alive.len()
                    && self.alive[u]
                    && !self.is_leaf(u)
                    && self.reconfigure_node(net, u, subtree_size)?.applied
                {
                    accepted += 1;
                }
            }
            if stop_reason == "deadline-during-sweep" {
                *self = sweep_checkpoint.expect("deadline-constrained sweep has checkpoint");
                break;
            }

            let replay_started = Instant::now();
            let path = self.to_path();
            let score = replay_objective_score(net, &path, self.objective)?;
            score_replay_wall_s += replay_started.elapsed().as_secs_f64();
            // 1 - new/old, evaluated stably from log2 scores.
            let gain = -((score - previous_score) * std::f64::consts::LN_2).exp_m1();
            scores.push(score);
            gains.push(gain.max(0.0));
            previous_score = score;
            low_gain_streak = if gain < relative_gain_threshold {
                low_gain_streak + 1
            } else {
                0
            };

            if accepted == 0 {
                stop_reason = "converged-no-accepted-local-reconfiguration";
                break;
            }
            let completed = sweep + 1;
            if completed >= minimum_sweeps && low_gain_streak >= patience {
                stop_reason = "relative-objective-gain-below-threshold";
                break;
            }
        }

        Ok(ObjectiveReconfigurationEarlyStopReport {
            sweeps_requested: max_sweeps,
            sweeps_completed: scores.len(),
            minimum_sweeps,
            patience,
            relative_gain_threshold,
            stop_reason,
            final_low_gain_streak: low_gain_streak,
            score_replay_wall_s,
            objective_log2_by_sweep: scores,
            relative_gain_by_sweep: gains,
        })
    }

    /// Deadline-aware whole-tree reconfiguration.
    ///
    /// A `None` deadline uses the unrestricted path. Deadline checks occur
    /// between sweeps and local subtrees. A `None` result means the deadline
    /// fired and the current tree must not be submitted as a new candidate.
    pub fn reconfigure_until(
        &mut self,
        net: &TensorNetwork,
        subtree_size: usize,
        max_sweeps: usize,
        deadline: Option<Instant>,
    ) -> Result<Option<usize>, String> {
        if deadline.is_none() {
            return Ok(Some(self.reconfigure(net, subtree_size, max_sweeps)?));
        }
        for sweep in 0..max_sweeps {
            if deadline_reached(deadline) {
                return Ok(None);
            }
            let mut nodes: Vec<usize> = (0..self.legs.len())
                .filter(|&u| self.alive[u] && !self.is_leaf(u))
                .collect();
            nodes.sort_by(|&a, &b| {
                self.local_priority_log2(net, b)
                    .total_cmp(&self.local_priority_log2(net, a))
            });
            let mut improved = false;
            for u in nodes {
                if deadline_reached(deadline) {
                    return Ok(None);
                }
                if u < self.alive.len() && self.alive[u] && !self.is_leaf(u) {
                    improved |= self.reconfigure_node(net, u, subtree_size)?.applied;
                }
            }
            if !improved {
                return Ok(Some(sweep + 1));
            }
        }
        Ok(Some(max_sweeps))
    }
}

impl<const TRACK_FLOPS: bool, const TRACK_READ_WRITE: bool>
    CTreeCore<TRACK_FLOPS, TRACK_READ_WRITE>
{
    /// Base-two logarithm of total tree FLOPs.
    pub fn total_cost_log2(&self, _net: &TensorNetwork) -> f64 {
        if TRACK_FLOPS && !TRACK_READ_WRITE {
            self.step_linear_total.log2()
        } else if TRACK_FLOPS {
            self.step_log2_sum.total()
        } else {
            (0..self.legs.len())
                .filter(|&node| self.alive[node] && !self.is_leaf(node))
                .fold(f64::NEG_INFINITY, |total, node| {
                    logaddexp2(
                        total,
                        self.union_log2(&self.legs[self.left[node]], &self.legs[self.right[node]]),
                    )
                })
        }
    }

    #[inline]
    fn search_flops_log2(&self) -> f64 {
        if TRACK_FLOPS && !TRACK_READ_WRITE {
            self.step_linear_total.log2()
        } else {
            self.step_log2_sum.total()
        }
    }

    #[inline]
    fn search_read_write_log2(&self) -> f64 {
        self.read_write_log2_sum.total()
    }

    /// Base-two logarithm of full-path read/write complexity.
    pub fn total_read_write_log2(&self, net: &TensorNetwork) -> f64 {
        if TRACK_READ_WRITE {
            self.search_read_write_log2()
        } else {
            (0..self.legs.len())
                .filter(|&node| self.alive[node])
                .fold(f64::NEG_INFINITY, |total, node| {
                    logaddexp2(total, self.read_write_term_log2(net, node))
                })
        }
    }

    /// Evaluate a rotation without mutating the tree.
    ///
    /// For internal node `b`, parent `v`, sibling `a`, and children `(c, d)`:
    ///
    /// - `promote_left`: `v=(c, b')`, `b'=(a, d)`;
    /// - otherwise: `v=(d, b')`, `b'=(a, c)`.
    ///
    /// The parent leaf set is unchanged. The returned replacement FLOPs and
    /// read/write terms are base-two logarithms.
    fn rotate_delta(
        &self,
        b: usize,
        promote_left: bool,
        sc: &mut RotScratch,
    ) -> Result<Option<RotationCostChange>, String> {
        let v = self.parent[b];
        if v == NONE || self.is_leaf(b) {
            return Ok(None);
        }
        let a = if self.left[v] == b {
            self.right[v]
        } else {
            self.left[v]
        };
        let (c, d) = (self.left[b], self.right[b]);
        let (promoted, kept) = if promote_left { (c, d) } else { (d, c) };
        // Build the replacement child directly into reusable scratch storage.
        bits_or_into(&self.leafset[a], &self.leafset[kept], &mut sc.ls_b);
        sc.legs_b.clear();
        {
            let (x, y) = (&self.legs[a], &self.legs[kept]);
            let (mut i, mut j) = (0usize, 0usize);
            let push = |l: LegId, sc_legs: &mut Vec<LegId>, ls: &Bits| {
                let slot = self.leg_slot(l);
                if self.is_output[slot] || bits_any_outside(&self.holders[slot], ls) {
                    sc_legs.push(l);
                }
            };
            while i < x.len() && j < y.len() {
                match x[i].cmp(&y[j]) {
                    std::cmp::Ordering::Less => {
                        push(x[i], &mut sc.legs_b, &sc.ls_b);
                        i += 1;
                    }
                    std::cmp::Ordering::Greater => {
                        push(y[j], &mut sc.legs_b, &sc.ls_b);
                        j += 1;
                    }
                    std::cmp::Ordering::Equal => {
                        push(x[i], &mut sc.legs_b, &sc.ls_b);
                        i += 1;
                        j += 1;
                    }
                }
            }
            while i < x.len() {
                push(x[i], &mut sc.legs_b, &sc.ls_b);
                i += 1;
            }
            while j < y.len() {
                push(y[j], &mut sc.legs_b, &sc.ls_b);
                j += 1;
            }
        }
        let (new_step_b_log2, new_step_v_log2) = if TRACK_FLOPS {
            (
                self.union_log2(&self.legs[a], &self.legs[kept]),
                self.union_log2(&self.legs[promoted], &sc.legs_b),
            )
        } else {
            (f64::NEG_INFINITY, f64::NEG_INFINITY)
        };
        let new_tensor_b_log2 = if TRACK_READ_WRITE {
            self.union_log2(&sc.legs_b, &[])
        } else {
            f64::NEG_INFINITY
        };
        let (new_step_b_linear, new_step_v_linear, next_flops_linear) =
            if TRACK_FLOPS && !TRACK_READ_WRITE {
                let new_b = checked_linear_flops(new_step_b_log2, "rotation step b")?;
                let new_v = checked_linear_flops(new_step_v_log2, "rotation step parent")?;
                let delta = new_b + new_v - self.step_linear_cache[b] - self.step_linear_cache[v];
                let next = self.step_linear_total + delta;
                if !delta.is_finite() || !next.is_finite() || next <= 0.0 {
                    return Err("pure-FLOPs linear f64 overflow in rotation total".into());
                }
                (new_b, new_v, next)
            } else {
                (0.0, 0.0, 0.0)
            };
        Ok(Some(RotationCostChange {
            next_flops_log2: if TRACK_FLOPS && !TRACK_READ_WRITE {
                f64::NEG_INFINITY
            } else if TRACK_FLOPS {
                self.step_log2_sum
                    .total_replacing((b, new_step_b_log2), Some((v, new_step_v_log2)))
            } else {
                f64::NEG_INFINITY
            },
            next_read_write_log2: if TRACK_READ_WRITE {
                self.read_write_log2_sum
                    .total_replacing((b, new_tensor_b_log2 + 1.0), None)
            } else {
                f64::NEG_INFINITY
            },
            new_step_b_log2,
            new_step_v_log2,
            new_step_b_linear,
            new_step_v_linear,
            next_flops_linear,
            new_tensor_b_log2,
        }))
    }

    /// Apply a rotation evaluated by [`Self::rotate_delta`].
    fn rotate_apply(&mut self, b: usize, promote_left: bool, change: RotationCostChange) {
        let v = self.parent[b];
        let a = if self.left[v] == b {
            self.right[v]
        } else {
            self.left[v]
        };
        let (c, d) = (self.left[b], self.right[b]);
        let (promoted, kept) = if promote_left { (c, d) } else { (d, c) };
        // Replace sibling `a` under `v` with the promoted child.
        if self.left[v] == a {
            self.left[v] = promoted;
        } else {
            self.right[v] = promoted;
        }
        self.parent[promoted] = v;
        // Rebuild `b` from `a` and the retained child.
        self.left[b] = a;
        self.right[b] = kept;
        self.parent[a] = b;
        self.parent[kept] = b;
        // Refresh `b`; the parent leaf set and result legs are unchanged.
        self.leafset[b] = bits_or(&self.leafset[a], &self.leafset[kept]);
        self.legs[b] = self.kept_legs(
            &legs_union(&self.legs[a], &self.legs[kept]),
            &self.leafset[b].clone(),
        );
        // Commit the values computed by the pure rotation evaluator.
        if TRACK_FLOPS && !TRACK_READ_WRITE {
            self.step_log2_cache[b] = change.new_step_b_log2;
            self.step_log2_cache[v] = change.new_step_v_log2;
            self.step_linear_cache[b] = change.new_step_b_linear;
            self.step_linear_cache[v] = change.new_step_v_linear;
            self.step_linear_total = change.next_flops_linear;
        } else if TRACK_FLOPS {
            self.step_log2_cache[b] = change.new_step_b_log2;
            self.step_log2_cache[v] = change.new_step_v_log2;
            self.step_log2_sum.update(b, change.new_step_b_log2);
            self.step_log2_sum.update(v, change.new_step_v_log2);
        }
        if TRACK_READ_WRITE {
            self.read_write_log2_sum
                .update(b, change.new_tensor_b_log2 + 1.0);
        }
    }
}

/// Anneal a contraction tree with random rotations and geometric cooling.
///
/// Uses relative-temperature Metropolis acceptance and retains the best path.
pub fn anneal_path(
    net: &TensorNetwork,
    path: &SsaPath,
    niters: usize,
    seed: u64,
    t0_rel: f64,
    t1_rel: f64,
) -> Result<(SsaPath, crate::path::PathStats), String> {
    anneal_path_with_objective(
        net,
        path,
        niters,
        seed,
        t0_rel,
        t1_rel,
        PlannerObjective::FIXED,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn anneal_path_with_objective(
    net: &TensorNetwork,
    path: &SsaPath,
    niters: usize,
    seed: u64,
    t0_rel: f64,
    t1_rel: f64,
    objective: PlannerObjective,
) -> Result<(SsaPath, crate::path::PathStats), String> {
    match objective.kind() {
        ObjectiveKind::TotalFlops => {
            anneal_path_core::<true, false>(net, path, niters, seed, t0_rel, t1_rel, objective)
        }
        ObjectiveKind::TotalReadWrite => {
            anneal_path_core::<false, true>(net, path, niters, seed, t0_rel, t1_rel, objective)
        }
        ObjectiveKind::Weighted => {
            anneal_path_core::<true, true>(net, path, niters, seed, t0_rel, t1_rel, objective)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn anneal_path_core<const TRACK_FLOPS: bool, const TRACK_READ_WRITE: bool>(
    net: &TensorNetwork,
    path: &SsaPath,
    niters: usize,
    seed: u64,
    t0_rel: f64,
    t1_rel: f64,
    objective: PlannerObjective,
) -> Result<(SsaPath, crate::path::PathStats), String> {
    use rand::Rng;
    use rand::SeedableRng;
    let mut tree =
        CTreeCore::<TRACK_FLOPS, TRACK_READ_WRITE>::from_path_with_objective(net, path, objective)?;
    let mut flops_log2 = tree.search_flops_log2();
    let mut flops_linear = if TRACK_FLOPS && !TRACK_READ_WRITE {
        tree.step_linear_total
    } else {
        0.0
    };
    let mut read_write_log2 = tree.search_read_write_log2();
    let init_stats = crate::path::simulate_path(net, path)?;
    // Rotatable nodes are non-root internal nodes.
    let rotatable: Vec<usize> = (0..tree.legs.len())
        .filter(|&v| tree.alive[v] && !tree.is_leaf(v) && tree.parent[v] != NONE)
        .collect();
    if rotatable.is_empty() {
        return Ok((path.clone(), init_stats));
    }
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    let mut sc = RotScratch::new();
    let mut best_total = if TRACK_FLOPS && !TRACK_READ_WRITE {
        flops_linear
    } else {
        objective.score_terms_log2(flops_log2, read_write_log2)
    };
    let mut best_path = path.clone();
    for it in 0..niters {
        let frac = it as f64 / niters.max(1) as f64;
        let t_rel = t0_rel * (t1_rel / t0_rel).powf(frac); // Geometric cooling.
        let b = rotatable[rng.gen_range(0..rotatable.len())];
        let promote_left = rng.gen_bool(0.5);
        let Some(change) = tree.rotate_delta(b, promote_left, &mut sc)? else {
            continue;
        };
        let next_flops_log2 = change.next_flops_log2;
        let next_read_write_log2 = change.next_read_write_log2;
        let (next_energy, accept) = if TRACK_FLOPS && !TRACK_READ_WRITE {
            let next = change.next_flops_linear;
            let relative_delta = (next - flops_linear) / flops_linear;
            let accept = next <= flops_linear
                || rng.gen::<f64>() < (-relative_delta / t_rel.max(f64::MIN_POSITIVE)).exp();
            (next, accept)
        } else {
            let current = objective.score_terms_log2(flops_log2, read_write_log2);
            let next = objective.score_terms_log2(next_flops_log2, next_read_write_log2);
            let accept = next <= current || {
                let relative_delta = relative_change(current, next);
                rng.gen::<f64>() < (-relative_delta / t_rel.max(f64::MIN_POSITIVE)).exp()
            };
            (next, accept)
        };
        if accept {
            tree.rotate_apply(b, promote_left, change);
            if TRACK_FLOPS && !TRACK_READ_WRITE {
                flops_linear = next_energy;
            } else {
                flops_log2 = next_flops_log2;
                read_write_log2 = next_read_write_log2;
            }
            let energy = next_energy;
            let improved = improves_ranking_by_relative::<TRACK_FLOPS, TRACK_READ_WRITE>(
                energy, best_total, 1e-12,
            );
            if improved {
                best_total = energy;
                best_path = tree.to_path();
            }
        }
    }
    let stats = crate::path::simulate_path(net, &best_path)?;
    // The original path remains the fallback incumbent.
    Ok(
        if objective.score_path_log2(&stats) <= objective.score_path_log2(&init_stats) {
            (best_path, stats)
        } else {
            (path.clone(), init_stats)
        },
    )
}

/// Run `chains` annealing trajectories in parallel and select by objective.
pub fn anneal_paths(
    net: &TensorNetwork,
    path: &SsaPath,
    chains: usize,
    niters: usize,
    seed: u64,
) -> (SsaPath, crate::path::PathStats) {
    anneal_paths_with_objective(net, path, chains, niters, seed, PlannerObjective::FIXED)
}

pub fn anneal_paths_with_objective(
    net: &TensorNetwork,
    path: &SsaPath,
    chains: usize,
    niters: usize,
    seed: u64,
    objective: PlannerObjective,
) -> (SsaPath, crate::path::PathStats) {
    use rayon::prelude::*;
    let init = crate::path::simulate_path(net, path).expect("anneal 输入路径非法");
    let best = (0..chains.max(1))
        .into_par_iter()
        .filter_map(|c| {
            let t0 = 0.05 * (1.5f64).powi(c as i32 % 4);
            anneal_path_with_objective(
                net,
                path,
                niters,
                seed.wrapping_add(c as u64 * 7919),
                t0,
                1e-4,
                objective,
            )
            .ok()
        })
        .min_by(|a, b| {
            objective
                .score_path_log2(&a.1)
                .total_cmp(&objective.score_path_log2(&b.1))
        });
    match best {
        Some((p, s)) if objective.score_path_log2(&s) < objective.score_path_log2(&init) => (p, s),
        _ => (path.clone(), init),
    }
}

#[inline]
fn path_stats_flops_roundtrip_log2(log2_flops: f64) -> f64 {
    (log2_flops * std::f64::consts::LOG10_2) / std::f64::consts::LOG10_2
}

#[cfg(test)]
#[inline]
fn fixed_score_log2(flops_log2: f64, read_write_log2: f64) -> f64 {
    PlannerObjective::FIXED.score_terms_log2(flops_log2, read_write_log2)
}

#[inline]
fn improves_by_relative(candidate_log2: f64, incumbent_log2: f64, minimum_gain: f64) -> bool {
    let threshold = (-minimum_gain.clamp(0.0, 1.0)).ln_1p() / std::f64::consts::LN_2;
    candidate_log2 < incumbent_log2 + threshold
}

#[inline]
fn improves_linear_by_relative(candidate: f64, incumbent: f64, minimum_gain: f64) -> bool {
    candidate < incumbent * (1.0 - minimum_gain.clamp(0.0, 1.0))
}

#[inline]
fn improves_ranking_by_relative<const TRACK_FLOPS: bool, const TRACK_READ_WRITE: bool>(
    candidate: f64,
    incumbent: f64,
    minimum_gain: f64,
) -> bool {
    if TRACK_FLOPS && !TRACK_READ_WRITE {
        improves_linear_by_relative(candidate, incumbent, minimum_gain)
    } else {
        improves_by_relative(candidate, incumbent, minimum_gain)
    }
}

#[inline]
fn relative_change(current_log2: f64, next_log2: f64) -> f64 {
    ((next_log2 - current_log2) * std::f64::consts::LN_2).exp_m1()
}

/// Select the best complete initial candidate with the requested planning objective.
fn fallback_init_index(init_evals: &[PathStats], objective: PlannerObjective) -> usize {
    (0..init_evals.len())
        .min_by(|&a, &b| {
            objective
                .score_path_log2(&init_evals[a])
                .total_cmp(&objective.score_path_log2(&init_evals[b]))
        })
        .expect("fallback candidates are non-empty")
}

/// State for one tempering replica.
struct Replica<const TRACK_FLOPS: bool, const TRACK_READ_WRITE: bool> {
    tree: CTreeCore<TRACK_FLOPS, TRACK_READ_WRITE>,
    flops_log2: f64,
    flops_linear: f64,
    score_log2: f64,
    rng: rand_chacha::ChaCha8Rng,
    /// Cached non-root internal nodes, rebuilt after subtree reconfiguration.
    rotatable: Vec<usize>,
    /// Best score: linear for pure FLOPs, otherwise base-two logarithmic.
    best_total: f64,
    best_path: SsaPath,
    /// Incrementally maintained base-two read/write complexity.
    read_write_log2: f64,
}

/// Work completed by a parallel-tempering call and the observed stop conditions.
///
/// A replica segment is one replica's move loop within one round. These counts
/// describe attempted search work, not numerical tensor-network execution.
#[derive(Clone, Copy, Debug, Default)]
pub struct TemperExecutionReport {
    /// Maximum number of rounds requested by the caller.
    pub rounds_requested: usize,
    /// Rounds whose replica work was started.
    pub rounds_started: usize,
    /// Rounds whose replica work finished before the deadline check.
    pub rounds_completed: usize,
    /// Replica move loops entered across all started rounds.
    pub replica_segments_started: usize,
    /// Replica move loops that finished, including those with no available rotation.
    pub replica_segments_completed: usize,
    /// Move-loop iterations entered, including reconfiguration and rejected proposals.
    pub moves_completed: usize,
    /// Scheduled attempts to reconfigure a local subtree.
    pub reconfiguration_attempts: usize,
    /// A cooperative deadline stopped initialization, a segment, or a round.
    pub stopped_by_deadline: bool,
    /// Consecutive rounds without sufficient improvement reached `patience`.
    pub stopped_by_patience: bool,
    /// A replica had no available rotation, or no initial path supported rotations.
    pub no_rotatable_replica: bool,
}

#[derive(Clone, Copy, Debug, Default)]
struct SegmentProgress {
    moves_completed: usize,
    reconfiguration_attempts: usize,
    completed: bool,
    stopped_by_deadline: bool,
    no_rotatable: bool,
}

impl<const TRACK_FLOPS: bool, const TRACK_READ_WRITE: bool> Replica<TRACK_FLOPS, TRACK_READ_WRITE> {
    #[inline]
    fn ranking_score(&self) -> f64 {
        if TRACK_FLOPS && !TRACK_READ_WRITE {
            self.flops_linear
        } else {
            self.score_log2
        }
    }

    #[inline]
    fn energy_log2(&self) -> f64 {
        if TRACK_FLOPS && !TRACK_READ_WRITE {
            self.flops_linear.log2()
        } else {
            self.score_log2
        }
    }
}

fn rotatable_nodes<const TRACK_FLOPS: bool, const TRACK_READ_WRITE: bool>(
    t: &CTreeCore<TRACK_FLOPS, TRACK_READ_WRITE>,
) -> Vec<usize> {
    (0..t.legs.len())
        .filter(|&v| t.alive[v] && !t.is_leaf(v) && t.parent[v] != NONE)
        .collect()
}

/// Coordinate parallel-tempering rounds and replica exchanges.
///
/// Moves run in parallel. Round boundaries refresh totals and attempt exchanges
/// between adjacent temperature slots.
#[allow(clippy::too_many_arguments)]
fn temper_core<const TRACK_FLOPS: bool, const TRACK_READ_WRITE: bool>(
    reps: &mut [Replica<TRACK_FLOPS, TRACK_READ_WRITE>],
    temps: &[f64],
    net: &TensorNetwork,
    rounds: usize,
    moves_per_round: usize,
    reconf_interval: usize,
    reconf_size: usize,
    xrng: &mut rand_chacha::ChaCha8Rng,
    patience: usize,
    min_round_improvement: f64,
    deadline: Option<Instant>,
) -> Result<TemperExecutionReport, String> {
    use rand::Rng;
    use rayon::prelude::*;
    let k = reps.len();
    let mut report = TemperExecutionReport {
        rounds_requested: rounds,
        ..TemperExecutionReport::default()
    };
    let mut stall = 0usize;
    let mut global_best = f64::INFINITY;
    for round in 0..rounds {
        if deadline_reached(deadline) {
            report.stopped_by_deadline = true;
            return Ok(report);
        }
        report.rounds_started += 1;
        let segments: Vec<_> = reps
            .par_iter_mut()
            .enumerate()
            .map(|(i, rep)| -> Result<SegmentProgress, String> {
                let progress = run_segment(
                    rep,
                    net,
                    temps[i],
                    moves_per_round,
                    reconf_interval,
                    reconf_size,
                    deadline,
                )?;
                if progress.completed && !deadline_reached(deadline) {
                    if TRACK_FLOPS && !TRACK_READ_WRITE {
                        rep.tree.refresh_linear_total()?;
                        rep.flops_linear = rep.tree.step_linear_total;
                    } else {
                        rep.flops_log2 = rep.tree.search_flops_log2();
                        rep.read_write_log2 = rep.tree.search_read_write_log2();
                        rep.score_log2 = rep.tree.score_log2(rep.flops_log2, rep.read_write_log2);
                    }
                }
                Ok(progress)
            })
            .collect::<Result<Vec<_>, String>>()?;
        report.replica_segments_started += segments.len();
        report.replica_segments_completed +=
            segments.iter().filter(|segment| segment.completed).count();
        report.moves_completed += segments
            .iter()
            .map(|segment| segment.moves_completed)
            .sum::<usize>();
        report.reconfiguration_attempts += segments
            .iter()
            .map(|segment| segment.reconfiguration_attempts)
            .sum::<usize>();
        report.no_rotatable_replica |= segments.iter().any(|segment| segment.no_rotatable);
        report.stopped_by_deadline |= segments.iter().any(|segment| segment.stopped_by_deadline);
        if deadline_reached(deadline) {
            report.stopped_by_deadline = true;
            return Ok(report);
        }
        report.rounds_completed += 1;
        let mut i = round % 2;
        while i + 1 < k {
            let ei = reps[i].energy_log2() * std::f64::consts::LN_2;
            let ej = reps[i + 1].energy_log2() * std::f64::consts::LN_2;
            let (bi, bj) = (1.0 / temps[i], 1.0 / temps[i + 1]);
            let p = ((bi - bj) * (ei - ej)).exp();
            if p >= 1.0 || xrng.gen::<f64>() < p {
                reps.swap(i, i + 1);
            }
            i += 2;
        }
        if patience > 0 {
            let cur = reps
                .iter()
                .map(|r| r.best_total)
                .fold(f64::INFINITY, f64::min);
            if improves_ranking_by_relative::<TRACK_FLOPS, TRACK_READ_WRITE>(
                cur,
                global_best,
                min_round_improvement,
            ) {
                global_best = cur;
                stall = 0;
            } else {
                stall += 1;
                if stall >= patience {
                    report.stopped_by_patience = true;
                    break;
                }
            }
        }
    }
    Ok(report)
}

/// Select an internal reconfiguration root weighted by local cost.
fn pick_cost_weighted<const TRACK_FLOPS: bool, const TRACK_READ_WRITE: bool>(
    tree: &CTreeCore<TRACK_FLOPS, TRACK_READ_WRITE>,
    net: &TensorNetwork,
    rng: &mut rand_chacha::ChaCha8Rng,
) -> Option<usize> {
    use rand::Rng;
    let nodes: Vec<usize> = (0..tree.legs.len())
        .filter(|&v| tree.alive[v] && !tree.is_leaf(v))
        .collect();
    if nodes.is_empty() {
        return None;
    }
    let weights: Vec<f64> = if TRACK_FLOPS && !TRACK_READ_WRITE {
        nodes.iter().map(|&v| tree.step_linear_cache[v]).collect()
    } else {
        let max_log2 = nodes
            .iter()
            .map(|&v| tree.local_priority_log2(net, v))
            .fold(f64::NEG_INFINITY, f64::max);
        nodes
            .iter()
            .map(|&v| (tree.local_priority_log2(net, v) - max_log2).exp2())
            .collect()
    };
    let sum: f64 = weights.iter().sum();
    if !(sum.is_finite() && sum > 0.0) {
        // Fall back to uniform sampling when weights degenerate.
        return Some(nodes[rng.gen_range(0..nodes.len())]);
    }
    let mut r = rng.gen::<f64>() * sum;
    for (v, w) in nodes.iter().zip(&weights) {
        r -= w;
        if r <= 0.0 {
            return Some(*v);
        }
    }
    Some(*nodes.last().unwrap())
}

/// Advance one replica at fixed relative temperature.
///
/// Local subtree reconfiguration runs at the configured interval.
fn run_segment<const TRACK_FLOPS: bool, const TRACK_READ_WRITE: bool>(
    rep: &mut Replica<TRACK_FLOPS, TRACK_READ_WRITE>,
    net: &TensorNetwork,
    t_rel: f64,
    moves: usize,
    reconf_interval: usize,
    reconf_size: usize,
    deadline: Option<Instant>,
) -> Result<SegmentProgress, String> {
    use rand::Rng;
    let mut sc = RotScratch::new();
    let mut progress = SegmentProgress::default();
    for m in 0..moves {
        if m % 256 == 0 && deadline_reached(deadline) {
            progress.stopped_by_deadline = true;
            return Ok(progress);
        }
        progress.moves_completed += 1;
        // Periodically apply non-regressing local dynamic-programming updates.
        if reconf_interval > 0 && (m + 1) % reconf_interval == 0 {
            if deadline_reached(deadline) {
                progress.stopped_by_deadline = true;
                return Ok(progress);
            }
            progress.reconfiguration_attempts += 1;
            if let Some(v) = pick_cost_weighted(&rep.tree, net, &mut rep.rng) {
                let outcome = rep.tree.reconfigure_node(net, v, reconf_size)?;
                if outcome.applied {
                    if TRACK_FLOPS && !TRACK_READ_WRITE {
                        rep.flops_linear = rep.tree.step_linear_total;
                    } else {
                        rep.flops_log2 = rep.tree.search_flops_log2();
                        rep.read_write_log2 = rep.tree.search_read_write_log2();
                        rep.score_log2 = rep.tree.score_log2(rep.flops_log2, rep.read_write_log2);
                    }
                    rep.rotatable = rotatable_nodes(&rep.tree);
                    let energy = rep.ranking_score();
                    let improved = improves_ranking_by_relative::<TRACK_FLOPS, TRACK_READ_WRITE>(
                        energy,
                        rep.best_total,
                        1e-12,
                    );
                    if improved {
                        rep.best_total = energy;
                        rep.best_path = rep.tree.to_path();
                    }
                }
            }
            continue;
        }
        if rep.rotatable.is_empty() {
            progress.no_rotatable = true;
            progress.completed = true;
            return Ok(progress);
        }
        let b = rep.rotatable[rep.rng.gen_range(0..rep.rotatable.len())];
        let promote_left = rep.rng.gen_bool(0.5);
        let Some(change) = rep.tree.rotate_delta(b, promote_left, &mut sc)? else {
            continue;
        };
        let next_flops_log2 = change.next_flops_log2;
        let next_read_write_log2 = change.next_read_write_log2;
        let (e_new, accept) = if TRACK_FLOPS && !TRACK_READ_WRITE {
            let next = change.next_flops_linear;
            let relative_delta = (next - rep.flops_linear) / rep.flops_linear;
            let accept = next <= rep.flops_linear
                || rep.rng.gen::<f64>() < (-relative_delta / t_rel.max(f64::MIN_POSITIVE)).exp();
            (next, accept)
        } else {
            let next = rep.tree.score_log2(next_flops_log2, next_read_write_log2);
            let current = rep.ranking_score();
            let accept = next <= current || {
                let relative_delta = relative_change(current, next);
                rep.rng.gen::<f64>() < (-relative_delta / t_rel.max(f64::MIN_POSITIVE)).exp()
            };
            (next, accept)
        };
        if accept {
            rep.tree.rotate_apply(b, promote_left, change);
            if TRACK_FLOPS && !TRACK_READ_WRITE {
                rep.flops_linear = e_new;
            } else {
                rep.flops_log2 = next_flops_log2;
                rep.read_write_log2 = next_read_write_log2;
                rep.score_log2 = e_new;
            }
            let energy = rep.ranking_score();
            let improved = improves_ranking_by_relative::<TRACK_FLOPS, TRACK_READ_WRITE>(
                energy,
                rep.best_total,
                1e-12,
            );
            if improved {
                rep.best_total = energy;
                rep.best_path = rep.tree.to_path();
            }
        }
    }
    progress.completed = true;
    Ok(progress)
}

/// Run parallel tempering on a fixed geometric temperature ladder.
///
/// Each replica performs `moves_per_round` moves with a private reproducible
/// RNG. Adjacent temperature slots exchange replicas with probability
/// `exp((beta_i - beta_j) * (E_i - E_j))`, where `E` is the natural logarithm
/// of the objective score and `beta = 1 / t_rel`. Each replica retains its best
/// complete path. A positive `reconf_interval` enables periodic local
/// reconfiguration.
#[allow(clippy::too_many_arguments)]
pub fn temper_path(
    net: &TensorNetwork,
    path: &SsaPath,
    n_replicas: usize,
    rounds: usize,
    moves_per_round: usize,
    t_min: f64,
    t_max: f64,
    reconf_interval: usize,
    reconf_size: usize,
    seed: u64,
) -> Result<(SsaPath, crate::path::PathStats), String> {
    temper_path_with_objective(
        net,
        path,
        n_replicas,
        rounds,
        moves_per_round,
        t_min,
        t_max,
        reconf_interval,
        reconf_size,
        seed,
        PlannerObjective::FIXED,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn temper_path_with_objective(
    net: &TensorNetwork,
    path: &SsaPath,
    n_replicas: usize,
    rounds: usize,
    moves_per_round: usize,
    t_min: f64,
    t_max: f64,
    reconf_interval: usize,
    reconf_size: usize,
    seed: u64,
    objective: PlannerObjective,
) -> Result<(SsaPath, crate::path::PathStats), String> {
    temper_paths_with_objective(
        net,
        std::slice::from_ref(path),
        n_replicas,
        rounds,
        moves_per_round,
        t_min,
        t_max,
        reconf_interval,
        reconf_size,
        seed,
        0,
        objective,
    )
}

/// Deadline-aware [`temper_path`].
///
/// Stops before starting another round or local reconfiguration and returns the
/// best complete path already produced.
#[allow(clippy::too_many_arguments)]
pub fn temper_path_until(
    net: &TensorNetwork,
    path: &SsaPath,
    n_replicas: usize,
    rounds: usize,
    moves_per_round: usize,
    t_min: f64,
    t_max: f64,
    reconf_interval: usize,
    reconf_size: usize,
    seed: u64,
    deadline: Option<Instant>,
) -> Result<(SsaPath, crate::path::PathStats), String> {
    temper_path_until_with_objective(
        net,
        path,
        n_replicas,
        rounds,
        moves_per_round,
        t_min,
        t_max,
        reconf_interval,
        reconf_size,
        seed,
        deadline,
        PlannerObjective::FIXED,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn temper_path_until_with_objective(
    net: &TensorNetwork,
    path: &SsaPath,
    n_replicas: usize,
    rounds: usize,
    moves_per_round: usize,
    t_min: f64,
    t_max: f64,
    reconf_interval: usize,
    reconf_size: usize,
    seed: u64,
    deadline: Option<Instant>,
    objective: PlannerObjective,
) -> Result<(SsaPath, crate::path::PathStats), String> {
    temper_paths_until_with_objective(
        net,
        std::slice::from_ref(path),
        n_replicas,
        rounds,
        moves_per_round,
        t_min,
        t_max,
        reconf_interval,
        reconf_size,
        seed,
        0,
        deadline,
        objective,
    )
}

/// Run tempering from multiple initial paths.
///
/// Replica `i` starts from `init_paths[i % len]`; the first path occupies the
/// coldest slot. The best initial path remains the fallback.
#[allow(clippy::too_many_arguments)]
pub fn temper_paths(
    net: &TensorNetwork,
    init_paths: &[SsaPath],
    n_replicas: usize,
    rounds: usize,
    moves_per_round: usize,
    t_min: f64,
    t_max: f64,
    reconf_interval: usize,
    reconf_size: usize,
    seed: u64,
    patience: usize,
) -> Result<(SsaPath, crate::path::PathStats), String> {
    temper_paths_with_objective(
        net,
        init_paths,
        n_replicas,
        rounds,
        moves_per_round,
        t_min,
        t_max,
        reconf_interval,
        reconf_size,
        seed,
        patience,
        PlannerObjective::FIXED,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn temper_paths_with_objective(
    net: &TensorNetwork,
    init_paths: &[SsaPath],
    n_replicas: usize,
    rounds: usize,
    moves_per_round: usize,
    t_min: f64,
    t_max: f64,
    reconf_interval: usize,
    reconf_size: usize,
    seed: u64,
    patience: usize,
    objective: PlannerObjective,
) -> Result<(SsaPath, crate::path::PathStats), String> {
    temper_paths_impl_with_objective(
        net,
        init_paths,
        n_replicas,
        rounds,
        moves_per_round,
        t_min,
        t_max,
        reconf_interval,
        reconf_size,
        seed,
        patience,
        1e-12,
        None,
        objective,
    )
}

/// Deadline-aware [`temper_paths`].
///
/// Stops before another round or reconfiguration step. Completed rotations
/// always produce complete trees, so the best path receives a final
/// authoritative score instead of discarding work completed before the deadline.
#[allow(clippy::too_many_arguments)]
pub fn temper_paths_until(
    net: &TensorNetwork,
    init_paths: &[SsaPath],
    n_replicas: usize,
    rounds: usize,
    moves_per_round: usize,
    t_min: f64,
    t_max: f64,
    reconf_interval: usize,
    reconf_size: usize,
    seed: u64,
    patience: usize,
    deadline: Option<Instant>,
) -> Result<(SsaPath, crate::path::PathStats), String> {
    temper_paths_until_with_objective(
        net,
        init_paths,
        n_replicas,
        rounds,
        moves_per_round,
        t_min,
        t_max,
        reconf_interval,
        reconf_size,
        seed,
        patience,
        deadline,
        PlannerObjective::FIXED,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn temper_paths_until_with_objective(
    net: &TensorNetwork,
    init_paths: &[SsaPath],
    n_replicas: usize,
    rounds: usize,
    moves_per_round: usize,
    t_min: f64,
    t_max: f64,
    reconf_interval: usize,
    reconf_size: usize,
    seed: u64,
    patience: usize,
    deadline: Option<Instant>,
    objective: PlannerObjective,
) -> Result<(SsaPath, crate::path::PathStats), String> {
    temper_paths_impl_with_objective(
        net,
        init_paths,
        n_replicas,
        rounds,
        moves_per_round,
        t_min,
        t_max,
        reconf_interval,
        reconf_size,
        seed,
        patience,
        1e-12,
        deadline,
        objective,
    )
}

/// Run parallel tempering with an improvement threshold and return work counts.
///
/// Uses the initial paths, replica temperatures, move limits, and cooperative
/// deadline of [`temper_paths_until_with_objective`]. When `patience` is positive,
/// stop after that many consecutive rounds without a sufficient relative
/// improvement in the best objective. `min_round_improvement` is clamped to
/// `[0, 1]`; callers should supply a finite value. Zero `patience` disables this
/// stopping condition.
///
/// Returns a complete path with recomputed metrics even when a deadline stops
/// further search. [`TemperExecutionReport`] records completed work and why the
/// search stopped; it does not contain preset configuration.
#[allow(clippy::too_many_arguments)]
pub fn temper_paths_until_with_threshold_traced_with_objective(
    net: &TensorNetwork,
    init_paths: &[SsaPath],
    n_replicas: usize,
    rounds: usize,
    moves_per_round: usize,
    t_min: f64,
    t_max: f64,
    reconf_interval: usize,
    reconf_size: usize,
    seed: u64,
    patience: usize,
    min_round_improvement: f64,
    deadline: Option<Instant>,
    objective: PlannerObjective,
) -> Result<((SsaPath, crate::path::PathStats), TemperExecutionReport), String> {
    temper_paths_impl_traced_with_objective(
        net,
        init_paths,
        n_replicas,
        rounds,
        moves_per_round,
        t_min,
        t_max,
        reconf_interval,
        reconf_size,
        seed,
        patience,
        min_round_improvement.clamp(0.0, 1.0),
        deadline,
        objective,
    )
}

#[allow(clippy::too_many_arguments)]
fn temper_paths_impl_with_objective(
    net: &TensorNetwork,
    init_paths: &[SsaPath],
    n_replicas: usize,
    rounds: usize,
    moves_per_round: usize,
    t_min: f64,
    t_max: f64,
    reconf_interval: usize,
    reconf_size: usize,
    seed: u64,
    patience: usize,
    min_round_improvement: f64,
    deadline: Option<Instant>,
    objective: PlannerObjective,
) -> Result<(SsaPath, crate::path::PathStats), String> {
    temper_paths_impl_traced_with_objective(
        net,
        init_paths,
        n_replicas,
        rounds,
        moves_per_round,
        t_min,
        t_max,
        reconf_interval,
        reconf_size,
        seed,
        patience,
        min_round_improvement,
        deadline,
        objective,
    )
    .map(|(candidate, _)| candidate)
}

#[allow(clippy::too_many_arguments)]
fn temper_paths_impl_traced_with_objective(
    net: &TensorNetwork,
    init_paths: &[SsaPath],
    n_replicas: usize,
    rounds: usize,
    moves_per_round: usize,
    t_min: f64,
    t_max: f64,
    reconf_interval: usize,
    reconf_size: usize,
    seed: u64,
    patience: usize,
    min_round_improvement: f64,
    deadline: Option<Instant>,
    objective: PlannerObjective,
) -> Result<((SsaPath, crate::path::PathStats), TemperExecutionReport), String> {
    match objective.kind() {
        ObjectiveKind::TotalFlops => temper_paths_impl_traced_core::<true, false>(
            net,
            init_paths,
            n_replicas,
            rounds,
            moves_per_round,
            t_min,
            t_max,
            reconf_interval,
            reconf_size,
            seed,
            patience,
            min_round_improvement,
            deadline,
            objective,
        ),
        ObjectiveKind::TotalReadWrite => temper_paths_impl_traced_core::<false, true>(
            net,
            init_paths,
            n_replicas,
            rounds,
            moves_per_round,
            t_min,
            t_max,
            reconf_interval,
            reconf_size,
            seed,
            patience,
            min_round_improvement,
            deadline,
            objective,
        ),
        ObjectiveKind::Weighted => temper_paths_impl_traced_core::<true, true>(
            net,
            init_paths,
            n_replicas,
            rounds,
            moves_per_round,
            t_min,
            t_max,
            reconf_interval,
            reconf_size,
            seed,
            patience,
            min_round_improvement,
            deadline,
            objective,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn temper_paths_impl_traced_core<const TRACK_FLOPS: bool, const TRACK_READ_WRITE: bool>(
    net: &TensorNetwork,
    init_paths: &[SsaPath],
    n_replicas: usize,
    rounds: usize,
    moves_per_round: usize,
    t_min: f64,
    t_max: f64,
    reconf_interval: usize,
    reconf_size: usize,
    seed: u64,
    patience: usize,
    min_round_improvement: f64,
    deadline: Option<Instant>,
    objective: PlannerObjective,
) -> Result<((SsaPath, crate::path::PathStats), TemperExecutionReport), String> {
    use rand::SeedableRng;
    if init_paths.is_empty() {
        return Err("temper_paths: init_paths 为空".into());
    }
    // Evaluate initial paths in order and retain the best complete candidate.
    let mut init_evals = Vec::with_capacity(init_paths.len());
    for p in init_paths {
        if !init_evals.is_empty() && deadline_reached(deadline) {
            break;
        }
        init_evals.push(crate::path::simulate_path(net, p)?);
    }
    let best_init_idx = fallback_init_index(&init_evals, objective);
    let fallback = (init_paths[best_init_idx].clone(), init_evals[best_init_idx]);

    // Build trees and exclude candidates without rotatable nodes.
    let mut seeds_pool: Vec<(
        CTreeCore<TRACK_FLOPS, TRACK_READ_WRITE>,
        f64,
        Vec<usize>,
        SsaPath,
    )> = Vec::new();
    for p in init_paths {
        if deadline_reached(deadline) {
            return Ok((
                fallback,
                TemperExecutionReport {
                    rounds_requested: rounds,
                    stopped_by_deadline: true,
                    ..TemperExecutionReport::default()
                },
            ));
        }
        let tree = CTreeCore::<TRACK_FLOPS, TRACK_READ_WRITE>::from_path_with_objective(
            net, p, objective,
        )?;
        let flops_log2 = tree.search_flops_log2();
        let read_write_log2 = tree.search_read_write_log2();
        let rot = rotatable_nodes(&tree);
        if objective
            .score_terms_log2(flops_log2, read_write_log2)
            .is_finite()
            && !rot.is_empty()
        {
            seeds_pool.push((tree, flops_log2, rot, p.clone()));
        }
    }
    if seeds_pool.is_empty() {
        return Ok((
            fallback,
            TemperExecutionReport {
                rounds_requested: rounds,
                no_rotatable_replica: true,
                ..TemperExecutionReport::default()
            },
        ));
    }
    let k = n_replicas.max(1);
    let temps: Vec<f64> = (0..k)
        .map(|i| {
            if k == 1 {
                t_min
            } else {
                t_min * (t_max / t_min).powf(i as f64 / (k - 1) as f64)
            }
        })
        .collect();
    let mut reps: Vec<Replica<TRACK_FLOPS, TRACK_READ_WRITE>> = (0..k)
        .map(|i| {
            let (tree, flops_log2, rot, p) = &seeds_pool[i % seeds_pool.len()];
            let read_write_log2 = tree.search_read_write_log2();
            let score_log2 = objective.score_terms_log2(*flops_log2, read_write_log2);
            let flops_linear = if TRACK_FLOPS && !TRACK_READ_WRITE {
                tree.step_linear_total
            } else {
                0.0
            };
            Replica {
                tree: tree.clone(),
                flops_log2: *flops_log2,
                flops_linear,
                score_log2,
                rng: rand_chacha::ChaCha8Rng::seed_from_u64(seed.wrapping_add(i as u64 * 7919)),
                rotatable: rot.clone(),
                best_total: if TRACK_FLOPS && !TRACK_READ_WRITE {
                    flops_linear
                } else {
                    score_log2
                },
                best_path: p.clone(),
                read_write_log2,
            }
        })
        .collect();
    let mut xrng = rand_chacha::ChaCha8Rng::seed_from_u64(seed ^ 0x9E37_79B9_7F4A_7C15);
    let temper_report = temper_core(
        &mut reps,
        &temps,
        net,
        rounds,
        moves_per_round,
        reconf_interval,
        reconf_size,
        &mut xrng,
        patience,
        min_round_improvement,
        deadline,
    )?;
    // `run_segment` snapshots a complete `best_path` whenever it improves.
    let best = reps
        .iter()
        .min_by(|a, b| a.best_total.total_cmp(&b.best_total))
        .unwrap();
    let stats = crate::path::simulate_path(net, &best.best_path)?;
    let result = if objective.score_path_log2(&stats) > objective.score_path_log2(&fallback.1) {
        fallback.clone()
    } else {
        (best.best_path.clone(), stats)
    };
    Ok((result, temper_report))
}

/// Run TreeSA-style annealing with linearly increasing inverse temperature.
///
/// Each beta value sweeps a random permutation of rotatable nodes. Energy is
/// the base-two objective logarithm and acceptance is `exp(-beta * delta_E)`.
/// Rayon runs chains in parallel, each retaining its best snapshot.
#[allow(clippy::too_many_arguments)]
pub fn treesa_path(
    net: &TensorNetwork,
    path: &SsaPath,
    chains: usize,
    beta0: f64,
    beta1: f64,
    beta_steps: usize,
    sweeps_per_beta: usize,
    reconf_interval: usize,
    reconf_size: usize,
    seed: u64,
) -> Result<(SsaPath, crate::path::PathStats), String> {
    treesa_path_with_objective(
        net,
        path,
        chains,
        beta0,
        beta1,
        beta_steps,
        sweeps_per_beta,
        reconf_interval,
        reconf_size,
        seed,
        PlannerObjective::FIXED,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn treesa_path_with_objective(
    net: &TensorNetwork,
    path: &SsaPath,
    chains: usize,
    beta0: f64,
    beta1: f64,
    beta_steps: usize,
    sweeps_per_beta: usize,
    reconf_interval: usize,
    reconf_size: usize,
    seed: u64,
    objective: PlannerObjective,
) -> Result<(SsaPath, crate::path::PathStats), String> {
    match objective.kind() {
        ObjectiveKind::TotalFlops => treesa_path_core::<true, false>(
            net,
            path,
            chains,
            beta0,
            beta1,
            beta_steps,
            sweeps_per_beta,
            reconf_interval,
            reconf_size,
            seed,
            objective,
        ),
        ObjectiveKind::TotalReadWrite => treesa_path_core::<false, true>(
            net,
            path,
            chains,
            beta0,
            beta1,
            beta_steps,
            sweeps_per_beta,
            reconf_interval,
            reconf_size,
            seed,
            objective,
        ),
        ObjectiveKind::Weighted => treesa_path_core::<true, true>(
            net,
            path,
            chains,
            beta0,
            beta1,
            beta_steps,
            sweeps_per_beta,
            reconf_interval,
            reconf_size,
            seed,
            objective,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn treesa_path_core<const TRACK_FLOPS: bool, const TRACK_READ_WRITE: bool>(
    net: &TensorNetwork,
    path: &SsaPath,
    chains: usize,
    beta0: f64,
    beta1: f64,
    beta_steps: usize,
    sweeps_per_beta: usize,
    reconf_interval: usize,
    reconf_size: usize,
    seed: u64,
    objective: PlannerObjective,
) -> Result<(SsaPath, crate::path::PathStats), String> {
    use rand::seq::SliceRandom;
    use rand::{Rng, SeedableRng};
    use rayon::prelude::*;
    let init_stats = crate::path::simulate_path(net, path)?;
    let tree0 =
        CTreeCore::<TRACK_FLOPS, TRACK_READ_WRITE>::from_path_with_objective(net, path, objective)?;
    let flops_log2_0 = tree0.search_flops_log2();
    let read_write_log2_0 = tree0.search_read_write_log2();
    if rotatable_nodes(&tree0).is_empty() {
        return Ok((path.clone(), init_stats));
    }
    let best = (0..chains.max(1))
        .into_par_iter()
        .map(|c| -> Result<(SsaPath, f64), String> {
            let mut rng =
                rand_chacha::ChaCha8Rng::seed_from_u64(seed.wrapping_add(c as u64 * 6151));
            let mut sc = RotScratch::new();
            let mut tree = tree0.clone();
            let mut flops_log2 = flops_log2_0;
            let mut flops_linear = if TRACK_FLOPS && !TRACK_READ_WRITE {
                tree.step_linear_total
            } else {
                0.0
            };
            let mut read_write_log2 = read_write_log2_0;
            let mut rotatable = rotatable_nodes(&tree);
            let mut best_total = if TRACK_FLOPS && !TRACK_READ_WRITE {
                flops_linear
            } else {
                objective.score_terms_log2(flops_log2, read_write_log2)
            };
            let mut best_path = path.clone();
            let mut tries = 0usize;
            for bi in 0..beta_steps.max(1) {
                let frac = bi as f64 / (beta_steps.max(2) - 1) as f64;
                let beta = beta0 + (beta1 - beta0) * frac;
                for _ in 0..sweeps_per_beta.max(1) {
                    let mut order = rotatable.clone();
                    order.shuffle(&mut rng);
                    for b in order {
                        // Reconfiguration may invalidate nodes queued for this sweep.
                        if b >= tree.alive.len()
                            || !tree.alive[b]
                            || tree.is_leaf(b)
                            || tree.parent[b] == NONE
                        {
                            continue;
                        }
                        tries += 1;
                        if reconf_interval > 0 && tries % reconf_interval == 0 {
                            if let Some(v) = pick_cost_weighted(&tree, net, &mut rng) {
                                if tree.reconfigure_node(net, v, reconf_size)?.applied {
                                    if TRACK_FLOPS && !TRACK_READ_WRITE {
                                        flops_linear = tree.step_linear_total;
                                    } else {
                                        flops_log2 = tree.search_flops_log2();
                                        read_write_log2 = tree.search_read_write_log2();
                                    }
                                    rotatable = rotatable_nodes(&tree);
                                    let score = if TRACK_FLOPS && !TRACK_READ_WRITE {
                                        flops_linear
                                    } else {
                                        objective.score_terms_log2(flops_log2, read_write_log2)
                                    };
                                    let improved = improves_ranking_by_relative::<
                                        TRACK_FLOPS,
                                        TRACK_READ_WRITE,
                                    >(
                                        score, best_total, 1e-12
                                    );
                                    if improved {
                                        best_total = score;
                                        best_path = tree.to_path();
                                    }
                                }
                            }
                            continue;
                        }
                        let promote_left = rng.gen_bool(0.5);
                        let Some(change) = tree.rotate_delta(b, promote_left, &mut sc)? else {
                            continue;
                        };
                        let next_flops_log2 = change.next_flops_log2;
                        let next_read_write_log2 = change.next_read_write_log2;
                        let (next_score, accept) = if TRACK_FLOPS && !TRACK_READ_WRITE {
                            let next = change.next_flops_linear;
                            let de = (next / flops_linear).log2();
                            (
                                next,
                                next <= flops_linear || rng.gen::<f64>() < (-beta * de).exp(),
                            )
                        } else {
                            let current = objective.score_terms_log2(flops_log2, read_write_log2);
                            let next =
                                objective.score_terms_log2(next_flops_log2, next_read_write_log2);
                            let de = next - current;
                            (
                                next,
                                next <= current || rng.gen::<f64>() < (-beta * de).exp(),
                            )
                        };
                        if accept {
                            tree.rotate_apply(b, promote_left, change);
                            if TRACK_FLOPS && !TRACK_READ_WRITE {
                                flops_linear = next_score;
                            } else {
                                flops_log2 = next_flops_log2;
                                read_write_log2 = next_read_write_log2;
                            }
                            let score = next_score;
                            let improved = improves_ranking_by_relative::<
                                TRACK_FLOPS,
                                TRACK_READ_WRITE,
                            >(score, best_total, 1e-12);
                            if improved {
                                best_total = score;
                                best_path = tree.to_path();
                            }
                        }
                    }
                }
                if TRACK_FLOPS && !TRACK_READ_WRITE {
                    tree.refresh_linear_total()?;
                    flops_linear = tree.step_linear_total;
                } else {
                    flops_log2 = tree.search_flops_log2();
                    read_write_log2 = tree.search_read_write_log2();
                }
            }
            Ok((best_path, best_total))
        })
        .collect::<Result<Vec<_>, String>>()?
        .into_iter()
        .min_by(|a, b| a.1.total_cmp(&b.1));
    match best {
        Some((p, _)) => {
            let stats = crate::path::simulate_path(net, &p)?;
            Ok(
                if objective.score_path_log2(&stats) <= objective.score_path_log2(&init_stats) {
                    (p, stats)
                } else {
                    (path.clone(), init_stats)
                },
            )
        }
        None => Ok((path.clone(), init_stats)),
    }
}

/// Reconfigure one path and return the resulting path and statistics.
pub fn reconfigure_path(
    net: &TensorNetwork,
    path: &SsaPath,
    subtree_size: usize,
    max_sweeps: usize,
) -> Result<(SsaPath, crate::path::PathStats), String> {
    reconfigure_path_with_objective(net, path, subtree_size, max_sweeps, PlannerObjective::FIXED)
}

pub fn reconfigure_path_with_objective(
    net: &TensorNetwork,
    path: &SsaPath,
    subtree_size: usize,
    max_sweeps: usize,
    objective: PlannerObjective,
) -> Result<(SsaPath, crate::path::PathStats), String> {
    match objective.kind() {
        ObjectiveKind::TotalFlops => {
            reconfigure_path_core::<true, false>(net, path, subtree_size, max_sweeps, objective)
        }
        ObjectiveKind::TotalReadWrite => {
            reconfigure_path_core::<false, true>(net, path, subtree_size, max_sweeps, objective)
        }
        ObjectiveKind::Weighted => {
            reconfigure_path_core::<true, true>(net, path, subtree_size, max_sweeps, objective)
        }
    }
}

fn reconfigure_path_core<const TRACK_FLOPS: bool, const TRACK_READ_WRITE: bool>(
    net: &TensorNetwork,
    path: &SsaPath,
    subtree_size: usize,
    max_sweeps: usize,
    objective: PlannerObjective,
) -> Result<(SsaPath, crate::path::PathStats), String> {
    let mut tree =
        CTreeCore::<TRACK_FLOPS, TRACK_READ_WRITE>::from_path_with_objective(net, path, objective)?;
    tree.reconfigure(net, subtree_size, max_sweeps)?;
    let new_path = tree.to_path();
    let stats = crate::path::simulate_path(net, &new_path)?;
    Ok((new_path, stats))
}

/// Deadline-aware [`reconfigure_path`].
///
/// Stops before the next local update. Every completed update leaves a valid
/// tree, so the incumbent is exported and scored instead of discarding work.
/// A `None` deadline uses the same path as [`reconfigure_path`].
pub fn reconfigure_path_until(
    net: &TensorNetwork,
    path: &SsaPath,
    subtree_size: usize,
    max_sweeps: usize,
    deadline: Option<Instant>,
) -> Result<(SsaPath, crate::path::PathStats), String> {
    reconfigure_path_until_with_objective(
        net,
        path,
        subtree_size,
        max_sweeps,
        deadline,
        PlannerObjective::FIXED,
    )
}

pub fn reconfigure_path_until_with_objective(
    net: &TensorNetwork,
    path: &SsaPath,
    subtree_size: usize,
    max_sweeps: usize,
    deadline: Option<Instant>,
    objective: PlannerObjective,
) -> Result<(SsaPath, crate::path::PathStats), String> {
    if deadline_reached(deadline) {
        return Err("reconfigure deadline exceeded".into());
    }
    let (path, stats, _) = reconfigure_path_cooperative_with_objective(
        net,
        path,
        subtree_size,
        max_sweeps,
        deadline,
        objective,
    )?;
    Ok((path, stats))
}

pub(crate) fn reconfigure_path_cooperative_with_objective(
    net: &TensorNetwork,
    path: &SsaPath,
    subtree_size: usize,
    max_sweeps: usize,
    deadline: Option<Instant>,
    objective: PlannerObjective,
) -> Result<(SsaPath, crate::path::PathStats, bool), String> {
    if deadline_reached(deadline) {
        return Ok((path.clone(), crate::path::simulate_path(net, path)?, false));
    }
    if deadline.is_none() {
        let (path, stats) =
            reconfigure_path_with_objective(net, path, subtree_size, max_sweeps, objective)?;
        return Ok((path, stats, true));
    }
    match objective.kind() {
        ObjectiveKind::TotalFlops => reconfigure_path_until_core::<true, false>(
            net,
            path,
            subtree_size,
            max_sweeps,
            deadline,
            objective,
        ),
        ObjectiveKind::TotalReadWrite => reconfigure_path_until_core::<false, true>(
            net,
            path,
            subtree_size,
            max_sweeps,
            deadline,
            objective,
        ),
        ObjectiveKind::Weighted => reconfigure_path_until_core::<true, true>(
            net,
            path,
            subtree_size,
            max_sweeps,
            deadline,
            objective,
        ),
    }
}

fn replay_objective_score(
    net: &TensorNetwork,
    path: &SsaPath,
    objective: PlannerObjective,
) -> Result<f64, String> {
    match objective.kind() {
        ObjectiveKind::TotalFlops => {
            let (flops_log2, _) = simulate_path_flops_log2_and_legs(net, path)?;
            Ok(objective.score_terms_log2(flops_log2, f64::NEG_INFINITY))
        }
        ObjectiveKind::TotalReadWrite => {
            let tree = CTreeCore::<false, true>::from_path_with_objective(net, path, objective)?;
            Ok(objective.score_terms_log2(f64::NEG_INFINITY, tree.search_read_write_log2()))
        }
        ObjectiveKind::Weighted => {
            Ok(objective.score_path_log2(&crate::path::simulate_path(net, path)?))
        }
    }
}

fn reconfigure_path_until_core<const TRACK_FLOPS: bool, const TRACK_READ_WRITE: bool>(
    net: &TensorNetwork,
    path: &SsaPath,
    subtree_size: usize,
    max_sweeps: usize,
    deadline: Option<Instant>,
    objective: PlannerObjective,
) -> Result<(SsaPath, crate::path::PathStats, bool), String> {
    let mut tree =
        CTreeCore::<TRACK_FLOPS, TRACK_READ_WRITE>::from_path_with_objective(net, path, objective)?;
    let completed = tree
        .reconfigure_until(net, subtree_size, max_sweeps, deadline)?
        .is_some();
    let new_path = tree.to_path();
    let stats = crate::path::simulate_path(net, &new_path)?;
    Ok((new_path, stats, completed))
}

/// Reconfigure local subtrees and stop after consecutive low-improvement sweeps.
///
/// The caller chooses the objective, subtree size, sweep limit, minimum sweep
/// count, patience, and relative gain threshold. Once `minimum_sweeps` complete,
/// `patience` consecutive gains below `relative_gain_threshold` stop the search.
/// Both counts must be positive, `minimum_sweeps` must be at least `patience`,
/// and the threshold must be finite and non-negative.
///
/// The objective is replayed after each complete sweep. A cooperative deadline
/// discards an incomplete sweep and returns the latest complete path. The
/// returned report contains sweep scores, gains, and the stop reason. Compare
/// the result with the supplied path when deciding whether to replace it.
#[allow(clippy::too_many_arguments)]
pub fn reconfigure_path_early_stop_with_objective(
    net: &TensorNetwork,
    path: &SsaPath,
    subtree_size: usize,
    max_sweeps: usize,
    minimum_sweeps: usize,
    patience: usize,
    relative_gain_threshold: f64,
    deadline: Option<Instant>,
    objective: PlannerObjective,
) -> Result<
    (
        SsaPath,
        crate::path::PathStats,
        ObjectiveReconfigurationEarlyStopReport,
    ),
    String,
> {
    match objective.kind() {
        ObjectiveKind::TotalFlops => reconfigure_path_early_stop_core::<true, false>(
            net,
            path,
            subtree_size,
            max_sweeps,
            minimum_sweeps,
            patience,
            relative_gain_threshold,
            deadline,
            objective,
        ),
        ObjectiveKind::TotalReadWrite => reconfigure_path_early_stop_core::<false, true>(
            net,
            path,
            subtree_size,
            max_sweeps,
            minimum_sweeps,
            patience,
            relative_gain_threshold,
            deadline,
            objective,
        ),
        ObjectiveKind::Weighted => reconfigure_path_early_stop_core::<true, true>(
            net,
            path,
            subtree_size,
            max_sweeps,
            minimum_sweeps,
            patience,
            relative_gain_threshold,
            deadline,
            objective,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn reconfigure_path_early_stop_core<const TRACK_FLOPS: bool, const TRACK_READ_WRITE: bool>(
    net: &TensorNetwork,
    path: &SsaPath,
    subtree_size: usize,
    max_sweeps: usize,
    minimum_sweeps: usize,
    patience: usize,
    relative_gain_threshold: f64,
    deadline: Option<Instant>,
    objective: PlannerObjective,
) -> Result<
    (
        SsaPath,
        crate::path::PathStats,
        ObjectiveReconfigurationEarlyStopReport,
    ),
    String,
> {
    let mut tree =
        CTreeCore::<TRACK_FLOPS, TRACK_READ_WRITE>::from_path_with_objective(net, path, objective)?;
    let report = tree.reconfigure_early_stop(
        net,
        subtree_size,
        max_sweeps,
        minimum_sweeps,
        patience,
        relative_gain_threshold,
        deadline,
    )?;
    let result_path = tree.to_path();
    let stats = crate::path::simulate_path(net, &result_path)?;
    Ok((result_path, stats, report))
}

#[inline]
fn deadline_reached(deadline: Option<Instant>) -> bool {
    deadline.map(|d| Instant::now() >= d).unwrap_or(false)
}

#[cfg(test)]
mod fixed_objective_tests {
    use super::fallback_init_index;
    use super::{
        anneal_path, fixed_score_log2, improves_by_relative, improves_ranking_by_relative,
        reconfigure_path, reconfigure_path_cooperative_with_objective,
        reconfigure_path_early_stop_with_objective, reconfigure_path_until,
        reconfigure_path_with_objective, relative_change, replay_objective_score, temper_path,
        temper_path_with_objective, temper_paths, temper_paths_with_objective, treesa_path, CTree,
        CTreeCore, LogSumTree, RotScratch,
    };
    use crate::network::TensorNetwork;
    use crate::objective::PlannerObjective;
    use crate::path::{PathStats, SsaPath};

    fn stats(log2_flops: f64, log2_read_write: f64) -> PathStats {
        PathStats {
            log10_flops: log2_flops * std::f64::consts::LOG10_2,
            log2_max_size: 0.0,
            log2_max_contraction_size: 0.0,
            log2_total_size: 0.0,
            log2_read_write,
            log2_peak_size: 0.0,
        }
    }

    fn assert_close(actual: f64, expected: f64) {
        let tolerance = 1e-12 * actual.abs().max(expected.abs()).max(1.0);
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual={actual} expected={expected} tolerance={tolerance}"
        );
    }

    #[test]
    fn fallback_uses_fixed_objective() {
        let low_flops_high_write = stats(10.0, 20.0);
        let high_flops_low_write = stats(12.0, 0.0);
        let candidates = [low_flops_high_write, high_flops_low_write];

        assert_eq!(fallback_init_index(&candidates, PlannerObjective::FIXED), 1);
    }

    #[test]
    fn fixed_score_is_finite_when_the_weighted_linear_sum_overflows() {
        let log2_flops: f64 = 1017.59;
        let log2_read_write: f64 = 1018.01;
        let flops = log2_flops.exp2();
        let read_write = log2_read_write.exp2();
        assert!(flops.is_finite());
        assert!(read_write.is_finite());
        assert!((flops + crate::READ_WRITE_WEIGHT * read_write).is_infinite());

        let actual = fixed_score_log2(log2_flops, log2_read_write);
        let expected = PlannerObjective::FIXED.score_path_log2(&stats(log2_flops, log2_read_write));
        assert!(actual.is_finite());
        assert!((actual - expected).abs() < 1e-12);
        assert!(actual > 1024.0 && actual < 1025.0);
    }

    #[test]
    fn fixed_log_score_matches_normal_linear_scoring() {
        let score_log2 = fixed_score_log2(10.0f64.log2(), 2.0f64.log2());
        let expected = crate::FLOPS_WEIGHT * 10.0 + crate::READ_WRITE_WEIGHT * 2.0;
        assert_close(score_log2.exp2(), expected);
        assert_close(relative_change(score_log2, score_log2 + 1.0), 1.0);
        assert!(improves_by_relative(score_log2 - 1.0, score_log2, 0.25));
    }

    #[test]
    fn relative_improvement_uses_the_specialized_score_domain() {
        let minimum_gain = 0.0025;

        assert!(!improves_ranking_by_relative::<true, false>(
            999.0,
            1000.0,
            minimum_gain,
        ));
        assert!(improves_ranking_by_relative::<true, false>(
            997.0,
            1000.0,
            minimum_gain,
        ));

        assert!(!improves_ranking_by_relative::<true, true>(
            999.0f64.log2(),
            1000.0f64.log2(),
            minimum_gain,
        ));
        assert!(improves_ranking_by_relative::<true, true>(
            997.0f64.log2(),
            1000.0f64.log2(),
            minimum_gain,
        ));
    }

    fn large_finite_network() -> (TensorNetwork, SsaPath) {
        let inputs = (0..20).chain(0..20).map(|leg| vec![leg]).collect();
        let size_dict = (0..20).map(|leg| (leg, 1usize << 60)).collect();
        let mut path = vec![(0, 1)];
        let mut partial = 40;
        for leaf in 2..20 {
            path.push((leaf, partial));
            partial += 1;
        }
        for leaf in 20..40 {
            path.push((leaf, partial));
            partial += 1;
        }
        (
            TensorNetwork {
                name: "large-finite-objective".into(),
                inputs,
                output: vec![],
                size_dict,
            },
            path,
        )
    }

    #[test]
    fn large_finite_objective_keeps_tree_search_active() {
        let (net, path) = large_finite_network();
        let stats = crate::path::simulate_path(&net, &path).unwrap();
        assert!(PlannerObjective::FIXED.score_path_log2(&stats).is_finite());
        let tree = CTree::from_path(&net, &path).unwrap();
        assert!(tree.total_cost_log2(&net).is_finite());
        assert!(tree.total_read_write_log2(&net).is_finite());
        assert!(tree.total_cost_log2(&net).exp2().is_infinite());
        assert!(tree.total_read_write_log2(&net).exp2().is_infinite());

        let (_, annealed) = anneal_path(&net, &path, 4, 2, 0.05, 1e-4).unwrap();
        let (_, tempered) = temper_path(&net, &path, 2, 1, 4, 1e-4, 0.1, 0, 8, 3).unwrap();
        let (_, treesa) = treesa_path(&net, &path, 1, 0.1, 1.0, 2, 1, 0, 8, 4).unwrap();
        for result in [annealed, tempered, treesa] {
            assert!(PlannerObjective::FIXED.score_path_log2(&result).is_finite());
        }
    }

    #[test]
    fn log_domain_acceptance_is_invariant_across_linear_overflow_boundary() {
        fn proposal_scores(shift: f64) -> (f64, f64) {
            let flops = LogSumTree::from_values(&[8.5 + shift, 8.0 + shift, 7.0 + shift]);
            let read_write = LogSumTree::from_values(&[9.0 + shift, 8.5 + shift, 7.5 + shift]);
            (
                fixed_score_log2(flops.total(), read_write.total()),
                fixed_score_log2(
                    flops.total_replacing((0, 8.25 + shift), None),
                    read_write.total_replacing((0, 9.2 + shift), None),
                ),
            )
        }

        let low = proposal_scores(0.0);
        let shift = 1016.0;
        let high = proposal_scores(shift);
        assert!(high.0 > 1024.0 && high.0.is_finite());
        assert_close(high.0 - low.0, shift);
        assert_close(high.1 - low.1, shift);
        assert_close(high.1 - high.0, low.1 - low.0);
        assert_close(
            relative_change(high.0, high.1),
            relative_change(low.0, low.1),
        );

        let temperature = 0.05;
        let draw = 0.4;
        let low_accept =
            low.1 <= low.0 || draw < (-relative_change(low.0, low.1) / temperature).exp();
        let high_accept =
            high.1 <= high.0 || draw < (-relative_change(high.0, high.1) / temperature).exp();
        assert_eq!(low_accept, high_accept);
    }

    #[test]
    fn replacing_a_dominant_term_recovers_the_next_largest_term() {
        let sums = LogSumTree::from_values(&[1000.0, 900.0]);
        assert_eq!(sums.total(), 1000.0);
        assert_close(sums.total_replacing((0, 800.0), None), 900.0);
    }

    fn fixed_reversal_net() -> TensorNetwork {
        TensorNetwork {
            name: "reconfiguration-fixed-objective-reversal".into(),
            inputs: vec![vec![0, 1], vec![0, 2], vec![1, 3], vec![2, 4]],
            output: vec![3, 4],
            size_dict: [(0, 6), (1, 4), (2, 6), (3, 5), (4, 8)]
                .into_iter()
                .collect(),
        }
    }

    #[test]
    fn pure_flops_linear_rotation_tracks_authoritative_replay_and_refreshes_drift() {
        let net = fixed_reversal_net();
        let input: SsaPath = vec![(0, 1), (3, 4), (2, 5)];
        let objective = PlannerObjective::new(1.0, 0.0).unwrap();
        let mut tree =
            CTreeCore::<true, false>::from_path_with_objective(&net, &input, objective).unwrap();
        let b = net.n_tensors();
        let mut scratch = RotScratch::new();

        for ordinal in 0..64 {
            let change = tree
                .rotate_delta(b, ordinal % 2 == 0, &mut scratch)
                .unwrap()
                .expect("chosen node remains rotatable");
            tree.rotate_apply(b, ordinal % 2 == 0, change);
            let replay_log2 = crate::path::simulate_path_flops_log2_and_legs(&net, &tree.to_path())
                .unwrap()
                .0;
            let replay_linear = replay_log2.exp2();
            assert_close(tree.step_linear_total, replay_linear);
            tree.refresh_linear_total().unwrap();
            assert_close(tree.step_linear_total, replay_linear);
        }
    }

    #[test]
    fn pure_flops_linear_overflow_is_rejected_before_search() {
        // Each input is individually valid, but the deliberately bad outer-product
        // prefix accumulates 36 open legs and exceeds the f64 exponent range.
        let inputs = (0..72u32)
            .map(|tensor| vec![tensor % 36])
            .collect::<Vec<_>>();
        let size_dict = (0..36u32).map(|leg| (leg, 1usize << 30)).collect();
        let mut path = vec![(0, 1)];
        let mut partial = 72usize;
        for leaf in 2..36usize {
            path.push((leaf, partial));
            partial += 1;
        }
        for leaf in 36..72usize {
            path.push((leaf, partial));
            partial += 1;
        }
        let net = TensorNetwork {
            name: "linear-f64-overflow-rejection".into(),
            inputs,
            output: vec![],
            size_dict,
        };
        let error = CTreeCore::<true, false>::from_path_with_objective(
            &net,
            &path,
            PlannerObjective::new(1.0, 0.0).unwrap(),
        )
        .err()
        .expect("linear f64 overflow must reject the cell");
        assert!(error.contains("pure-FLOPs linear f64 overflow"), "{error}");
    }

    #[test]
    fn runtime_objective_dispatch_preserves_default_and_returns_complete_metrics() {
        fn assert_stats_match(actual: &PathStats, expected: &PathStats) {
            assert_eq!(actual.log10_flops.to_bits(), expected.log10_flops.to_bits());
            assert_eq!(
                actual.log2_max_size.to_bits(),
                expected.log2_max_size.to_bits()
            );
            assert_eq!(
                actual.log2_max_contraction_size.to_bits(),
                expected.log2_max_contraction_size.to_bits()
            );
            assert_eq!(
                actual.log2_total_size.to_bits(),
                expected.log2_total_size.to_bits()
            );
            assert_eq!(
                actual.log2_read_write.to_bits(),
                expected.log2_read_write.to_bits()
            );
            assert_eq!(
                actual.log2_peak_size.to_bits(),
                expected.log2_peak_size.to_bits()
            );
        }

        let net = fixed_reversal_net();
        let input: SsaPath = vec![(0, 1), (3, 4), (2, 5)];

        let default = temper_paths(
            &net,
            std::slice::from_ref(&input),
            2,
            2,
            8,
            1e-4,
            0.1,
            0,
            8,
            17,
            0,
        )
        .unwrap();
        let explicit = temper_paths_with_objective(
            &net,
            std::slice::from_ref(&input),
            2,
            2,
            8,
            1e-4,
            0.1,
            0,
            8,
            17,
            0,
            PlannerObjective::FIXED,
        )
        .unwrap();
        assert_eq!(default.0, explicit.0);
        assert_eq!(
            default.1.log10_flops.to_bits(),
            explicit.1.log10_flops.to_bits()
        );
        assert_eq!(
            default.1.log2_read_write.to_bits(),
            explicit.1.log2_read_write.to_bits()
        );

        let input_stats = crate::path::simulate_path(&net, &input).unwrap();
        for objective in [
            PlannerObjective::new(1.0, 0.0).unwrap(),
            PlannerObjective::new(0.0, 1.0).unwrap(),
            PlannerObjective::new(3.0, 7.0).unwrap(),
        ] {
            let results = [
                reconfigure_path_with_objective(&net, &input, 8, 4, objective).unwrap(),
                temper_path_with_objective(&net, &input, 2, 2, 8, 1e-4, 0.1, 0, 8, 19, objective)
                    .unwrap(),
            ];
            for (path, stats) in results {
                assert_eq!(path.len(), net.n_tensors() - 1);
                let replay = crate::path::simulate_path(&net, &path).unwrap();
                assert_stats_match(&stats, &replay);
                assert!(
                    objective.score_path_log2(&stats)
                        <= objective.score_path_log2(&input_stats) + 1e-12
                );
            }
        }

        let read_write_objective = PlannerObjective::new(0.0, 1.0).unwrap();
        let read_write_tree =
            CTreeCore::<false, true>::from_path_with_objective(&net, &input, read_write_objective)
                .unwrap();
        assert_eq!(read_write_tree.search_flops_log2(), f64::NEG_INFINITY);
        assert_eq!(
            read_write_tree.search_read_write_log2().to_bits(),
            input_stats.log2_read_write.to_bits()
        );
    }

    #[test]
    fn early_stop_replays_only_the_active_objective_and_returns_full_final_metrics() {
        let net = fixed_reversal_net();
        let input: SsaPath = vec![(0, 1), (3, 4), (2, 5)];
        for objective in [
            PlannerObjective::new(1.0, 0.0).unwrap(),
            PlannerObjective::new(0.0, 1.0).unwrap(),
            PlannerObjective::new(3.0, 7.0).unwrap(),
        ] {
            let full_input = crate::path::simulate_path(&net, &input).unwrap();
            assert_close(
                replay_objective_score(&net, &input, objective).unwrap(),
                objective.score_path_log2(&full_input),
            );

            let (path, stats, report) = reconfigure_path_early_stop_with_objective(
                &net, &input, 8, 10, 4, 2, 1e-4, None, objective,
            )
            .unwrap();
            let full_result = crate::path::simulate_path(&net, &path).unwrap();
            assert_eq!(
                stats.log10_flops.to_bits(),
                full_result.log10_flops.to_bits()
            );
            assert_eq!(
                stats.log2_read_write.to_bits(),
                full_result.log2_read_write.to_bits()
            );
            assert!(report
                .objective_log2_by_sweep
                .iter()
                .all(|score| score.is_finite()));
        }
    }

    #[test]
    fn tree_read_write_matches_authoritative_path_with_repeated_leaf_leg() {
        let net = TensorNetwork {
            name: "tree-read-write-repeated-leaf-leg".into(),
            inputs: vec![vec![0, 0, 1], vec![1]],
            output: vec![],
            size_dict: [(0, 3), (1, 2)].into_iter().collect(),
        };
        let path = vec![(0, 1)];
        let tree = CTree::from_path(&net, &path).unwrap();
        let replay = crate::path::simulate_path(&net, &path).unwrap();

        assert_close(tree.total_read_write_log2(&net), 21.0f64.log2());
        assert_close(tree.total_read_write_log2(&net), replay.log2_read_write);
    }

    #[test]
    fn single_tensor_tree_has_no_contraction_read_write_cost() {
        let net = TensorNetwork {
            name: "single-tensor-read-write".into(),
            inputs: vec![vec![0]],
            output: vec![0],
            size_dict: [(0, 2)].into_iter().collect(),
        };
        let path = Vec::new();
        let tree = CTree::from_path(&net, &path).unwrap();
        let replay = crate::path::simulate_path(&net, &path).unwrap();
        assert_eq!(tree.total_read_write_log2(&net), f64::NEG_INFINITY);
        assert_eq!(tree.total_read_write_log2(&net), replay.log2_read_write);
    }

    #[test]
    fn rotation_cost_delta_matches_full_replay() {
        let net = fixed_reversal_net();
        let input: SsaPath = vec![(0, 1), (3, 4), (2, 5)];

        for promote_left in [false, true] {
            let mut tree = CTree::from_path(&net, &input).unwrap();
            let b = net.n_tensors();
            let mut scratch = RotScratch::new();
            let change = tree
                .rotate_delta(b, promote_left, &mut scratch)
                .unwrap()
                .expect("chosen node is rotatable");
            let expected_flops_log2 = change.next_flops_log2;
            let expected_read_write_log2 = change.next_read_write_log2;
            tree.rotate_apply(b, promote_left, change);
            let after_flops_log2 = tree.total_cost_log2(&net);
            let after_read_write_log2 = tree.total_read_write_log2(&net);
            let replay = crate::path::simulate_path(&net, &tree.to_path()).unwrap();

            assert_close(after_flops_log2, expected_flops_log2);
            assert_close(
                after_flops_log2,
                replay.log10_flops / std::f64::consts::LOG10_2,
            );
            assert_close(after_read_write_log2, expected_read_write_log2);
            assert_close(after_read_write_log2, replay.log2_read_write);
        }
    }

    #[test]
    fn fixed_objective_local_dp_can_accept_higher_flops_for_lower_writes() {
        let net = fixed_reversal_net();
        let input: SsaPath = vec![(0, 1), (3, 4), (2, 5)];
        let input_stats = crate::path::simulate_path(&net, &input).unwrap();
        let (fixed_path, fixed_stats) = reconfigure_path(&net, &input, 8, 10).unwrap();

        assert_eq!(fixed_path, vec![(0, 1), (2, 4), (3, 5)]);
        assert!(fixed_stats.log10_flops > input_stats.log10_flops);
        assert!(fixed_stats.log2_total_size < input_stats.log2_total_size);
        assert!(
            PlannerObjective::FIXED.score_path_log2(&fixed_stats)
                < PlannerObjective::FIXED.score_path_log2(&input_stats)
        );
    }

    #[test]
    fn deadline_free_reconfiguration_matches_default_entry() {
        let net = fixed_reversal_net();
        let input: SsaPath = vec![(0, 2), (1, 4), (3, 5)];
        let direct = reconfigure_path(&net, &input, 8, 10).unwrap();
        let deadline_free = reconfigure_path_until(&net, &input, 8, 10, None).unwrap();
        assert_eq!(direct.0, deadline_free.0);
        assert_eq!(
            direct.1.log10_flops.to_bits(),
            deadline_free.1.log10_flops.to_bits()
        );
        assert_eq!(
            direct.1.log2_total_size.to_bits(),
            deadline_free.1.log2_total_size.to_bits()
        );
        assert_eq!(
            direct.1.log2_read_write.to_bits(),
            deadline_free.1.log2_read_write.to_bits()
        );
    }

    #[test]
    fn cooperative_reconfiguration_reports_an_expired_deadline() {
        let net = fixed_reversal_net();
        let input: SsaPath = vec![(0, 2), (1, 4), (3, 5)];
        let input_stats = crate::path::simulate_path(&net, &input).unwrap();
        let expired = std::time::Instant::now() - std::time::Duration::from_millis(1);

        let (path, stats, completed) = reconfigure_path_cooperative_with_objective(
            &net,
            &input,
            8,
            10,
            Some(expired),
            PlannerObjective::FIXED,
        )
        .unwrap();

        assert!(!completed);
        assert_eq!(path, input);
        assert_eq!(
            stats.log10_flops.to_bits(),
            input_stats.log10_flops.to_bits()
        );
        assert_eq!(
            stats.log2_total_size.to_bits(),
            input_stats.log2_total_size.to_bits()
        );
        assert_eq!(
            stats.log2_read_write.to_bits(),
            input_stats.log2_read_write.to_bits()
        );
    }
}
