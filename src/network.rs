//! Tensor-network structure represented by tensor leg lists and leg dimensions.
//! Supports cotengra-style `inputs/output/size_dict` JSON and OMECO-style
//! `ixs/iy/sizes` JSON. A leg may form a hyperedge, and output legs remain open.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use rand::seq::SliceRandom;
use rand::Rng;

use crate::tensor::{checked_numel, DenseTensor, Scalar};

pub type LegId = u32;

/// Maps external [`LegId`] values to compact cache slots.
///
/// Dense small labels are indexed directly. Sparse labels use an O(number of
/// legs) map so untrusted large labels cannot force huge cache allocations.
#[derive(Clone, Debug)]
pub(crate) struct LegIndex {
    sparse_slots: Option<HashMap<LegId, usize>>,
    len: usize,
}

impl LegIndex {
    pub(crate) fn new(legs: impl IntoIterator<Item = LegId>) -> Self {
        let mut legs: Vec<LegId> = legs.into_iter().collect();
        legs.sort_unstable();
        legs.dedup();

        let dense_len = legs
            .last()
            .copied()
            .and_then(|leg| usize::try_from(leg).ok())
            .and_then(|leg| leg.checked_add(1));
        // Allow small gaps without scaling memory with the largest label.
        let dense_limit = legs.len().saturating_mul(4).max(64);
        if let Some(len) = dense_len.filter(|&len| len <= dense_limit) {
            return Self {
                sparse_slots: None,
                len,
            };
        }

        let sparse_slots: HashMap<LegId, usize> = legs
            .into_iter()
            .enumerate()
            .map(|(slot, leg)| (leg, slot))
            .collect();
        let len = sparse_slots.len();
        Self {
            sparse_slots: Some(sparse_slots),
            len,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub(crate) fn slot(&self, leg: LegId) -> usize {
        match &self.sparse_slots {
            Some(slots) => *slots
                .get(&leg)
                .unwrap_or_else(|| panic!("LegIndex 缺少已校验的腿 {leg}")),
            None => usize::try_from(leg)
                .ok()
                .filter(|&slot| slot < self.len)
                .unwrap_or_else(|| panic!("LegIndex 缺少已校验的腿 {leg}")),
        }
    }
}

/// Loads input tensors from little-endian binary data.
/// Supports f32, f64, Complex32, and Complex64; complex values store real then imaginary parts.
///
/// Tensors follow `net.inputs` order and row-major layout, including repeated
/// legs. The byte length must equal the sum of `numel * T::NBYTES`.
/// `tnmpi --data` uses the same layout but accepts only f64.
pub fn load_tensors_bin<T: Scalar>(
    path: &Path,
    net: &TensorNetwork,
) -> Result<Vec<DenseTensor<T>>, String> {
    net.validate()?;
    let bytes = std::fs::read(path).map_err(|e| format!("读数据文件失败: {e}"))?;
    let mut shapes = Vec::with_capacity(net.inputs.len());
    let mut numels = Vec::with_capacity(net.inputs.len());
    for (i, legs) in net.inputs.iter().enumerate() {
        let shape: Vec<usize> = legs
            .iter()
            .map(|&l| net.try_dim(l))
            .collect::<Result<_, _>>()?;
        let numel = checked_numel(&shape)
            .map_err(|e| format!("输入张量 {i} 的 shape {:?} 非法: {e}", shape))?;
        shapes.push(shape);
        numels.push(numel);
    }
    let total_elems = numels.iter().try_fold(0usize, |acc, &n| {
        acc.checked_add(n)
            .ok_or_else(|| "输入张量总元素数溢出 usize".to_string())
    })?;
    let expect = total_elems
        .checked_mul(T::NBYTES)
        .ok_or_else(|| "输入张量总字节数溢出 usize".to_string())?;
    if bytes.len() != expect {
        return Err(format!(
            "数据文件字节数 {} 与网络期望 {} 不符（应为 Σ numel × {} 字节的小端；\
             检查 dtype 与文件是否匹配该网络）",
            bytes.len(),
            expect,
            T::NBYTES
        ));
    }
    let mut tensors = Vec::with_capacity(net.inputs.len());
    let mut off = 0usize;
    for (i, (&numel, shape)) in numels.iter().zip(shapes).enumerate() {
        let mut data = Vec::with_capacity(numel);
        for _ in 0..numel {
            let end = off
                .checked_add(T::NBYTES)
                .ok_or_else(|| "读取输入张量时字节偏移溢出".to_string())?;
            data.push(T::from_le_bytes(&bytes[off..end]));
            off = end;
        }
        tensors.push(
            DenseTensor::try_from_data(shape, data)
                .map_err(|e| format!("输入张量 {i} 布局非法: {e}"))?,
        );
    }
    Ok(tensors)
}

#[derive(Clone, Debug)]
pub struct TensorNetwork {
    pub name: String,
    /// Leg list for each input tensor; repeated legs denote a trace.
    pub inputs: Vec<Vec<LegId>>,
    /// Open legs in final output order.
    pub output: Vec<LegId>,
    /// Leg dimensions.
    pub size_dict: HashMap<LegId, usize>,
}

impl TensorNetwork {
    pub fn n_tensors(&self) -> usize {
        self.inputs.len() // Number of tensors.
    }

    pub fn dim(&self, leg: LegId) -> usize {
        self.try_dim(leg)
            .unwrap_or_else(|e| panic!("TensorNetwork::dim: {e}"))
    }

    /// Fallible dimension lookup for external-input boundaries.
    pub fn try_dim(&self, leg: LegId) -> Result<usize, String> {
        self.size_dict
            .get(&leg)
            .copied()
            .ok_or_else(|| format!("腿 {leg} 缺少 size_dict 维度"))
    }

    pub fn log2_dim(&self, leg: LegId) -> f64 {
        (self.dim(leg) as f64).log2() // Log2 dimension of one leg.
    }

    /// Validates that the network is safe for current planners and executors.
    ///
    /// Because fields are public, call this before GEMM or binary loading. It
    /// checks dimensions, element-count overflow, and unsupported leg layouts.
    pub fn validate(&self) -> Result<(), String> {
        if self.inputs.is_empty() {
            return Err("TensorNetwork 至少需要一个输入张量；空网络没有统一的收缩语义".into());
        }
        let max_gemm_dim = isize::MAX as usize;
        for (&leg, &dim) in &self.size_dict {
            if dim == 0 {
                return Err(format!("腿 {leg} 的维度为 0；ArcTN 暂不支持零维网络"));
            }
            if dim > max_gemm_dim {
                return Err(format!(
                    "腿 {leg} 的维度 {dim} 超过 GEMM stride 可表示范围 {max_gemm_dim}"
                ));
            }
        }

        let mut held = HashSet::new();
        for (i, legs) in self.inputs.iter().enumerate() {
            let mut counts: HashMap<LegId, usize> = HashMap::new();
            let mut shape = Vec::with_capacity(legs.len());
            for &leg in legs {
                let dim = self.try_dim(leg)?;
                *counts.entry(leg).or_insert(0) += 1;
                held.insert(leg);
                shape.push(dim);
            }
            if let Some((&leg, &count)) = counts.iter().find(|(_, &count)| count > 2) {
                return Err(format!(
                    "腿 {leg} 在输入张量 {i} 出现 {count} 次；最多支持 2 次"
                ));
            }
            checked_numel(&shape)
                .map_err(|e| format!("输入张量 {i} 的 shape {:?} 非法: {e}", shape))?;
        }

        let mut output_seen = HashSet::new();
        let mut output_shape = Vec::with_capacity(self.output.len());
        for &leg in &self.output {
            if !output_seen.insert(leg) {
                return Err(format!("output 含重复腿，暂不支持: {:?}", self.output));
            }
            if !held.contains(&leg) {
                return Err(format!("output 腿 {leg} 不属于任何输入张量"));
            }
            output_shape.push(self.try_dim(leg)?);
        }
        checked_numel(&output_shape)
            .map_err(|e| format!("output shape {:?} 非法: {e}", output_shape))?;
        if let Some(leg) = self
            .size_dict
            .keys()
            .copied()
            .filter(|leg| !held.contains(leg))
            .min()
        {
            return Err(format!("size_dict 中的腿 {leg} 未被任何输入张量使用"));
        }
        Ok(())
    }

    /// Maps each leg to distinct input tensors that hold it.
    pub fn leg_holders(&self) -> HashMap<LegId, Vec<usize>> {
        let mut m: HashMap<LegId, Vec<usize>> = HashMap::new();
        for (t, legs) in self.inputs.iter().enumerate() {
            for &l in legs {
                let v = m.entry(l).or_default();
                if v.last() != Some(&t) {
                    v.push(t);
                }
            }
        }
        m
    }

    /// Loads a network from `.net.json` and interns labels as consecutive IDs.
    ///
    /// This discards original labels. Use [`Self::load_json_with_labels`] when
    /// results must be mapped back to external labels.
    pub fn load_json(path: &Path) -> Result<Self, String> {
        Self::load_json_with_labels(path).map(|(net, _)| net)
    }

    /// Loads JSON and returns the mapping from `LegId` to original labels.
    /// IDs follow first occurrence in `inputs`, then `output`.
    pub fn load_json_with_labels(path: &Path) -> Result<(Self, Vec<String>), String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("读文件失败: {e}"))?;
        Self::parse_json_with_labels(&text)
    }

    /// Parses either JSON schema accepted by [`Self::load_json_with_labels`].
    ///
    /// Exactly one complete schema must be present; mixed or partial schemas
    /// are rejected before field parsing.
    fn parse_json_with_labels(text: &str) -> Result<(Self, Vec<String>), String> {
        let v: serde_json::Value =
            serde_json::from_str(text).map_err(|e| format!("JSON 解析失败: {e}"))?;
        let root = v
            .as_object()
            .ok_or_else(|| "JSON 根节点必须是对象".to_string())?;
        let canonical_keys = ["inputs", "output", "size_dict"];
        let omeco_keys = ["ixs", "iy", "sizes"];
        let canonical_count = canonical_keys
            .iter()
            .filter(|&&key| root.contains_key(key))
            .count();
        let omeco_count = omeco_keys
            .iter()
            .filter(|&&key| root.contains_key(key))
            .count();
        let (inputs_key, output_key, sizes_key) = match (canonical_count, omeco_count) {
            (3, 0) => ("inputs", "output", "size_dict"),
            (0, 3) => ("ixs", "iy", "sizes"),
            (canonical, omeco) if canonical > 0 && omeco > 0 => {
                return Err(
                    "JSON 不能混用 canonical inputs/output/size_dict 与 OMECO ixs/iy/sizes schema"
                        .to_string(),
                );
            }
            (canonical, 0) if canonical > 0 => {
                let missing: Vec<&str> = canonical_keys
                    .iter()
                    .copied()
                    .filter(|key| !root.contains_key(*key))
                    .collect();
                return Err(format!(
                    "canonical schema 残缺：缺少 {}",
                    missing.join(", ")
                ));
            }
            (0, omeco) if omeco > 0 => {
                let missing: Vec<&str> = omeco_keys
                    .iter()
                    .copied()
                    .filter(|key| !root.contains_key(*key))
                    .collect();
                return Err(format!("OMECO schema 残缺：缺少 {}", missing.join(", ")));
            }
            (0, 0) => {
                return Err(
                    "缺少完整的网络 schema：需要 inputs/output/size_dict 或 ixs/iy/sizes"
                        .to_string(),
                );
            }
            _ => unreachable!("三字段 schema 的计数只能在 0..=3"),
        };
        let name = root
            .get("name")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("unnamed")
            .to_string();

        let mut intern: HashMap<String, LegId> = HashMap::new();
        let mut label_is_integer: HashMap<String, bool> = HashMap::new();
        let mut intern_leg = |x: &serde_json::Value| -> Result<LegId, String> {
            let (key, is_integer) = if let Some(n) = x.as_u64() {
                (n.to_string(), true) // Normalize nonnegative integer labels as string keys.
            } else if let Some(s) = x.as_str() {
                (s.to_string(), false)
            } else {
                return Err(format!("无法解析腿标签: {x}"));
            };
            if let Some(&prior_is_integer) = label_is_integer.get(&key) {
                if prior_is_integer != is_integer {
                    return Err(format!(
                        "不同 JSON 腿标签在正规化后冲突: 整数/字符串 {key:?}"
                    ));
                }
            } else {
                label_is_integer.insert(key.clone(), is_integer);
            }
            let next = intern.len() as LegId;
            Ok(*intern.entry(key).or_insert(next))
        };

        let inputs = root[inputs_key]
            .as_array()
            .ok_or_else(|| format!("{inputs_key} 必须是数组"))?
            .iter()
            .map(|legs| {
                legs.as_array()
                    .ok_or_else(|| format!("{inputs_key} 元素不是数组"))?
                    .iter()
                    .map(&mut intern_leg)
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        let output = root[output_key]
            .as_array()
            .ok_or_else(|| format!("{output_key} 必须是数组"))?
            .iter()
            .map(&mut intern_leg)
            .collect::<Result<Vec<_>, _>>()?;
        let mut size_dict = HashMap::new();
        for (k, val) in root[sizes_key]
            .as_object()
            .ok_or_else(|| format!("{sizes_key} 必须是对象"))?
        {
            let leg = *intern.get(k).ok_or_else(|| {
                format!("{sizes_key} 中的腿 {k} 未在 {inputs_key}/{output_key} 出现")
            })?;
            let raw = val.as_u64().ok_or_else(|| format!("维度不是整数: {val}"))?;
            let d =
                usize::try_from(raw).map_err(|_| format!("维度 {raw} 超出当前平台 usize 范围"))?;
            size_dict.insert(leg, d);
        }
        // Every referenced leg must have a dimension.
        for (key, &leg) in &intern {
            if !size_dict.contains_key(&leg) {
                return Err(format!("腿 {key} 缺少 size_dict 维度"));
            }
        }
        // Restore original labels in LegId order.
        let mut labels = vec![String::new(); intern.len()];
        for (key, &leg) in &intern {
            labels[leg as usize] = key.clone();
        }
        let net = TensorNetwork {
            name,
            inputs,
            output,
            size_dict,
        };
        net.validate()?;
        Ok((net, labels))
    }

    /// Serializes the network as `.net.json`.
    ///
    /// Leg labels are integer strings. Reloading may renumber them but preserves
    /// structure, dimensions, sharing, and tensor-indexed SSA paths.
    ///
    /// This is suitable for simplified, sliced, or generated networks.
    pub fn to_json_string(&self) -> String {
        let inputs: Vec<Vec<String>> = self
            .inputs
            .iter()
            .map(|t| t.iter().map(|l| l.to_string()).collect())
            .collect();
        let output: Vec<String> = self.output.iter().map(|l| l.to_string()).collect();
        // Sorted keys make serialized output stable and byte-reproducible.
        let size_dict: std::collections::BTreeMap<String, usize> = self
            .size_dict
            .iter()
            .map(|(&l, &d)| (l.to_string(), d))
            .collect();
        let v = serde_json::json!({
            "name": self.name,
            "inputs": inputs,
            "output": output,
            "size_dict": size_dict,
        });
        serde_json::to_string(&v).expect("TensorNetwork 序列化不应失败")
    }

    /// Writes the network to a file.
    pub fn to_json(&self, path: &Path) -> Result<(), String> {
        std::fs::write(path, self.to_json_string()).map_err(|e| format!("写文件失败: {e}"))
    }

    /// Generates a random d-regular graph network with edge dimension `dim`.
    /// Parallel edges are allowed; self-loops are rejected and resampled.
    pub fn rand_regular<R: Rng>(n: usize, degree: usize, dim: usize, rng: &mut R) -> Self {
        assert!(n * degree % 2 == 0, "n*degree 必须为偶数");
        let mut stubs: Vec<usize> = (0..n)
            .flat_map(|i| std::iter::repeat_n(i, degree))
            .collect();
        // Resample until no stub pair forms a self-loop.
        let edges = loop {
            stubs.shuffle(rng);
            let pairs: Vec<(usize, usize)> = stubs.chunks(2).map(|c| (c[0], c[1])).collect();
            if pairs.iter().all(|&(a, b)| a != b) {
                break pairs;
            }
        };
        let mut inputs = vec![Vec::new(); n];
        let mut size_dict = HashMap::new();
        for (leg, &(a, b)) in edges.iter().enumerate() {
            inputs[a].push(leg as LegId);
            inputs[b].push(leg as LegId);
            size_dict.insert(leg as LegId, dim);
        }
        TensorNetwork {
            name: format!("rand{degree}reg_n{n}"),
            inputs,
            output: vec![],
            size_dict,
        }
    }

    /// Builds an open-boundary 2D grid whose nearest-neighbor legs contract to a scalar.
    pub fn grid_2d(rows: usize, cols: usize, dim: usize) -> Self {
        let mut inputs = vec![Vec::new(); rows * cols];
        let mut size_dict = HashMap::new();
        let mut next_leg: LegId = 0;
        let id = |r: usize, c: usize| r * cols + c;
        for r in 0..rows {
            for c in 0..cols {
                if c + 1 < cols {
                    inputs[id(r, c)].push(next_leg);
                    inputs[id(r, c + 1)].push(next_leg);
                    size_dict.insert(next_leg, dim);
                    next_leg += 1;
                }
                if r + 1 < rows {
                    inputs[id(r, c)].push(next_leg);
                    inputs[id(r + 1, c)].push(next_leg);
                    size_dict.insert(next_leg, dim);
                    next_leg += 1;
                }
            }
        }
        TensorNetwork {
            name: format!("grid_{rows}x{cols}"),
            inputs,
            output: vec![],
            size_dict,
        }
    }

    /// Generates a small connected random network for tests.
    /// A random tree ensures connectivity before extra and open legs are added.
    pub fn random_connected<R: Rng>(
        n: usize,
        extra: usize,
        dims: &[usize],
        n_open: usize,
        rng: &mut R,
    ) -> Self {
        let mut inputs = vec![Vec::new(); n];
        let mut size_dict = HashMap::new();
        let mut next_leg: LegId = 0;
        let mut add_edge = |a: usize,
                            b: usize,
                            inputs: &mut Vec<Vec<LegId>>,
                            size_dict: &mut HashMap<LegId, usize>,
                            rng: &mut R| {
            inputs[a].push(next_leg);
            inputs[b].push(next_leg);
            size_dict.insert(next_leg, dims[rng.gen_range(0..dims.len())]);
            next_leg += 1;
        };
        // Random spanning tree.
        for i in 1..n {
            let j = rng.gen_range(0..i);
            add_edge(i, j, &mut inputs, &mut size_dict, rng);
        }
        for _ in 0..extra {
            let a = rng.gen_range(0..n);
            let mut b = rng.gen_range(0..n);
            while b == a {
                b = rng.gen_range(0..n);
            }
            add_edge(a, b, &mut inputs, &mut size_dict, rng);
        }
        // Open legs.
        let mut output = Vec::new();
        for _ in 0..n_open {
            let t = rng.gen_range(0..n);
            inputs[t].push(next_leg);
            output.push(next_leg);
            size_dict.insert(next_leg, dims[rng.gen_range(0..dims.len())]);
            next_leg += 1;
        }
        TensorNetwork {
            name: format!("rand_conn_n{n}"),
            inputs,
            output,
            size_dict,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::TensorNetwork;

    #[test]
    fn json_loader_accepts_complete_canonical_schema() {
        let json = r#"{
            "name": "canonical",
            "inputs": [["a", "bond"], ["bond", "c"]],
            "output": ["a", "c"],
            "size_dict": {"a": 2, "bond": 3, "c": 5}
        }"#;
        let (net, labels) = TensorNetwork::parse_json_with_labels(json).unwrap();

        assert_eq!(net.name, "canonical");
        assert_eq!(net.inputs, vec![vec![0, 1], vec![1, 2]]);
        assert_eq!(net.output, vec![0, 2]);
        assert_eq!(labels, vec!["a", "bond", "c"]);
        assert_eq!(net.dim(0), 2);
        assert_eq!(net.dim(1), 3);
        assert_eq!(net.dim(2), 5);
    }

    #[test]
    fn json_loader_accepts_complete_omeco_schema() {
        let json = r#"{
            "name": "omeco",
            "ixs": [[7, 9], [9, 11]],
            "iy": [7, 11],
            "sizes": {"7": 2, "9": 3, "11": 5}
        }"#;
        let (net, labels) = TensorNetwork::parse_json_with_labels(json).unwrap();

        assert_eq!(net.name, "omeco");
        assert_eq!(net.inputs, vec![vec![0, 1], vec![1, 2]]);
        assert_eq!(net.output, vec![0, 2]);
        assert_eq!(labels, vec!["7", "9", "11"]);
        assert_eq!(net.dim(0), 2);
        assert_eq!(net.dim(1), 3);
        assert_eq!(net.dim(2), 5);
    }

    #[test]
    fn json_loader_ignores_unrelated_metadata() {
        let json = r#"{
            "inputs": [["a"]],
            "output": ["a"],
            "size_dict": {"a": 2},
            "meta": {
                "kind": "future-network-kind",
                "producer": {"name": "future-tool"},
                "arctn": {"future_option": 7}
            }
        }"#;
        let (_, labels) = TensorNetwork::parse_json_with_labels(json).unwrap();
        assert_eq!(labels, vec!["a"]);
    }

    #[test]
    fn json_loader_rejects_mixed_schema() {
        let json = r#"{
            "inputs": [["a"]],
            "output": ["a"],
            "size_dict": {"a": 2},
            "ixs": [["a"]]
        }"#;
        let err = TensorNetwork::parse_json_with_labels(json)
            .expect_err("两种 schema 出现在同一文件时必须拒绝");
        assert!(err.contains("不能混用"), "{err}");
    }

    #[test]
    fn json_loader_rejects_partial_schema() {
        let canonical = r#"{"inputs": [["a"]], "output": ["a"]}"#;
        let err = TensorNetwork::parse_json_with_labels(canonical)
            .expect_err("缺 size_dict 的 canonical schema 必须拒绝");
        assert!(err.contains("canonical schema 残缺"), "{err}");
        assert!(err.contains("size_dict"), "{err}");

        let omeco = r#"{"ixs": [[7]], "sizes": {"7": 2}}"#;
        let err = TensorNetwork::parse_json_with_labels(omeco)
            .expect_err("缺 iy 的 OMECO schema 必须拒绝");
        assert!(err.contains("OMECO schema 残缺"), "{err}");
        assert!(err.contains("iy"), "{err}");
    }

    #[test]
    fn json_loader_rejects_integer_string_label_collision() {
        let text = r#"{
            "inputs": [[1], ["1"]],
            "output": [],
            "size_dict": {"1": 2}
        }"#;
        let err = TensorNetwork::parse_json_with_labels(text)
            .expect_err("整数 1 与字符串 \"1\" 不能静默合并");
        assert!(err.contains("正规化后冲突"), "{err}");
    }

    #[test]
    fn validate_rejects_missing_zero_or_orphan_legs() {
        let empty = TensorNetwork {
            name: "empty".into(),
            inputs: vec![],
            output: vec![],
            size_dict: Default::default(),
        };
        assert!(empty
            .validate()
            .unwrap_err()
            .contains("至少需要一个输入张量"));

        let missing = TensorNetwork {
            name: "missing".into(),
            inputs: vec![vec![0]],
            output: vec![],
            size_dict: Default::default(),
        };
        assert!(missing.validate().unwrap_err().contains("缺少 size_dict"));

        let zero = TensorNetwork {
            name: "zero".into(),
            inputs: vec![vec![0]],
            output: vec![],
            size_dict: [(0, 0)].into_iter().collect(),
        };
        assert!(zero.validate().unwrap_err().contains("维度为 0"));

        let orphan = TensorNetwork {
            name: "orphan".into(),
            inputs: vec![vec![0]],
            output: vec![1],
            size_dict: [(0, 2), (1, 2)].into_iter().collect(),
        };
        assert!(orphan.validate().unwrap_err().contains("不属于任何输入"));
    }

    #[test]
    fn validate_rejects_dense_shape_overflow_before_execution() {
        let net = TensorNetwork {
            name: "overflow".into(),
            inputs: vec![vec![0, 1]],
            output: vec![],
            size_dict: [(0, isize::MAX as usize), (1, 3)].into_iter().collect(),
        };
        assert!(net.validate().unwrap_err().contains("元素数溢出"));
    }

    #[test]
    fn validate_rejects_unused_sparse_dimension_entry() {
        let net = TensorNetwork {
            name: "unused-sparse-dimension".into(),
            inputs: vec![vec![0]],
            output: vec![0],
            size_dict: [(0, 2), (u32::MAX, 2)].into_iter().collect(),
        };
        let err = net
            .validate()
            .expect_err("未被任何张量使用的巨大 key 必须明确拒绝");
        assert!(err.contains("未被任何输入张量使用"), "{err}");
    }
}
