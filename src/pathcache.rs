//! Tensor-network path cache.
//!
//! Keys include network structure and planner configuration but not network
//! names. Disk filenames use FNV-1a, while hits still compare the full key and
//! validate the path. Writes use a same-directory temporary file and atomic
//! rename. Cache directories must be trusted because file access follows symlinks.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::network::TensorNetwork;
use crate::path::{simulate_path, PathStats, SsaPath};

#[derive(Clone, Debug, Serialize, Deserialize)]
struct CacheEntry {
    /// Full canonical key compared before accepting a hit.
    canon: String,
    path: SsaPath,
    /// Archived reference value; hits are rescored against the current network.
    /// Non-finite values serialize as null and deserialize as NaN.
    #[serde(deserialize_with = "f64_or_null")]
    log10_flops: f64,
    #[serde(deserialize_with = "f64_or_null")]
    log2_peak_size: f64,
    /// Source method label.
    source: String,
    /// ArcTN version at write time; excluded from the key.
    arctn_version: String,
}

/// Cache hit with a path and freshly recomputed statistics.
#[derive(Clone, Debug)]
pub struct CacheHit {
    pub path: SsaPath,
    pub stats: PathStats,
    pub source: String,
    pub from_disk: bool,
}

/// Path cache with an in-memory map and an optional one-file-per-key disk layer.
pub struct PathCache {
    dir: Option<PathBuf>,
    mem: HashMap<u64, CacheEntry>,
}

impl PathCache {
    /// Creates a process-local in-memory cache.
    pub fn in_memory() -> Self {
        PathCache {
            dir: None,
            mem: HashMap::new(),
        }
    }

    /// Creates a persistent cache, creating the directory if needed.
    pub fn on_disk(dir: impl Into<PathBuf>) -> Result<Self, String> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir).map_err(|e| format!("创建缓存目录失败: {e}"))?;
        Ok(PathCache {
            dir: Some(dir),
            mem: HashMap::new(),
        })
    }

    /// Returns the stable network canonical form without name or planner configuration.
    ///
    /// Tensor and leg order are preserved; no graph-isomorphism normalization is
    /// attempted. Consumers must compare this full string, not only its hash.
    pub fn network_canon(net: &TensorNetwork) -> String {
        let mut s = String::with_capacity(net.size_dict.len() * 8 + 32);
        s.push_str("t:");
        for (i, legs) in net.inputs.iter().enumerate() {
            if i > 0 {
                s.push(';');
            }
            for (j, l) in legs.iter().enumerate() {
                if j > 0 {
                    s.push(',');
                }
                let _ = write!(s, "{l}");
            }
        }
        s.push_str("|o:");
        for (j, l) in net.output.iter().enumerate() {
            if j > 0 {
                s.push(',');
            }
            let _ = write!(s, "{l}");
        }
        s.push_str("|d:");
        let mut dims: Vec<(u32, usize)> = net.size_dict.iter().map(|(&l, &d)| (l, d)).collect();
        dims.sort_unstable();
        for (j, (l, d)) in dims.iter().enumerate() {
            if j > 0 {
                s.push(',');
            }
            let _ = write!(s, "{l}={d}");
        }
        s
    }

    /// Returns the canonical key for a network and planner configuration.
    pub fn canon_key(net: &TensorNetwork, cfg: &str) -> String {
        let mut s = Self::network_canon(net);
        s.reserve(cfg.len() + 5);
        s.push_str("|cfg:");
        s.push_str(cfg);
        s
    }

    /// Looks up an entry after exact key comparison and live path simulation.
    /// Collisions, corrupt files, and invalid paths are treated as misses.
    pub fn lookup(&mut self, net: &TensorNetwork, cfg: &str) -> Option<CacheHit> {
        let canon = Self::canon_key(net, cfg);
        let h = fnv1a64(&canon);
        // Check memory first.
        if let Some(e) = self.mem.get(&h) {
            if e.canon == canon {
                if let Ok(stats) = simulate_path(net, &e.path) {
                    return Some(CacheHit {
                        path: e.path.clone(),
                        stats,
                        source: e.source.clone(),
                        from_disk: false,
                    });
                }
            }
            return None;
        }
        // Fall back to disk.
        let dir = self.dir.as_ref()?;
        let text = std::fs::read_to_string(entry_file(dir, h)).ok()?;
        let e: CacheEntry = serde_json::from_str(&text).ok()?;
        if e.canon != canon {
            return None;
        }
        let stats = simulate_path(net, &e.path).ok()?;
        self.mem.insert(h, e.clone());
        Some(CacheHit {
            path: e.path,
            stats,
            source: e.source,
            from_disk: true,
        })
    }

    /// Validates and stores an entry. Disk failure does not discard the memory entry.
    pub fn store(
        &mut self,
        net: &TensorNetwork,
        cfg: &str,
        path: &SsaPath,
        source: &str,
    ) -> Result<PathStats, String> {
        let stats = simulate_path(net, path).map_err(|e| format!("拒绝入库（路径非法）: {e}"))?;
        let canon = Self::canon_key(net, cfg);
        let h = fnv1a64(&canon);
        let e = CacheEntry {
            canon,
            path: path.clone(),
            log10_flops: stats.log10_flops,
            log2_peak_size: stats.log2_peak_size,
            source: source.to_string(),
            arctn_version: env!("CARGO_PKG_VERSION").to_string(),
        };
        if let Some(dir) = &self.dir {
            match serde_json::to_string(&e) {
                Ok(json) => {
                    // PID and a process-local sequence avoid temporary-name collisions.
                    static TMP_SEQ: std::sync::atomic::AtomicU64 =
                        std::sync::atomic::AtomicU64::new(0);
                    let seq = TMP_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let tmp = dir.join(format!("{h:016x}.tmp.{}.{seq}", std::process::id()));
                    let dst = entry_file(dir, h);
                    if let Err(io) =
                        std::fs::write(&tmp, json).and_then(|_| std::fs::rename(&tmp, &dst))
                    {
                        eprintln!("路径缓存落盘失败（本条仅驻内存）: {io}");
                        let _ = std::fs::remove_file(&tmp);
                    }
                }
                Err(x) => eprintln!("路径缓存序列化失败（本条仅驻内存）: {x}"),
            }
        }
        self.mem.insert(h, e);
        Ok(stats)
    }

    /// Returns a cached result or validates and stores the path produced by `f`.
    /// Invalid networks and paths return an error without populating the cache.
    pub fn get_or_compute<F>(
        &mut self,
        net: &TensorNetwork,
        cfg: &str,
        f: F,
    ) -> Result<(SsaPath, PathStats, String, bool), String>
    where
        F: FnOnce() -> (SsaPath, String),
    {
        if let Some(hit) = self.lookup(net, cfg) {
            return Ok((hit.path, hit.stats, hit.source, true));
        }
        let (path, source) = f();
        let stats = self.store(net, cfg, &path, &source)?;
        Ok((path, stats, source, false))
    }

    /// Returns the disk directory, or `None` for an in-memory cache.
    pub fn dir(&self) -> Option<&Path> {
        self.dir.as_deref()
    }
}

fn entry_file(dir: &Path, h: u64) -> PathBuf {
    dir.join(format!("{h:016x}.json"))
}

/// Restores archived JSON null as NaN; this field does not determine cache hits.
fn f64_or_null<'de, D>(d: D) -> Result<f64, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Ok(Option::<f64>::deserialize(d)?.unwrap_or(f64::NAN))
}

/// Stable 64-bit FNV-1a hash used only for filenames.
pub(crate) fn fnv1a64(s: &str) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in s.as_bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paths::greedy::greedy;

    /// Chain T0(0,1), T1(1,2), T2(2,3) with open legs 0 and 3.
    fn toy(name: &str, dim: usize) -> TensorNetwork {
        let mut size_dict = HashMap::new();
        for l in 0..4u32 {
            size_dict.insert(l, dim);
        }
        TensorNetwork {
            name: name.to_string(),
            inputs: vec![vec![0, 1], vec![1, 2], vec![2, 3]],
            output: vec![0, 3],
            size_dict,
        }
    }

    fn tmpdir(tag: &str) -> PathBuf {
        let d =
            std::env::temp_dir().join(format!("arctn_pathcache_test_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    #[test]
    fn 内存层_未命中后命中_路径与评分一致() {
        let net = toy("a", 4);
        let mut c = PathCache::in_memory();
        let (p1, s1, src1, hit1) = c
            .get_or_compute(&net, "greedy|seed=0", || {
                (greedy(&net).unwrap().0, "greedy".into())
            })
            .unwrap();
        assert!(!hit1);
        assert_eq!(src1, "greedy");
        let (p2, s2, _, hit2) = c
            .get_or_compute(&net, "greedy|seed=0", || panic!("命中时不得现算"))
            .unwrap();
        assert!(hit2);
        assert_eq!(p1, p2);
        assert_eq!(s1.log10_flops, s2.log10_flops);
        assert_eq!(s1.log2_max_size, s2.log2_max_size);
        assert_eq!(s1.log2_max_contraction_size, s2.log2_max_contraction_size);
        assert_eq!(s1.log2_total_size, s2.log2_total_size);
        assert_eq!(s1.log2_peak_size, s2.log2_peak_size);
    }

    #[test]
    fn 键_不含名字_含维度与配置() {
        let net = toy("a", 4);
        let mut c = PathCache::in_memory();
        let p = greedy(&net).unwrap().0;
        c.store(&net, "cfg1", &p, "greedy").unwrap();
        // Network names do not affect the key.
        let renamed = toy("b", 4);
        assert!(c.lookup(&renamed, "cfg1").is_some());
        // Dimension changes invalidate the key.
        assert!(c.lookup(&toy("a", 8), "cfg1").is_none());
        // Configuration changes invalidate the key.
        assert!(c.lookup(&net, "cfg2").is_none());
    }

    #[test]
    fn 磁盘层_跨实例命中_坏文件与错网降级() {
        let dir = tmpdir("disk");
        let net = toy("a", 4);
        let p = greedy(&net).unwrap().0;
        {
            let mut c = PathCache::on_disk(&dir).unwrap();
            c.store(&net, "cfg", &p, "greedy").unwrap();
        }
        // A new instance loads the entry from disk.
        let mut c2 = PathCache::on_disk(&dir).unwrap();
        let hit = c2.lookup(&net, "cfg").expect("应从磁盘命中");
        assert!(hit.from_disk);
        assert_eq!(hit.path, p);
        // Corrupt files become misses and are recomputed.
        let h = fnv1a64(&PathCache::canon_key(&net, "cfg"));
        std::fs::write(entry_file(&dir, h), "not json").unwrap();
        let mut c3 = PathCache::on_disk(&dir).unwrap();
        assert!(c3.lookup(&net, "cfg").is_none());
        let (_, _, _, hit3) = c3
            .get_or_compute(&net, "cfg", || (greedy(&net).unwrap().0, "greedy".into()))
            .unwrap();
        assert!(!hit3);
        assert!(c3.lookup(&net, "cfg").is_some(), "自愈后应重新命中");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn 碰撞与失效防护_canon不符或路径非法都算未命中() {
        let dir = tmpdir("guard");
        let net = toy("a", 4);
        let canon = PathCache::canon_key(&net, "cfg");
        let h = fnv1a64(&canon);
        std::fs::create_dir_all(&dir).unwrap();
        // A colliding filename with a different canonical key must miss.
        let alien = CacheEntry {
            canon: "别的网的规范串".to_string(),
            path: greedy(&net).unwrap().0,
            log10_flops: 0.0,
            log2_peak_size: 0.0,
            source: "x".into(),
            arctn_version: "0".into(),
        };
        std::fs::write(entry_file(&dir, h), serde_json::to_string(&alien).unwrap()).unwrap();
        let mut c = PathCache::on_disk(&dir).unwrap();
        assert!(c.lookup(&net, "cfg").is_none());
        // A path incompatible with the current network must miss.
        let broken = CacheEntry {
            canon,
            path: vec![(97, 98)],
            ..alien
        };
        std::fs::write(entry_file(&dir, h), serde_json::to_string(&broken).unwrap()).unwrap();
        let mut c2 = PathCache::on_disk(&dir).unwrap();
        assert!(c2.lookup(&net, "cfg").is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn 单张量网_非有限flops_磁盘往返仍命中() {
        // Non-finite costs round-trip as null without invalidating an empty path.
        let dir = tmpdir("nullf64");
        let mut size_dict = HashMap::new();
        size_dict.insert(0u32, 2);
        size_dict.insert(1u32, 2);
        let net = TensorNetwork {
            name: "标量样".into(),
            inputs: vec![vec![0, 1]],
            output: vec![0, 1],
            size_dict,
        };
        {
            let mut c = PathCache::on_disk(&dir).unwrap();
            c.store(&net, "cfg", &vec![], "import").unwrap(); // A single tensor has an empty path and zero FLOPs.
        }
        let text = std::fs::read_to_string(entry_file(
            &dir,
            fnv1a64(&PathCache::canon_key(&net, "cfg")),
        ))
        .unwrap();
        assert!(
            text.contains("null"),
            "前提校验：非有限 f64 确实落盘为 null"
        );
        let mut c2 = PathCache::on_disk(&dir).unwrap();
        let hit = c2.lookup(&net, "cfg").expect("null 字段不得毒死整条记录");
        assert!(hit.from_disk);
        assert_eq!(hit.path, Vec::<(usize, usize)>::new());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn invalid_network_returns_error_without_caching() {
        let mut size_dict = HashMap::new();
        for l in 0..3u32 {
            size_dict.insert(l, 2);
        }
        let net = TensorNetwork {
            name: "怪网".into(),
            inputs: vec![vec![0, 1], vec![1, 2]],
            output: vec![0, 2, 2],
            size_dict,
        };
        let mut c = PathCache::in_memory();
        assert!(c
            .get_or_compute(&net, "cfg", || (vec![(0, 1)], "import".into()))
            .is_err());
        assert!(c.lookup(&net, "cfg").is_none(), "不支持的网不得入库");
    }

    #[test]
    fn 拒绝入库非法路径() {
        let net = toy("a", 4);
        let mut c = PathCache::in_memory();
        assert!(c.store(&net, "cfg", &vec![(97, 98)], "x").is_err());
        assert!(c
            .get_or_compute(&net, "cfg", || (vec![(97, 98)], "x".into()))
            .is_err());
        assert!(c.lookup(&net, "cfg").is_none());
    }
}
