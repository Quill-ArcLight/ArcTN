//! ArcTN tensor-network contraction library.
//!
//! - [`tensor`]: dense real and complex tensors
//! - [`network`]: tensor-network structure and serialization
//! - [`path`]: SSA paths and cost metrics
//! - [`objective`]: planning objectives
//! - [`paths`]: pathfinding algorithms
//! - [`contract`]: path execution
//! - [`naive`]: reference einsum implementation for tests

pub mod auto;
pub mod compiled;
pub mod contract;
pub mod execution_plan;
pub mod linalg;
pub mod naive;
pub mod network;
pub mod objective;
pub mod path;
pub mod pathcache;
pub mod paths;
pub mod simplify;
pub mod slice;
pub mod tensor;
pub mod tree;

pub use auto::{
    auto_path_preset, auto_path_preset_to_size, auto_path_preset_to_size_with_mode,
    auto_path_preset_to_size_with_mode_and_objective, auto_path_preset_to_size_with_objective,
    auto_path_preset_with_objective, AutoPreset, SlicingMode,
};
pub use compiled::CompiledContraction;
pub use contract::contract_network;
pub use execution_plan::{
    complete_execution_plan_v2, parse_embedded_execution_plan, parse_execution_plan_for_network,
    EmbeddedExecutionPlan, ExecutionPlanNetwork, LoadedExecutionPlan,
};
pub use naive::naive_einsum;
pub use network::{LegId, TensorNetwork};
pub use objective::PlannerObjective;
pub use objective::{FLOPS_WEIGHT, READ_WRITE_WEIGHT};
pub use path::{simulate_path, PathStats, SsaPath};
pub use pathcache::{CacheHit, PathCache};
pub use paths::greedy::{greedy, random_greedy};
pub use paths::optimal::optimal_dp;
pub use simplify::{random_greedy_simplified, simplify, stitch, Simplified};
pub use slice::{
    contract_network_sliced, find_slices, find_slices_to_size, path_fits_target_size,
    slice_and_reconf_to_size, slice_result_fits_target_size, slice_temper_to_size, SliceResult,
};
pub use tensor::{checked_numel, DenseTensor, Scalar};
pub use tree::{anneal_path, anneal_paths, reconfigure_path, temper_path, temper_paths, CTree};
