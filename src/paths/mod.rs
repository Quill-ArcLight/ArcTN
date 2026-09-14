pub mod bisect;
pub mod budgeted;
pub mod greedy;
mod greedy_slicing;
pub mod optimal;
pub mod ordertree;
pub use greedy_slicing::{
    greedy_path_with_slicing, greedy_path_with_slicing_to_size, greedy_path_with_slicing_until,
};
