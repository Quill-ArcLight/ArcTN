//! Execute a saved contraction path and its slices across MPI ranks.
//!
//! Rank zero loads the execution plan; ranks evaluate disjoint slices and sum
//! their outputs with Allreduce. Path optimization happens before this program.

use arctn::{
    contract_network, contract_network_sliced, parse_embedded_execution_plan,
    parse_execution_plan_for_network, simulate_path, DenseTensor, ExecutionPlanNetwork, LegId,
    Scalar, SsaPath, TensorNetwork,
};
use mpi::collective::SystemOperation;
use mpi::datatype::Equivalence;
use mpi::topology::SimpleCommunicator;
use mpi::traits::*;
use num_complex::{Complex32, Complex64};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Dtype {
    F32,
    F64,
    Complex64,
    Complex128,
}

impl Dtype {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "f32" => Ok(Self::F32),
            "f64" => Ok(Self::F64),
            "complex64" => Ok(Self::Complex64),
            "complex128" => Ok(Self::Complex128),
            _ => Err("--dtype must be f32, f64, complex64 or complex128".into()),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::Complex64 => "complex64",
            Self::Complex128 => "complex128",
        }
    }

    fn check_tolerance(self) -> f64 {
        match self {
            Self::F32 | Self::Complex64 => 1e-4,
            Self::F64 | Self::Complex128 => 1e-6,
        }
    }
}

struct Args {
    load_path: String,
    net_file: Option<String>,
    data_file: Option<String>,
    dtype: Dtype,
    seed: u64,
    check: bool,
}

fn print_help() {
    println!(
        "tnmpi {}\n\
Usage: tnmpi --load-path <PLAN.json> [OPTIONS]\n\n\
Execute the saved SSA path and sliced indices; no path search is performed.\n\
  --net <NETWORK.json> Optional matching network; required for version 1 plans\n\
  --data <tensors.bin> Row-major little-endian tensors in network input order\n\
  --dtype <TYPE>       f32, f64, complex64 or complex128 (default f64)\n\
  --seed <N>           Seed for example inputs when --data is omitted (default 1)\n\
  --check              Compare a small result with single-process execution\n\
  --help               Show this help\n\n\
Version 2 plans contain the network and are written by arctn_plan().save() or\n\
tnpath --save-path. Every rank must access identical input tensor data.\n\
Complex elements store the real component followed by the imaginary component.\n\
Execution time includes the final MPI Allreduce, but not loading or --check.\n\
Use the MPI launcher or job scheduler for CPU binding and time limits.",
        env!("CARGO_PKG_VERSION")
    );
}

fn parse_args() -> Result<Option<Args>, String> {
    let mut args = Args {
        load_path: String::new(),
        net_file: None,
        data_file: None,
        dtype: Dtype::F64,
        seed: 1,
        check: false,
    };
    let mut argv = std::env::args().skip(1);
    while let Some(flag) = argv.next() {
        match flag.as_str() {
            "--help" | "-h" => return Ok(None),
            "--check" => args.check = true,
            "--load-path" | "--net" | "--data" | "--seed" | "--dtype" => {
                let value = argv
                    .next()
                    .filter(|v| !v.starts_with("--"))
                    .ok_or_else(|| format!("{flag} requires a value"))?;
                match flag.as_str() {
                    "--load-path" => args.load_path = value,
                    "--net" => args.net_file = Some(value),
                    "--data" => args.data_file = Some(value),
                    "--dtype" => args.dtype = Dtype::parse(&value)?,
                    "--seed" => {
                        args.seed = value
                            .parse()
                            .map_err(|_| "--seed requires an unsigned 64-bit integer".to_string())?
                    }
                    _ => unreachable!(),
                }
            }
            _ => return Err(format!("unknown option: {flag}; use --help")),
        }
    }
    if args.load_path.is_empty() {
        return Err("--load-path <PLAN.json> is required".into());
    }
    Ok(Some(args))
}

/// In-memory broadcast payload. Saved files use the existing execution-plan schema.
#[derive(Debug, Serialize, Deserialize)]
struct ExecutionInput {
    network_name: String,
    network: ExecutionPlanNetwork,
    path: SsaPath,
    sliced: Vec<LegId>,
    target_size: Option<usize>,
    target_log2: Option<f64>,
}

fn load_execution_input(args: &Args) -> Result<ExecutionInput, String> {
    let text = std::fs::read_to_string(&args.load_path)
        .map_err(|e| format!("cannot read {}: {e}", args.load_path))?;
    let (network, execution) = if let Some(file) = &args.net_file {
        let (net, labels) = TensorNetwork::load_json_with_labels(Path::new(file))?;
        let execution = parse_execution_plan_for_network(&text, &net, &labels, false)?;
        (net, execution)
    } else {
        let loaded = parse_embedded_execution_plan(&text)?;
        (loaded.network, loaded.execution)
    };
    Ok(ExecutionInput {
        network_name: network.name.clone(),
        network: ExecutionPlanNetwork::from_network(&network),
        path: execution.path,
        sliced: execution
            .sliced
            .ok_or("saved plan must specify sliced.legs")?,
        target_size: execution.declared_target_size,
        target_log2: execution.declared_target_log2,
    })
}

/// Coordinate recoverable errors before any rank enters the next collective.
fn require_all<T>(world: &SimpleCommunicator, value: Result<T, String>) -> T {
    let failed = i32::from(value.is_err());
    let mut any_failed = 0;
    world.all_reduce_into(&failed, &mut any_failed, SystemOperation::max());
    if any_failed != 0 {
        if let Err(error) = &value {
            eprintln!("tnmpi rank {}: {error}", world.rank());
        }
        world.abort(3);
    }
    value.unwrap_or_else(|_| unreachable!())
}

fn broadcast_input(world: &SimpleCommunicator, args: &Args) -> ExecutionInput {
    let mut bytes = if world.rank() == 0 {
        serde_json::to_vec(&load_execution_input(args)).expect("serialize execution input")
    } else {
        Vec::new()
    };
    let root = world.process_at_rank(0);
    let mut len = bytes.len() as u64;
    root.broadcast_into(&mut len);
    let len = require_all(
        world,
        usize::try_from(len).map_err(|_| "plan size overflows usize".into()),
    );
    bytes.resize(len, 0);
    for chunk in bytes.chunks_mut(i32::MAX as usize) {
        root.broadcast_into(chunk);
    }
    let decoded = serde_json::from_slice::<Result<ExecutionInput, String>>(&bytes)
        .map_err(|e| format!("invalid broadcast plan: {e}"))
        .and_then(|value| value);
    require_all(world, decoded)
}

fn checked_product(dims: impl IntoIterator<Item = usize>) -> Result<usize, String> {
    dims.into_iter().try_fold(1usize, |total, dim| {
        total
            .checked_mul(dim)
            .ok_or_else(|| "element count overflows usize".into())
    })
}

fn sliced_network(net: &TensorNetwork, sliced: &[LegId]) -> Result<TensorNetwork, String> {
    arctn::slice::validate_slice_legs(net, sliced)?;
    let mut sub = net.clone();
    for &leg in sliced {
        for indices in &mut sub.inputs {
            indices.retain(|&index| index != leg);
        }
        sub.size_dict.remove(&leg);
    }
    Ok(sub)
}

fn load_tensors<T: Scalar>(
    args: &Args,
    net: &TensorNetwork,
) -> Result<Vec<DenseTensor<T>>, String> {
    let shapes: Vec<Vec<usize>> = net
        .inputs
        .iter()
        .map(|indices| indices.iter().map(|&i| net.dim(i)).collect())
        .collect();
    let counts = shapes
        .iter()
        .map(|s| checked_product(s.iter().copied()))
        .collect::<Result<Vec<_>, _>>()?;
    if let Some(file) = &args.data_file {
        let total = counts
            .iter()
            .try_fold(0usize, |n, &count| n.checked_add(count))
            .and_then(|n| n.checked_mul(T::NBYTES))
            .ok_or("input byte count overflows usize")?;
        let bytes = std::fs::read(file).map_err(|e| format!("cannot read {file}: {e}"))?;
        if bytes.len() != total {
            return Err(format!(
                "input has {} bytes, expected {total} (row-major {} little-endian)",
                bytes.len(),
                args.dtype.name()
            ));
        }
        let mut offset = 0;
        shapes
            .into_iter()
            .zip(counts)
            .map(|(shape, count)| {
                let end = offset + count * T::NBYTES;
                let data = bytes[offset..end]
                    .chunks_exact(T::NBYTES)
                    .map(T::from_le_bytes)
                    .collect();
                offset = end;
                DenseTensor::try_from_data(shape, data)
            })
            .collect()
    } else {
        Ok(shapes
            .into_iter()
            .enumerate()
            .map(|(index, shape)| {
                let mut rng = ChaCha8Rng::seed_from_u64(args.seed ^ 0x5eed_0000 ^ index as u64);
                DenseTensor::random(shape, &mut rng)
            })
            .collect())
    }
}

/// Assign consecutive slices, including empty ranges when ranks outnumber slices.
fn slice_range(count: usize, ranks: usize, rank: usize) -> (usize, usize) {
    let chunk = count / ranks;
    let extra = count % ranks;
    let begin = rank * chunk + rank.min(extra);
    (begin, begin + chunk + usize::from(rank < extra))
}

/// Contract and accumulate slices in linear range `[begin, end)`.
/// This mirrors `contract_network_sliced`; the last sliced leg varies fastest.
fn contract_slice_range<T: Scalar>(
    net: &TensorNetwork,
    tensors: &[DenseTensor<T>],
    path: &SsaPath,
    sliced: &[LegId],
    begin: usize,
    end: usize,
) -> Result<DenseTensor<T>, String> {
    arctn::slice::validate_sliced_contraction_inputs(net, tensors, sliced)?;
    let sliced_pos: HashMap<LegId, usize> = sliced
        .iter()
        .copied()
        .enumerate()
        .map(|(index, leg)| (leg, index))
        .collect();
    // Each assignment uses the same reduced network and saved path.
    let sub = sliced_network(net, sliced)?;
    let dims: Vec<usize> = sliced.iter().map(|&l| net.dim(l)).collect();
    let n_slices = checked_product(dims.iter().copied())?;
    if begin >= end || end > n_slices {
        return Err(format!(
            "切片范围 [{begin}, {end}) 超出有效范围 [0, {n_slices})"
        ));
    }
    let mut result: Option<DenseTensor<T>> = None;
    for lin in begin..end {
        // Decode the linear id as mixed-radix indices, last dimension fastest.
        let mut idx = vec![0usize; dims.len()];
        let mut rem = lin;
        for d in (0..dims.len()).rev() {
            idx[d] = rem % dims[d];
            rem /= dims[d];
        }
        let sliced_tensors: Vec<DenseTensor<T>> = net
            .inputs
            .iter()
            .zip(tensors)
            .map(|(legs, t)| {
                let mut t = t.clone();
                // Remove axes from the end so earlier positions remain stable.
                for ax in (0..legs.len()).rev() {
                    if let Some(&k) = sliced_pos.get(&legs[ax]) {
                        t = t.select_axis(ax, idx[k]);
                    }
                }
                t
            })
            .collect();
        let part = contract_network(&sub, sliced_tensors, path)?;
        match &mut result {
            None => result = Some(part),
            Some(acc) => {
                for (a, b) in acc.data_mut().iter_mut().zip(part.data()) {
                    *a += *b;
                }
            }
        }
    }
    result.ok_or_else(|| "空切片范围".to_string())
}

fn check_result<T: Scalar>(
    net: &TensorNetwork,
    tensors: &[DenseTensor<T>],
    input: &ExecutionInput,
    total: &DenseTensor<T>,
    n_slices: usize,
    tolerance: f64,
) -> Result<String, String> {
    if total.numel() > 1 << 22 || n_slices > 1 << 12 {
        return Ok("skipped: reference exceeds size limit".into());
    }
    let reference = contract_network_sliced(net, tensors, &input.path, &input.sliced)?;
    let diff = reference.max_abs_diff(total);
    let scale = reference.data().iter().fold(1f64, |m, x| m.max(x.abs()));
    if !reference.data().iter().all(|x| x.abs().is_finite())
        || !total.data().iter().all(|x| x.abs().is_finite())
        || !diff.is_finite()
        || diff > tolerance * scale
    {
        return Err(format!(
            "single-process comparison failed: max_abs_diff={diff}, scale={scale}"
        ));
    }
    Ok(format!("PASS(max_abs_diff={diff:.3e})"))
}

fn run<T: Scalar + Equivalence>(args: Args) {
    let universe = mpi::initialize().expect("MPI initialization");
    let world = universe.world();
    let rank = world.rank() as usize;
    let ranks = world.size() as usize;
    // A user-specified Rayon pool size is honored; default to one thread per rank.
    if std::env::var_os("RAYON_NUM_THREADS").is_none() {
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build_global();
    }
    let load_start = Instant::now();
    let input = broadcast_input(&world, &args);
    let net = require_all(&world, input.network.to_network(input.network_name.clone()));
    let sub = require_all(&world, sliced_network(&net, &input.sliced));
    let per_slice = require_all(&world, simulate_path(&sub, &input.path));
    let n_slices = require_all(
        &world,
        checked_product(input.sliced.iter().map(|&i| net.dim(i))),
    );
    let tensors = require_all(&world, load_tensors::<T>(&args, &net));
    require_all(
        &world,
        arctn::slice::validate_sliced_contraction_inputs(&net, &tensors, &input.sliced),
    );
    let out_shape: Vec<_> = net.output.iter().map(|&i| net.dim(i)).collect();
    let (begin, end) = slice_range(n_slices, ranks, rank);
    let load_seconds = load_start.elapsed().as_secs_f64();

    world.barrier();
    let exec_start = Instant::now();
    let local = require_all(
        &world,
        if begin == end {
            DenseTensor::try_zeros(out_shape.clone())
        } else {
            contract_slice_range(&net, &tensors, &input.path, &input.sliced, begin, end)
        },
    );
    let mut output = vec![T::zero(); local.numel()];
    for (local_chunk, global_chunk) in local
        .data()
        .chunks(i32::MAX as usize)
        .zip(output.chunks_mut(i32::MAX as usize))
    {
        world.all_reduce_into(local_chunk, global_chunk, SystemOperation::sum());
    }
    let exec_seconds = exec_start.elapsed().as_secs_f64();
    let total = require_all(&world, DenseTensor::try_from_data(out_shape, output));
    let check = require_all(
        &world,
        if args.check && rank == 0 {
            check_result(
                &net,
                &tensors,
                &input,
                &total,
                n_slices,
                args.dtype.check_tolerance(),
            )
        } else {
            Ok("not requested".into())
        },
    );
    let mut load_max = 0f64;
    let mut exec_max = 0f64;
    world.all_reduce_into(&load_seconds, &mut load_max, SystemOperation::max());
    world.all_reduce_into(&exec_seconds, &mut exec_max, SystemOperation::max());
    if rank == 0 {
        println!(
            "{}",
            serde_json::json!({
                "ranks": ranks,
                "network": net.name,
                "mode": "saved-path",
                "dtype": args.dtype.name(),
                "sliced_indices": input.sliced,
                "n_slices": n_slices,
                "target_size": input.target_size,
                "target_log2_size": input.target_log2,
                "per_slice_log10_flops": per_slice.log10_flops,
                "per_slice_log2_max_size": per_slice.log2_max_size,
                "per_slice_log2_peak_size": per_slice.log2_peak_size,
                "per_slice_log2_max_contraction_size": per_slice.log2_max_contraction_size,
                "output_shape": total.shape(),
                "load_wall_seconds_max": load_max,
                "execution_wall_seconds_max": exec_max,
                "check": check,
            })
        );
    }
}

fn main() {
    match parse_args() {
        Ok(Some(args)) => match args.dtype {
            Dtype::F32 => run::<f32>(args),
            Dtype::F64 => run::<f64>(args),
            Dtype::Complex64 => run::<Complex32>(args),
            Dtype::Complex128 => run::<Complex64>(args),
        },
        Ok(None) => print_help(),
        Err(error) => {
            eprintln!("tnmpi: {error}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const REAL_PARTS: [f64; 8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    const IMAGINARY_PARTS: [f64; 8] = [1.0, -1.0, 2.0, 0.0, 0.0, 1.0, -1.0, 2.0];

    fn matrix_network() -> TensorNetwork {
        TensorNetwork {
            name: "mpi-matrix-product".into(),
            inputs: vec![vec![0, 1], vec![1, 2]],
            output: vec![0, 2],
            size_dict: [(0, 2), (1, 2), (2, 2)].into_iter().collect(),
        }
    }

    fn load_matrix_bytes<T: Scalar>(
        dtype: Dtype,
        bytes: &[u8],
    ) -> Result<Vec<DenseTensor<T>>, String> {
        use std::io::Write;
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let file = std::env::temp_dir().join(format!(
            "arctn-tnmpi-{}-{}-{nanos}.bin",
            dtype.name(),
            std::process::id()
        ));
        let mut handle = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&file)
            .unwrap();
        handle.write_all(bytes).unwrap();
        drop(handle);
        let args = Args {
            load_path: String::new(),
            net_file: None,
            data_file: Some(file.to_str().unwrap().into()),
            dtype,
            seed: 1,
            check: false,
        };
        let result = load_tensors(&args, &matrix_network());
        std::fs::remove_file(file).unwrap();
        result
    }

    fn verify_matrix<T: Scalar + PartialEq>(
        dtype: Dtype,
        values: Vec<T>,
        expected: Vec<T>,
        encode: impl Fn(T) -> Vec<u8>,
        imaginary_unit: Option<T>,
    ) {
        let bytes = values.iter().copied().flat_map(encode).collect::<Vec<_>>();
        let tensors = load_matrix_bytes::<T>(dtype, &bytes).unwrap();
        assert_eq!(tensors.len(), 2);
        for (tensor, expected_input) in tensors.iter().zip(values.chunks_exact(4)) {
            assert_eq!(tensor.shape(), &[2, 2]);
            assert_eq!(tensor.data(), expected_input);
        }
        let net = matrix_network();
        let path = vec![(0, 1)];
        for sliced in [vec![], vec![1]] {
            let n_slices = if sliced.is_empty() { 1 } else { 2 };
            let mut total =
                contract_slice_range(&net, &tensors, &path, &sliced, 0, n_slices).unwrap();
            assert_eq!(total.shape(), &[2, 2]);
            // These expected products are calculated independently of the executor.
            assert_eq!(total.data(), expected.as_slice());
            let input = ExecutionInput {
                network_name: net.name.clone(),
                network: ExecutionPlanNetwork::from_network(&net),
                path: path.clone(),
                sliced,
                target_size: None,
                target_log2: None,
            };
            let tolerance = dtype.check_tolerance();
            assert!(
                check_result(&net, &tensors, &input, &total, n_slices, tolerance)
                    .unwrap()
                    .starts_with("PASS")
            );
            if let Some(delta) = imaginary_unit {
                total.data_mut()[0] += delta;
                assert!(
                    check_result(&net, &tensors, &input, &total, n_slices, tolerance)
                        .unwrap_err()
                        .contains("single-process comparison failed")
                );
            }
        }
        let first = contract_slice_range(&net, &tensors, &path, &[1], 0, 1).unwrap();
        let second = contract_slice_range(&net, &tensors, &path, &[1], 1, 2).unwrap();
        let combined: Vec<T> = first
            .data()
            .iter()
            .zip(second.data())
            .map(|(&a, &b)| a + b)
            .collect();
        assert_eq!(combined, expected);
    }

    #[test]
    fn f32_little_endian_inputs_and_slice_products() {
        verify_matrix(
            Dtype::F32,
            REAL_PARTS.map(|x| x as f32).to_vec(),
            vec![19.0f32, 22.0, 43.0, 50.0],
            |x| x.to_le_bytes().to_vec(),
            None,
        );
    }

    #[test]
    fn f64_little_endian_inputs_and_slice_products() {
        verify_matrix(
            Dtype::F64,
            REAL_PARTS.to_vec(),
            vec![19.0f64, 22.0, 43.0, 50.0],
            |x| x.to_le_bytes().to_vec(),
            None,
        );
    }

    #[test]
    fn complex64_inputs_slice_products_and_imaginary_error_check() {
        verify_matrix(
            Dtype::Complex64,
            REAL_PARTS
                .into_iter()
                .zip(IMAGINARY_PARTS)
                .map(|(re, im)| Complex32::new(re as f32, im as f32))
                .collect(),
            [(18.0, -4.0), (23.0, 3.0), (43.0, 6.0), (48.0, 23.0)]
                .map(|(re, im)| Complex32::new(re, im))
                .to_vec(),
            |x| {
                x.re.to_le_bytes()
                    .into_iter()
                    .chain(x.im.to_le_bytes())
                    .collect()
            },
            Some(Complex32::new(0.0, 1.0)),
        );
    }

    #[test]
    fn complex128_inputs_slice_products_and_imaginary_error_check() {
        verify_matrix(
            Dtype::Complex128,
            REAL_PARTS
                .into_iter()
                .zip(IMAGINARY_PARTS)
                .map(|(re, im)| Complex64::new(re, im))
                .collect(),
            [(18.0, -4.0), (23.0, 3.0), (43.0, 6.0), (48.0, 23.0)]
                .map(|(re, im)| Complex64::new(re, im))
                .to_vec(),
            |x| {
                x.re.to_le_bytes()
                    .into_iter()
                    .chain(x.im.to_le_bytes())
                    .collect()
            },
            Some(Complex64::new(0.0, 1.0)),
        );
    }

    #[test]
    fn wrong_input_byte_counts_are_rejected_for_every_dtype() {
        fn check<T: Scalar>(dtype: Dtype) {
            let expected = 8 * T::NBYTES;
            for actual in [expected - 1, expected + 1] {
                let error = load_matrix_bytes::<T>(dtype, &vec![0; actual]).unwrap_err();
                assert!(error.contains(&format!("input has {actual} bytes, expected {expected}")));
                assert!(error.contains(dtype.name()));
            }
        }
        check::<f32>(Dtype::F32);
        check::<f64>(Dtype::F64);
        check::<Complex32>(Dtype::Complex64);
        check::<Complex64>(Dtype::Complex128);
    }

    #[test]
    fn slice_assignments_cover_every_slice_once() {
        for (count, ranks) in [(10, 3), (2, 4), (1, 4), (0, 4)] {
            let assigned: Vec<_> = (0..ranks)
                .flat_map(|rank| {
                    let (begin, end) = slice_range(count, ranks, rank);
                    begin..end
                })
                .collect();
            assert_eq!(assigned, (0..count).collect::<Vec<_>>());
        }
    }

    #[test]
    fn slice_counts_are_checked() {
        assert_eq!(checked_product([]), Ok(1));
        assert_eq!(checked_product([2, 3]), Ok(6));
        assert!(checked_product([usize::MAX, 2]).is_err());
    }
}
