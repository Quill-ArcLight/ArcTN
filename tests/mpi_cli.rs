#![cfg(feature = "mpi")]

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use arctn::{
    complete_execution_plan_v2, contract_network_sliced, parse_embedded_execution_plan,
    parse_execution_plan_for_network, DenseTensor, Scalar, TensorNetwork,
};
use num_complex::Complex;

// Default CLI tests exit during argument parsing. The ignored integration test
// below explicitly launches mpirun when requested on an OpenMPI installation.
fn assert_usage_error(args: &[&str], expected: &str) {
    let output = Command::new(env!("CARGO_BIN_EXE_tnmpi"))
        .args(args)
        .output()
        .expect("run tnmpi argument validation");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(output.status.code(), Some(2), "{args:?}: {stderr}");
    assert!(stderr.contains(expected), "{args:?}: {stderr}");
    assert!(!stderr.contains("panicked"), "{args:?}: {stderr}");
}

fn dtype_fixture_network(scalar_output: bool) -> TensorNetwork {
    if scalar_output {
        TensorNetwork {
            name: "mpi-vector-dot".into(),
            inputs: vec![vec![0], vec![0]],
            output: vec![],
            size_dict: [(0, 2)].into_iter().collect(),
        }
    } else {
        TensorNetwork {
            name: "mpi-matrix-product".into(),
            inputs: vec![vec![0, 1], vec![1, 2]],
            output: vec![0, 2],
            size_dict: [(0, 2), (1, 2), (2, 2)].into_iter().collect(),
        }
    }
}

fn checked_fixture_bytes<T: Scalar + PartialEq>(
    scalar_output: bool,
    values: Vec<T>,
    expected: Vec<T>,
    encode: impl Fn(T) -> Vec<u8>,
) -> Vec<u8> {
    let network = dtype_fixture_network(scalar_output);
    let shape = if scalar_output { vec![2] } else { vec![2, 2] };
    let tensors = values
        .chunks_exact(values.len() / 2)
        .map(|data| DenseTensor::try_from_data(shape.clone(), data.to_vec()).unwrap())
        .collect::<Vec<_>>();
    for sliced in [vec![], vec![if scalar_output { 0 } else { 1 }]] {
        let result = contract_network_sliced(&network, &tensors, &vec![(0, 1)], &sliced).unwrap();
        assert_eq!(result.data(), expected.as_slice());
        assert_eq!(
            result.shape(),
            if scalar_output { &[][..] } else { &[2, 2][..] }
        );
    }
    values.into_iter().flat_map(encode).collect()
}

fn dtype_fixture_bytes(dtype: &str, scalar_output: bool) -> Vec<u8> {
    // Independent, hand-calculated oracles. The complex cases have nonzero
    // imaginary inputs and outputs, so dropping either component cannot pass.
    let values = if scalar_output {
        vec![(1.0, 1.0), (2.0, -2.0), (5.0, -1.0), (6.0, 2.0)]
    } else {
        vec![
            (1.0, 1.0),
            (2.0, -2.0),
            (3.0, 0.5),
            (4.0, 1.0),
            (5.0, -1.0),
            (6.0, 2.0),
            (7.0, 3.0),
            (8.0, -0.5),
        ]
    };
    let real_expected = if scalar_output {
        vec![17.0]
    } else {
        vec![19.0, 22.0, 43.0, 50.0]
    };
    let complex_expected = if scalar_output {
        vec![(22.0, -4.0)]
    } else {
        vec![(26.0, -4.0), (19.0, -9.0), (40.5, 18.5), (49.5, 15.0)]
    };
    match dtype {
        "f32" => checked_fixture_bytes(
            scalar_output,
            values.iter().map(|&(re, _)| re as f32).collect(),
            real_expected.iter().map(|&x| x as f32).collect(),
            |x: f32| x.to_le_bytes().to_vec(),
        ),
        "f64" => checked_fixture_bytes(
            scalar_output,
            values.iter().map(|&(re, _)| re).collect(),
            real_expected,
            |x: f64| x.to_le_bytes().to_vec(),
        ),
        "complex64" => checked_fixture_bytes(
            scalar_output,
            values
                .iter()
                .map(|&(re, im)| Complex::new(re as f32, im as f32))
                .collect(),
            complex_expected
                .iter()
                .map(|&(re, im)| Complex::new(re as f32, im as f32))
                .collect(),
            |x: Complex<f32>| {
                x.re.to_le_bytes()
                    .into_iter()
                    .chain(x.im.to_le_bytes())
                    .collect()
            },
        ),
        "complex128" => checked_fixture_bytes(
            scalar_output,
            values
                .iter()
                .map(|&(re, im)| Complex::new(re, im))
                .collect(),
            complex_expected
                .iter()
                .map(|&(re, im)| Complex::new(re, im))
                .collect(),
            |x: Complex<f64>| {
                x.re.to_le_bytes()
                    .into_iter()
                    .chain(x.im.to_le_bytes())
                    .collect()
            },
        ),
        _ => panic!("unsupported fixture dtype: {dtype}"),
    }
}

#[test]
fn all_dtype_fixtures_match_known_matrix_and_scalar_results() {
    for dtype in ["f32", "f64", "complex64", "complex128"] {
        for scalar_output in [false, true] {
            assert!(!dtype_fixture_bytes(dtype, scalar_output).is_empty());
        }
    }
}

struct MpiFixtureDirectory(PathBuf);

impl Drop for MpiFixtureDirectory {
    fn drop(&mut self) {
        if std::thread::panicking() {
            eprintln!("MPI fixture files retained at {}", self.0.display());
        } else {
            std::fs::remove_dir_all(&self.0).expect("remove this test's fixture directory");
        }
    }
}

fn write_mpi_plan(directory: &Path, scalar_output: bool, sliced: bool) -> PathBuf {
    let network = dtype_fixture_network(scalar_output);
    let labels = (0..network.size_dict.len())
        .map(|id| id.to_string())
        .collect::<Vec<_>>();
    let sliced_legs = if sliced {
        vec![if scalar_output { 0 } else { 1 }]
    } else {
        vec![]
    };
    let record = complete_execution_plan_v2(
        &network,
        &labels,
        serde_json::json!({
            "ssa_path": [[0, 1]],
            "method": "test",
            "sliced": {"legs": sliced_legs},
        }),
    )
    .unwrap();
    let path = directory.join(format!("scalar-{scalar_output}-sliced-{sliced}.json"));
    std::fs::write(&path, serde_json::to_vec_pretty(&record).unwrap()).unwrap();
    path
}

fn run_mpi_fixture(plan: &Path, data: &Path, dtype: Option<&str>, ranks: usize) -> Output {
    let mut command = Command::new("mpirun");
    command
        // OpenMPI terminates all ranks if a regression hangs in a collective.
        .args([
            "--timeout",
            "60",
            "--oversubscribe",
            "-n",
            &ranks.to_string(),
        ])
        .arg(env!("CARGO_BIN_EXE_tnmpi"))
        .arg("--load-path")
        .arg(plan)
        .arg("--data")
        .arg(data)
        .arg("--check")
        .env("RAYON_NUM_THREADS", "1")
        .env("MATMUL_NUM_THREADS", "1");
    if let Some(dtype) = dtype {
        command.args(["--dtype", dtype]);
    }
    command
        .output()
        .expect("launch OpenMPI with the built tnmpi")
}

fn assert_mpi_fixture(
    plan: &Path,
    data: &Path,
    dtype: Option<&str>,
    ranks: usize,
    scalar_output: bool,
    sliced: bool,
) {
    let output = run_mpi_fixture(plan, data, dtype, ranks);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "{dtype:?}, {ranks} ranks:\n{stdout}\n{stderr}"
    );
    let summary: serde_json::Value = stdout
        .lines()
        .find_map(|line| serde_json::from_str(line).ok())
        .expect("rank 0 JSON summary");
    assert_eq!(summary["dtype"], dtype.unwrap_or("f64"));
    assert_eq!(summary["ranks"], ranks);
    assert_eq!(summary["n_slices"], if sliced { 2 } else { 1 });
    assert_eq!(
        summary["output_shape"],
        if scalar_output {
            serde_json::json!([])
        } else {
            serde_json::json!([2, 2])
        }
    );
    assert!(
        summary["check"]
            .as_str()
            .is_some_and(|check| check.starts_with("PASS(")),
        "{summary}"
    );
}

#[test]
#[ignore = "requires OpenMPI; run explicitly with --ignored --exact"]
fn mpi_dtypes_match_known_results() {
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let directory = MpiFixtureDirectory(
        std::env::temp_dir().join(format!("arctn-mpi-dtypes-{}-{unique}", std::process::id())),
    );
    std::fs::create_dir(&directory.0).unwrap();
    let sliced_matrix = write_mpi_plan(&directory.0, false, true);
    let unsliced_matrix = write_mpi_plan(&directory.0, false, false);
    let sliced_scalar = write_mpi_plan(&directory.0, true, true);
    for dtype in ["f32", "f64", "complex64", "complex128"] {
        let matrix_data = directory.0.join(format!("matrix-{dtype}.bin"));
        // This verifies the exact same fixture against hand-calculated values
        // locally before tnmpi --check compares the distributed result.
        let bytes = dtype_fixture_bytes(dtype, false);
        std::fs::write(&matrix_data, &bytes).unwrap();
        for ranks in [1, 2, 4] {
            // There are only two slices: four ranks exercise empty ranges.
            assert_mpi_fixture(
                &sliced_matrix,
                &matrix_data,
                Some(dtype),
                ranks,
                false,
                true,
            );
        }
        assert_mpi_fixture(&unsliced_matrix, &matrix_data, Some(dtype), 2, false, false);
        let scalar_data = directory.0.join(format!("scalar-{dtype}.bin"));
        std::fs::write(&scalar_data, dtype_fixture_bytes(dtype, true)).unwrap();
        assert_mpi_fixture(&sliced_scalar, &scalar_data, Some(dtype), 2, true, true);

        if dtype == "f64" {
            assert_mpi_fixture(&sliced_matrix, &matrix_data, None, 2, false, true);
        }
        // Test both a partial component and a whole missing/extra element.
        let invalid = match dtype {
            "f32" => Some(bytes[..bytes.len() - 1].to_vec()),
            "complex64" => Some([bytes.as_slice(), &[0; 8]].concat()),
            "complex128" => Some(bytes[..bytes.len() - 16].to_vec()),
            _ => None,
        };
        if let Some(invalid) = invalid {
            let invalid_data = directory.0.join(format!("invalid-{dtype}.bin"));
            std::fs::write(&invalid_data, invalid).unwrap();
            let output = run_mpi_fixture(&sliced_matrix, &invalid_data, Some(dtype), 2);
            assert!(
                !output.status.success(),
                "{dtype}: invalid data length accepted"
            );
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                stderr.contains("input has") && stderr.contains("bytes, expected"),
                "{stderr}"
            );
        }
    }
    let output = run_mpi_fixture(
        &sliced_matrix,
        &directory.0.join("missing.bin"),
        Some("f64"),
        2,
    );
    assert!(!output.status.success(), "missing data file accepted");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("cannot read") && stderr.contains("missing.bin"),
        "{stderr}"
    );
}

#[test]
fn help_describes_saved_plan_execution() {
    let output = Command::new(env!("CARGO_BIN_EXE_tnmpi"))
        .arg("--help")
        .output()
        .expect("run tnmpi help");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{stdout}\n{stderr}");
    assert!(stdout.contains("Usage: tnmpi --load-path"), "{stdout}");
    for option in ["--net", "--data", "--seed", "--dtype", "--check"] {
        assert!(stdout.contains(option), "{stdout}");
    }
    for dtype in ["f32", "f64", "complex64", "complex128"] {
        assert!(stdout.contains(dtype), "{stdout}");
    }
}

#[test]
fn execution_requires_a_saved_plan() {
    assert_usage_error(&[], "--load-path");
    assert_usage_error(&["--check"], "--load-path");
    assert_usage_error(
        &["--net", "tests/fixtures/demo_tiny6.net.json"],
        "--load-path",
    );
}

#[test]
fn execution_rejects_missing_and_invalid_values() {
    for flag in ["--load-path", "--net", "--data", "--seed", "--dtype"] {
        assert_usage_error(&[flag], flag);
        assert_usage_error(&[flag, "--check"], flag);
    }
    for value in ["invalid", "-1", "18446744073709551616"] {
        assert_usage_error(
            &["--load-path", "unused-plan.json", "--seed", value],
            "--seed",
        );
    }
}

#[test]
fn execution_accepts_only_the_four_supported_dtypes() {
    for dtype in ["f32", "f64", "complex64", "complex128"] {
        // A valid dtype reaches the missing-plan check without initializing MPI.
        assert_usage_error(&["--dtype", dtype], "--load-path");
    }
    for dtype in ["", "float32", "float64", "fp128", "complex32", "F64"] {
        assert_usage_error(
            &["--load-path", "unused-plan.json", "--dtype", dtype],
            "--dtype",
        );
    }
}

#[test]
fn execution_rejects_obsolete_planning_and_partition_options() {
    for (flag, value) in [
        ("--trials", Some("1")),
        ("--scaling-mode", None),
        ("--partition", None),
        ("--slices-min", Some("2")),
        ("--target-size", Some("4")),
    ] {
        let mut args = vec!["--load-path", "unused-plan.json", flag];
        if let Some(value) = value {
            args.push(value);
        }
        assert_usage_error(&args, flag);
    }
}

#[test]
fn saved_sliced_plan_fixture_round_trips_and_contracts() {
    let network = TensorNetwork {
        name: "mpi-matrix-product".to_owned(),
        inputs: vec![vec![0, 1], vec![1, 2]],
        output: vec![0, 2],
        size_dict: [(0, 2), (1, 2), (2, 2)].into_iter().collect(),
    };
    let labels = vec!["0".to_owned(), "1".to_owned(), "2".to_owned()];
    let record = complete_execution_plan_v2(
        &network,
        &labels,
        serde_json::json!({
            "ssa_path": [[0, 1]],
            "method": "test",
            "sliced": {"legs": [1]},
        }),
    )
    .unwrap();
    let plan_text = serde_json::to_string_pretty(&record).unwrap();
    let loaded = parse_embedded_execution_plan(&plan_text).unwrap();
    let with_network =
        parse_execution_plan_for_network(&plan_text, &network, &labels, false).unwrap();
    assert_eq!(loaded.execution.schema_version, Some(2));
    assert_eq!(loaded.execution.path, vec![(0, 1)]);
    assert_eq!(loaded.execution.sliced, Some(vec![1]));
    assert_eq!(with_network.path, loaded.execution.path);
    assert_eq!(with_network.sliced, loaded.execution.sliced);

    let values = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let tensors: Vec<_> = values
        .chunks_exact(4)
        .map(|data| DenseTensor::try_from_data(vec![2, 2], data.to_vec()).unwrap())
        .collect();
    let result = contract_network_sliced(
        &loaded.network,
        &tensors,
        &loaded.execution.path,
        loaded.execution.sliced.as_deref().unwrap(),
    )
    .unwrap();
    assert_eq!(result.shape(), &[2, 2]);
    assert_eq!(result.data(), &[19.0, 22.0, 43.0, 50.0]);

    // Opt-in export for a manual two-rank check:
    // ARCTN_MPI_FIXTURE_DIR=<empty-directory> cargo test --features mpi --test mpi_cli \
    //     saved_sliced_plan_fixture_round_trips_and_contracts
    // mpirun -n 2 target/debug/tnmpi --load-path <directory>/plan.json \
    //     --data <directory>/input.f64le --check
    if let Some(directory) = std::env::var_os("ARCTN_MPI_FIXTURE_DIR") {
        use std::io::Write;

        let directory = std::path::PathBuf::from(directory);
        std::fs::create_dir_all(&directory).unwrap();
        let data: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        for (filename, bytes) in [
            ("plan.json", plan_text.into_bytes()),
            ("network.json", network.to_json_string().into_bytes()),
            ("input.f64le", data),
        ] {
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(directory.join(filename))
                .unwrap();
            file.write_all(&bytes).unwrap();
        }
    }
}
