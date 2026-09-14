use std::process::Command;

fn assert_usage_error(binary: &str, args: &[&str]) {
    let output = Command::new(binary).args(args).output().unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(output.status.code(), Some(2), "{args:?}: {stderr}");
    assert!(!stderr.contains("panicked"), "{args:?}: {stderr}");
}

#[test]
fn pathfinder_rejects_missing_and_invalid_values_without_panicking() {
    let binary = env!("CARGO_BIN_EXE_tnpath");
    assert_usage_error(binary, &[]);
    assert_usage_error(binary, &["--slice-mode", "invalid"]);
    assert_usage_error(
        binary,
        &["tests/fixtures/demo_tiny6.net.json", "--method", "invalid"],
    );
    for flag in [
        "--load-path",
        "--save-path",
        "--ready-file",
        "--go-file",
        "--chains",
        "--rounds",
        "--moves",
        "--tmin",
        "--tmax",
        "--reconf-every",
        "--patience",
        "--slice-mode",
        "--method",
        "--trials",
        "--cache-dir",
        "--seed",
        "--max-n",
        "--reconf-size",
    ] {
        assert_usage_error(binary, &[flag]);
        assert_usage_error(binary, &[flag, "--quiet"]);
    }
    for flag in ["--trials", "--seed", "--tmin", "--chains"] {
        assert_usage_error(binary, &[flag, "invalid"]);
    }
    assert_usage_error(binary, &["--method", "auto"]);
    assert_usage_error(binary, &["--method", "import"]);
    assert_usage_error(binary, &["--method", "greedy", "--cache"]);
    assert_usage_error(
        binary,
        &[
            "tests/fixtures/demo_tiny6.net.json",
            "--method",
            "import",
            "--load-path",
            "tests/fixtures/nonexistent-path.json",
        ],
    );
}

#[test]
fn pathfinder_does_not_report_success_when_search_fails() {
    let output = Command::new(env!("CARGO_BIN_EXE_tnpath"))
        .args([
            "tests/fixtures/demo_tiny6.net.json",
            "--method",
            "optimal",
            "--max-n",
            "1",
        ])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(3));
    assert!(!String::from_utf8_lossy(&output.stderr).contains("panicked"));
}

#[test]
fn executor_rejects_invalid_numbers_without_panicking() {
    let binary = env!("CARGO_BIN_EXE_tnexec");
    for flag in ["--trials", "--seed"] {
        assert_usage_error(binary, &[flag]);
        assert_usage_error(binary, &[flag, "invalid"]);
        assert_usage_error(binary, &[flag, "--check"]);
    }
}

#[test]
fn executor_reports_bad_input_files_without_panicking() {
    let binary = env!("CARGO_BIN_EXE_tnexec");
    let missing = std::env::temp_dir().join(format!("arctn-missing-input-{}", std::process::id()));
    let _ = std::fs::remove_file(&missing);

    let missing_network = Command::new(binary)
        .args(["--net"])
        .arg(&missing)
        .output()
        .unwrap();
    assert_eq!(missing_network.status.code(), Some(2));
    assert!(!String::from_utf8_lossy(&missing_network.stderr).contains("panicked"));

    let missing_data = Command::new(binary)
        .args([
            "--net",
            "tests/fixtures/demo_tiny6.net.json",
            "--method",
            "greedy",
            "--data",
        ])
        .arg(&missing)
        .output()
        .unwrap();
    assert_eq!(missing_data.status.code(), Some(2));
    assert!(!String::from_utf8_lossy(&missing_data.stderr).contains("panicked"));
}

#[test]
fn executor_check_mismatch_exits_four() {
    let directory = std::env::temp_dir();
    let stem = format!("arctn-check-mismatch-{}", std::process::id());
    let network = directory.join(format!("{stem}.json"));
    let data = directory.join(format!("{stem}.bin"));
    std::fs::write(
        &network,
        r#"{"name":"nan-check","inputs":[["a","b"],["b","c"]],"output":["a","c"],"size_dict":{"a":2,"b":2,"c":2}}"#,
    )
    .unwrap();
    let mut bytes = Vec::with_capacity(8 * std::mem::size_of::<f64>());
    for _ in 0..8 {
        bytes.extend_from_slice(&f64::NAN.to_le_bytes());
    }
    std::fs::write(&data, bytes).unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_tnexec"))
        .args(["--net"])
        .arg(&network)
        .args(["--method", "greedy", "--data"])
        .arg(&data)
        .arg("--check")
        .output()
        .unwrap();
    let _ = std::fs::remove_file(network);
    let _ = std::fs::remove_file(data);

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(output.status.code(), Some(4), "{stdout}\n{stderr}");
    assert!(stdout.contains("check=FAIL("), "{stdout}");
    assert!(!stderr.contains("panicked"), "{stderr}");
}

#[cfg(feature = "mpi")]
#[test]
fn mpi_rejects_invalid_numbers_without_panicking() {
    let binary = env!("CARGO_BIN_EXE_tnmpi");
    for flag in ["--trials", "--seed", "--slices-min"] {
        assert_usage_error(binary, &[flag]);
        assert_usage_error(binary, &[flag, "invalid"]);
    }
}
