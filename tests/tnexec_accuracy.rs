use std::process::Command;

#[test]
fn check_uses_double_precision_references_for_all_execution_dtypes() {
    for (dtype_flags, dtype, oracle_dtype) in [
        (vec!["--single"], "f32", "f64"),
        (vec!["--single", "--complex"], "c32", "c64"),
        (vec![], "f64", "f64"),
        (vec!["--complex"], "c64", "c64"),
    ] {
        for target in [None, Some("8")] {
            let mut command = Command::new(env!("CARGO_BIN_EXE_tnexec"));
            command
                .current_dir(env!("CARGO_MANIFEST_DIR"))
                .env("MATMUL_NUM_THREADS", "1")
                .env("RAYON_NUM_THREADS", "1")
                .args([
                    "--net",
                    "tests/fixtures/demo_tiny6.net.json",
                    "--method",
                    "greedy",
                    "--check",
                ])
                .args(&dtype_flags);
            if let Some(target) = target {
                command.args(["--target-size", target]);
            }
            let output = command.output().expect("run tnexec accuracy regression");
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                output.status.success(),
                "dtype={dtype}, target={target:?}:\n{stdout}\n{stderr}"
            );
            assert!(stdout.contains(&format!("dtype={dtype},")), "{stdout}");
            assert!(stdout.contains("check=PASS(naive_"), "{stdout}");
            assert!(
                stdout.contains(&format!("oracle_dtype={oracle_dtype}")),
                "{stdout}"
            );
            if target.is_some() {
                assert!(!stdout.contains("n_sliced_legs=0,"), "{stdout}");
            }
        }
    }
}
