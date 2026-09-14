use std::path::PathBuf;
use std::process::Command;

fn git_output(root: &std::path::Path, args: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(args)
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
        .filter(|value| !value.is_empty())
}

fn emit_git_rerun_paths(root: &std::path::Path) {
    let Some(git_dir) = git_output(root, &["rev-parse", "--git-dir"]) else {
        return;
    };
    let git_dir = PathBuf::from(git_dir);
    let git_dir = if git_dir.is_absolute() {
        git_dir
    } else {
        root.join(git_dir)
    };
    let head = git_dir.join("HEAD");
    println!("cargo:rerun-if-changed={}", head.display());
    println!(
        "cargo:rerun-if-changed={}",
        git_dir.join("packed-refs").display()
    );
    let Ok(head_contents) = std::fs::read_to_string(head) else {
        return;
    };
    if let Some(reference) = head_contents.trim().strip_prefix("ref: ") {
        println!(
            "cargo:rerun-if-changed={}",
            git_dir.join(reference).display()
        );
    }
}

fn emit_source_rerun_paths(root: &std::path::Path, manifest_dir: &std::path::Path) {
    for relative in ["src", "Cargo.toml", "Cargo.lock", "build.rs"] {
        println!("cargo:rerun-if-changed={}", root.join(relative).display());
    }
    for relative in [
        "src",
        "python/arctn/__init__.py",
        "python/arctn/_execution_plan.py",
        "python/arctn/py.typed",
        "Cargo.toml",
        "Cargo.lock",
        "build.rs",
        "pyproject.toml",
    ] {
        println!(
            "cargo:rerun-if-changed={}",
            manifest_dir.join(relative).display()
        );
    }
}

fn main() {
    // Python provides these symbols when loading the extension.
    if std::env::var("CARGO_CFG_TARGET_VENDOR").as_deref() == Ok("apple") {
        println!("cargo:rustc-link-arg=-undefined");
        println!("cargo:rustc-link-arg=dynamic_lookup");
    }

    println!("cargo:rerun-if-env-changed=ARCTN_BUILD_COMMIT");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_MT");

    let manifest = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let root = manifest
        .parent()
        .expect("pybind must have a repository root");
    emit_git_rerun_paths(root);
    emit_source_rerun_paths(root, &manifest);
    let commit = std::env::var("ARCTN_BUILD_COMMIT")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .or_else(|| git_output(root, &["rev-parse", "--verify", "HEAD"]))
        .unwrap_or_else(|| "unknown".to_owned());
    let source_state = match Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["status", "--porcelain=v1", "--untracked-files=normal"])
        .output()
    {
        Ok(output) if output.status.success() && output.stdout.is_empty() => "clean",
        Ok(output) if output.status.success() => "dirty",
        _ => "unknown",
    };
    println!("cargo:rustc-env=ARCTN_PY_BUILD_COMMIT={commit}");
    println!("cargo:rustc-env=ARCTN_PY_BUILD_SOURCE_STATE={source_state}");
    println!(
        "cargo:rustc-env=ARCTN_PY_BUILD_FEATURE_MT={}",
        u8::from(std::env::var_os("CARGO_FEATURE_MT").is_some())
    );
}
