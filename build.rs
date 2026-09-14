use std::path::{Path, PathBuf};
use std::process::Command;

fn git_output(root: &Path, args: &[&str]) -> Option<String> {
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

fn git_source_state(root: &Path) -> &'static str {
    let Ok(output) = Command::new("git")
        .arg("-C")
        .arg(root)
        .args([
            "status",
            "--porcelain=v1",
            "--untracked-files=normal",
            "--ignore-submodules=none",
        ])
        .output()
    else {
        return "unknown";
    };
    if !output.status.success() {
        "unknown"
    } else if output.stdout.is_empty() {
        "clean"
    } else {
        "dirty"
    }
}

/// Watches HEAD and its branch reference so incremental builds refresh provenance.
fn emit_git_rerun_paths(root: &Path) {
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

fn emit_source_rerun_paths(root: &Path) {
    for relative in ["src", "Cargo.toml", "Cargo.lock", "build.rs"] {
        println!("cargo:rerun-if-changed={}", root.join(relative).display());
    }
}

fn main() {
    println!("cargo:rerun-if-env-changed=ARCTN_BUILD_COMMIT");
    println!("cargo:rerun-if-env-changed=PROFILE");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_MT");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_MPI");

    let root = std::env::var_os("CARGO_MANIFEST_DIR").map(PathBuf::from);
    if let Some(root) = root.as_deref() {
        emit_git_rerun_paths(root);
        emit_source_rerun_paths(root);
    }

    // Source archives and CI may provide a commit explicitly; checkouts read HEAD.
    let commit = std::env::var("ARCTN_BUILD_COMMIT")
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
        .or_else(|| {
            root.as_deref()
                .and_then(|root| git_output(root, &["rev-parse", "--verify", "HEAD"]))
        })
        .unwrap_or_else(|| "unknown".to_owned());
    let profile = std::env::var("PROFILE").unwrap_or_else(|_| "unknown".to_owned());
    let source_state = root.as_deref().map(git_source_state).unwrap_or("unknown");
    let mt = std::env::var_os("CARGO_FEATURE_MT").is_some();
    let mpi = std::env::var_os("CARGO_FEATURE_MPI").is_some();

    println!("cargo:rustc-env=ARCTN_GIT_COMMIT={commit}");
    println!("cargo:rustc-env=ARCTN_BUILD_SOURCE_STATE={source_state}");
    println!("cargo:rustc-env=ARCTN_BUILD_PROFILE={profile}");
    println!("cargo:rustc-env=ARCTN_BUILD_FEATURE_MT={}", u8::from(mt));
    println!("cargo:rustc-env=ARCTN_BUILD_FEATURE_MPI={}", u8::from(mpi));
}
