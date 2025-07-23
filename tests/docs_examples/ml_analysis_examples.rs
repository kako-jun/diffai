#[allow(unused_imports)]
use assert_cmd::prelude::*;
#[allow(unused_imports)]
use predicates::prelude::*;
use std::io::Write;
use std::process::Command;
use tempfile::NamedTempFile;

// Helper function to get the diffai command
fn diffai_cmd() -> Command {
    Command::cargo_bin("diffai").expect("Failed to find diffai binary")
}

// Helper function to create temporary files for testing
fn create_temp_file(content: &str, suffix: &str) -> NamedTempFile {
    let mut file = NamedTempFile::with_suffix(suffix).expect("Failed to create temp file");
    writeln!(file, "{}", content).expect("Failed to write to temp file");
    file
}

/// Test case 1: diffai model1.safetensors model2.safetensors (comprehensive analysis automatic)
#[test]
fn test_comprehensive_analysis_automatic() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(r#"{"fc1": {"bias": 0.0018, "weight": -0.0002}}"#, ".json");
    let file2 = create_temp_file(r#"{"fc1": {"bias": 0.0017, "weight": -0.0001}}"#, ".json");

    let mut cmd = diffai_cmd();
    cmd.arg(file1.path()).arg(file2.path());
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("bias:"))
        .stdout(predicates::str::contains("weight:"));

    Ok(())
}

/// Test case 2: diffai model1.safetensors model2.safetensors --architecture-comparison
#[test]
fn test_architecture_comparison() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(
        r#"{"architecture": {"type": "transformer", "layers": 12}}"#,
        ".json",
    );
    let file2 = create_temp_file(
        r#"{"architecture": {"type": "transformer", "layers": 24}}"#,
        ".json",
    );

    let mut cmd = diffai_cmd();
    cmd.arg(file1.path())
        .arg(file2.path())
        .arg("--architecture-comparison");
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("layers:"));

    Ok(())
}

/// Test case 3: diffai model1.safetensors model2.safetensors --output json
#[test]
fn test_json_output_automation() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(
        r#"{"analysis": {"features": 30, "enabled": true}}"#,
        ".json",
    );
    let file2 = create_temp_file(
        r#"{"analysis": {"features": 35, "enabled": true}}"#,
        ".json",
    );

    let mut cmd = diffai_cmd();
    cmd.arg(file1.path())
        .arg(file2.path())
        .arg("--output")
        .arg("json");
    cmd.assert()
        .code(1)
        .stdout(predicates::str::is_match(r#"\[.*\]"#).unwrap());

    Ok(())
}
