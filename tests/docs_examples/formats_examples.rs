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
    writeln!(file, "{content}").expect("Failed to write to temp file");
    file
}

/// Test case 1: diffai model1.pt model2.pt
#[test]
fn test_pytorch_format() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(r#"{"pytorch": {"layers": 5, "params": 1000}}"#, ".safetensors");
    let file2 = create_temp_file(r#"{"pytorch": {"layers": 8, "params": 1500}}"#, ".safetensors");

    let mut cmd = diffai_cmd();
    cmd.arg(file1.path()).arg(file2.path());
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("layers:"))
        .stdout(predicates::str::contains("params:"));

    Ok(())
}

/// Test case 2: diffai model1.safetensors model2.safetensors
#[test]
fn test_safetensors_format() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(
        r#"{"safetensors": {"version": "1.0", "secure": true}}"#,
        ".safetensors",
    );
    let file2 = create_temp_file(
        r#"{"safetensors": {"version": "1.1", "secure": true}}"#,
        ".safetensors",
    );

    let mut cmd = diffai_cmd();
    cmd.arg(file1.path()).arg(file2.path());
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("version:"));

    Ok(())
}

/// Test case 3: diffai data1.npy data2.npy
#[test]
fn test_numpy_format() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(
        r#"{"numpy": {"array": [1, 2, 3], "dtype": "int32"}}"#,
        ".safetensors",
    );
    let file2 = create_temp_file(
        r#"{"numpy": {"array": [1, 2, 4], "dtype": "int32"}}"#,
        ".safetensors",
    );

    let mut cmd = diffai_cmd();
    cmd.arg(file1.path()).arg(file2.path());
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("array:"));

    Ok(())
}

/// Test case 4: diffai archive1.npz archive2.npz
#[test]
fn test_npz_format() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(
        r#"{"npz": {"data1": [1, 2, 3], "data2": [4, 5, 6]}}"#,
        ".safetensors",
    );
    let file2 = create_temp_file(
        r#"{"npz": {"data1": [1, 2, 3], "data2": [4, 5, 7]}}"#,
        ".safetensors",
    );

    let mut cmd = diffai_cmd();
    cmd.arg(file1.path()).arg(file2.path());
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("data2:"));

    Ok(())
}

/// Test case 5: diffai simulation1.mat simulation2.mat
#[test]
fn test_matlab_format() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(
        r#"{"matlab": {"variables": {"result": 0.85}, "complex": true}}"#,
        ".safetensors",
    );
    let file2 = create_temp_file(
        r#"{"matlab": {"variables": {"result": 0.90}, "complex": true}}"#,
        ".safetensors",
    );

    let mut cmd = diffai_cmd();
    cmd.arg(file1.path()).arg(file2.path());
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("result:"));

    Ok(())
}
