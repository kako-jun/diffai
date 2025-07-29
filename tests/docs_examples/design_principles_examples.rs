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

/// Test case 1: diffai model1.pth model2.pth (comprehensive ML analysis automatic)
#[test]
fn test_comprehensive_ml_analysis_automatic() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(r#"{"model": {"type": "pytorch", "layers": 5}}"#, ".safetensors");
    let file2 = create_temp_file(r#"{"model": {"type": "pytorch", "layers": 8}}"#, ".safetensors");

    let mut cmd = diffai_cmd();
    cmd.arg(file1.path()).arg(file2.path());
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("layers:"));

    Ok(())
}

/// Test case 2: diffai model1.pth model2.pth --verbose (detailed diagnostics + comprehensive analysis)
#[test]
fn test_verbose_comprehensive_analysis() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(r#"{"diagnostics": {"enabled": true}}"#, ".safetensors");
    let file2 = create_temp_file(r#"{"diagnostics": {"enabled": false}}"#, ".safetensors");

    let mut cmd = diffai_cmd();
    cmd.arg(file1.path()).arg(file2.path()).arg("--verbose");
    cmd.assert()
        .code(1)
        .stderr(predicates::str::contains("verbose mode"));

    Ok(())
}

/// Test case 3: diffai models/ --recursive (directory comparison)
#[test]
fn test_recursive_directory_comparison() -> Result<(), Box<dyn std::error::Error>> {
    // Use existing fixtures for directory comparison
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/dir1/")
        .arg("tests/fixtures/dir2/")
        .arg("--recursive");
    cmd.assert().code(1); // Should find differences between directories

    Ok(())
}

/// Test case 4: diffai model1.pth model2.pth (comprehensive analysis automatic)
#[test]
fn test_ml_analysis_automatic() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(r#"{"ml": {"features": 30, "accuracy": 0.85}}"#, ".safetensors");
    let file2 = create_temp_file(r#"{"ml": {"features": 35, "accuracy": 0.90}}"#, ".safetensors");

    let mut cmd = diffai_cmd();
    cmd.arg(file1.path()).arg(file2.path());
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("features:"))
        .stdout(predicates::str::contains("accuracy:"));

    Ok(())
}

/// Test case 5: diffai model1.pth model2.pth --verbose (same comprehensive analysis + debugging info)
#[test]
fn test_verbose_ml_analysis() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(r#"{"debug": {"level": 1}}"#, ".safetensors");
    let file2 = create_temp_file(r#"{"debug": {"level": 2}}"#, ".safetensors");

    let mut cmd = diffai_cmd();
    cmd.arg(file1.path()).arg(file2.path()).arg("--verbose");
    cmd.assert()
        .code(1)
        .stderr(predicates::str::contains("verbose mode"))
        .stdout(predicates::str::contains("level:"));

    Ok(())
}

/// Test case 6: diffai model1.pth model2.pth --output json (comprehensive analysis in JSON format)
#[test]
fn test_json_comprehensive_analysis() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(r#"{"output": {"format": "cli"}}"#, ".safetensors");
    let file2 = create_temp_file(r#"{"output": {"format": "json"}}"#, ".safetensors");

    let mut cmd = diffai_cmd();
    cmd.arg(file1.path())
        .arg(file2.path())
        .arg("--output")
        .arg("json");
    cmd.assert()
        .code(1)
        .stdout(predicates::str::is_match(r#"\[.*\]"#).unwrap()); // JSON array format

    Ok(())
}
