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

/// Test case 1: diffai model1.safetensors model2.safetensors --verbose
#[test]
fn test_basic_verbose_output() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(r#"{"model": {"layers": 2}}"#, ".safetensors");
    let file2 = create_temp_file(r#"{"model": {"layers": 3}}"#, ".safetensors");

    let mut cmd = diffai_cmd();
    cmd.arg(file1.path()).arg(file2.path()).arg("--verbose");
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("model"));

    Ok(())
}

/// Test case 2: diffai model1.pt model2.pt -v
#[test]
fn test_verbose_short_form() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(r#"{"state_dict": {"layer1": 1}}"#, ".pt");
    let file2 = create_temp_file(r#"{"state_dict": {"layer1": 2}}"#, ".pt");

    let mut cmd = diffai_cmd();
    cmd.arg(file1.path()).arg(file2.path()).arg("-v");
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("state_dict"));

    Ok(())
}

// Test case 3 removed: complex options with general formats now refer to diffx in documentation

/// Test case 4: diffai model1.safetensors model2.safetensors --verbose --architecture-comparison --memory-analysis --anomaly-detection
#[test]
fn test_verbose_ml_analysis_features() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(
        r#"{"model": {"architecture": "transformer", "memory": "2GB"}}"#,
        ".safetensors",
    );
    let file2 = create_temp_file(
        r#"{"model": {"architecture": "transformer", "memory": "2.5GB"}}"#,
        ".safetensors",
    );

    let mut cmd = diffai_cmd();
    cmd.arg(file1.path())
        .arg(file2.path())
        .arg("--verbose")
        .arg("--architecture-comparison")
        .arg("--memory-analysis")
        .arg("--anomaly-detection");
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("model"));

    Ok(())
}

/// Test case 5: diffai dir1/ dir2/ --verbose --recursive
#[test]
fn test_verbose_directory_comparison() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(r#"{"directory": {"files": 12}}"#, ".safetensors");
    let file2 = create_temp_file(r#"{"directory": {"files": 14}}"#, ".safetensors");

    let mut cmd = diffai_cmd();
    cmd.arg(file1.path())
        .arg(file2.path())
        .arg("--verbose")
        .arg("--recursive");
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("directory"));

    Ok(())
}

/// Test case 6: diffai problematic_file1.dat problematic_file2.dat --verbose
#[test]
fn test_verbose_debugging_format_detection() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(r#"{"format": "unknown", "data": "test1"}"#, ".safetensors");
    let file2 = create_temp_file(r#"{"format": "unknown", "data": "test2"}"#, ".safetensors");

    let mut cmd = diffai_cmd();
    cmd.arg(file1.path()).arg(file2.path()).arg("--verbose");
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("format"));

    Ok(())
}

/// Test case 7: diffai model1.pt model2.pt --verbose
#[test]
fn test_verbose_ml_analysis_automatic() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(
        r#"{"model": {"type": "pytorch", "version": "1.0"}}"#,
        ".safetensors",
    );
    let file2 = create_temp_file(
        r#"{"model": {"type": "pytorch", "version": "2.0"}}"#,
        ".safetensors",
    );

    let mut cmd = diffai_cmd();
    cmd.arg(file1.path()).arg(file2.path()).arg("--verbose");
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("model"));

    Ok(())
}

/// Test case 8: diffai dir1/ dir2/ --verbose --recursive
#[test]
fn test_verbose_directory_analysis() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(r#"{"scan": {"dir1": {"files": 12}}}"#, ".safetensors");
    let file2 = create_temp_file(r#"{"scan": {"dir2": {"files": 14}}}"#, ".safetensors");

    let mut cmd = diffai_cmd();
    cmd.arg(file1.path())
        .arg(file2.path())
        .arg("--verbose")
        .arg("--recursive");
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("scan"));

    Ok(())
}

/// Test case 9: diffai large_model1.safetensors large_model2.safetensors --verbose
#[test]
fn test_verbose_performance_analysis() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(
        r#"{"large_model": {"size": "1GB", "parameters": 1000000}}"#,
        ".safetensors",
    );
    let file2 = create_temp_file(
        r#"{"large_model": {"size": "1.2GB", "parameters": 1200000}}"#,
        ".safetensors",
    );

    let mut cmd = diffai_cmd();
    cmd.arg(file1.path()).arg(file2.path()).arg("--verbose");
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("large_model"));

    Ok(())
}

/// Test case 10: diffai data1.json data2.json --verbose --epsilon 0.0001
#[test]
fn test_verbose_performance_with_options() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(r#"{"data": {"precision": 0.12345}}"#, ".safetensors");
    let file2 = create_temp_file(r#"{"data": {"precision": 0.12346}}"#, ".safetensors");

    let mut cmd = diffai_cmd();
    cmd.arg(file1.path())
        .arg(file2.path())
        .arg("--verbose")
        .arg("--epsilon")
        .arg("0.0001");
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("data"));

    Ok(())
}

/// Test case 11: diffai config1.yaml config2.yaml --verbose --ignore-keys-regex "^(id|timestamp)$" --path "application.settings" --epsilon 0.01 --output json
#[test]
fn test_verbose_configuration_validation() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(
        r#"{"id": "001", "timestamp": "12:00", "application": {"settings": {"timeout": 30}}}"#,
        ".safetensors",
    );
    let file2 = create_temp_file(
        r#"{"id": "002", "timestamp": "13:00", "application": {"settings": {"timeout": 60}}}"#,
        ".safetensors",
    );

    let mut cmd = diffai_cmd();
    cmd.arg(file1.path())
        .arg(file2.path())
        .arg("--verbose")
        .arg("--ignore-keys-regex")
        .arg("^(id|timestamp)$")
        .arg("--path")
        .arg("application.settings")
        .arg("--epsilon")
        .arg("0.01")
        .arg("--output")
        .arg("json");
    cmd.assert()
        .code(1)
        .stdout(predicates::str::is_match(r#"\[.*\]"#).unwrap());

    Ok(())
}

/// Test case 12: diffai file1.json file2.json --verbose --output json > results.json
#[test]
fn test_verbose_output_redirection() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(r#"{"result": {"status": "success"}}"#, ".safetensors");
    let file2 = create_temp_file(r#"{"result": {"status": "failure"}}"#, ".safetensors");

    let mut cmd = diffai_cmd();
    cmd.arg(file1.path())
        .arg(file2.path())
        .arg("--verbose")
        .arg("--output")
        .arg("json");
    cmd.assert()
        .code(1)
        .stdout(predicates::str::is_match(r#"\[.*\]"#).unwrap());

    Ok(())
}

/// Test case 13: diffai baseline.safetensors new_model.safetensors --verbose
#[test]
fn test_verbose_cicd_integration() -> Result<(), Box<dyn std::error::Error>> {
    let baseline = create_temp_file(
        r#"{"model": {"type": "baseline", "accuracy": 0.85}}"#,
        ".safetensors",
    );
    let new_model = create_temp_file(
        r#"{"model": {"type": "improved", "accuracy": 0.90}}"#,
        ".safetensors",
    );

    let mut cmd = diffai_cmd();
    cmd.arg(baseline.path())
        .arg(new_model.path())
        .arg("--verbose");
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("model"));

    Ok(())
}

/// Test case 14: diffai file1.json file2.json --verbose 2>&1
#[test]
fn test_verbose_script_automation() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(r#"{"script": {"test": "automation"}}"#, ".safetensors");
    let file2 = create_temp_file(r#"{"script": {"test": "automated"}}"#, ".safetensors");

    let mut cmd = diffai_cmd();
    cmd.arg(file1.path()).arg(file2.path()).arg("--verbose");
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("script"));

    Ok(())
}

/// Test case 15: diffai file1.json file2.json --verbose 2>&1 >/dev/null
#[test]
fn test_verbose_only_information() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(r#"{"verbose": {"info": "test"}}"#, ".safetensors");
    let file2 = create_temp_file(r#"{"verbose": {"info": "tested"}}"#, ".safetensors");

    let mut cmd = diffai_cmd();
    cmd.arg(file1.path()).arg(file2.path()).arg("--verbose");
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("verbose"));

    Ok(())
}
