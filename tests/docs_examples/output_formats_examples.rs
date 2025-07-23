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

/// Test case 1: diffai model1.safetensors model2.safetensors --output cli
#[test]
fn test_cli_output_format() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(
        r#"{"fc1": {"bias": {"mean": 0.0018, "std": 0.0518}}}"#,
        ".json",
    );
    let file2 = create_temp_file(
        r#"{"fc1": {"bias": {"mean": 0.0017, "std": 0.0647}}}"#,
        ".json",
    );

    let mut cmd = diffai_cmd();
    cmd.arg(file1.path())
        .arg(file2.path())
        .arg("--output")
        .arg("cli");
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("fc1"));

    Ok(())
}

/// Test case 2: diffai model1.safetensors model2.safetensors (default)
#[test]
fn test_default_cli_output() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(r#"{"layers": 12, "hidden_size": 768}"#, ".json");
    let file2 = create_temp_file(r#"{"layers": 24, "hidden_size": 768}"#, ".json");

    let mut cmd = diffai_cmd();
    cmd.arg(file1.path()).arg(file2.path());
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("layers"));

    Ok(())
}

/// Test case 3: diffai model1.safetensors model2.safetensors --output json
#[test]
fn test_json_output_format() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(
        r#"{"fc1": {"bias": {"mean": 0.0018, "std": 0.0518}}}"#,
        ".json",
    );
    let file2 = create_temp_file(
        r#"{"fc1": {"bias": {"mean": 0.0017, "std": 0.0647}}}"#,
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

/// Test case 4: diffai model1.safetensors model2.safetensors --output yaml
#[test]
fn test_yaml_output_format() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(r#"{"tensor": {"mean": 0.0018, "std": 0.0518}}"#, ".json");
    let file2 = create_temp_file(r#"{"tensor": {"mean": 0.0017, "std": 0.0647}}"#, ".json");

    let mut cmd = diffai_cmd();
    cmd.arg(file1.path())
        .arg(file2.path())
        .arg("--output")
        .arg("yaml");
    cmd.assert().code(1).stdout(predicates::str::contains("-"));

    Ok(())
}

/// Test case 5: diffai config1.json config2.json --output unified
#[test]
fn test_unified_output_format() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(r#"{"model": {"layers": 12, "hidden_size": 768}}"#, ".json");
    let file2 = create_temp_file(
        r#"{"model": {"layers": 24, "hidden_size": 768}, "optimizer": "adam"}"#,
        ".json",
    );

    let mut cmd = diffai_cmd();
    cmd.arg(file1.path())
        .arg(file2.path())
        .arg("--output")
        .arg("unified");
    cmd.assert().code(1).stdout(predicates::str::contains("@@"));

    Ok(())
}

/// Test case 6: diffai model1.safetensors model2.safetensors --output json | jq '.[] | select(.TensorStatsChanged)'
#[test]
fn test_json_with_jq_filter() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(
        r#"{"fc1": {"bias": {"mean": 0.0018, "std": 0.0518}}}"#,
        ".json",
    );
    let file2 = create_temp_file(
        r#"{"fc1": {"bias": {"mean": 0.0017, "std": 0.0647}}}"#,
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

/// Test case 7: diffai config1.yaml config2.yaml --output yaml > changes.yaml
#[test]
fn test_yaml_output_to_file() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(r#"{"config": {"timeout": 30, "retries": 3}}"#, ".json");
    let file2 = create_temp_file(r#"{"config": {"timeout": 60, "retries": 5}}"#, ".json");

    let mut cmd = diffai_cmd();
    cmd.arg(file1.path())
        .arg(file2.path())
        .arg("--output")
        .arg("yaml");
    cmd.assert().code(1).stdout(predicates::str::contains("-"));

    Ok(())
}

/// Test case 8: if diffai model1.safetensors model2.safetensors --output json | jq -e 'length > 0'
#[test]
fn test_conditional_logic_check() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(r#"{"model": {"parameters": 1000}}"#, ".json");
    let file2 = create_temp_file(r#"{"model": {"parameters": 2000}}"#, ".json");

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

/// Test case 9: diffai model1.safetensors model2.safetensors > human_readable.txt
#[test]
fn test_human_readable_output() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(r#"{"layer1": {"weights": [1.0, 2.0, 3.0]}}"#, ".json");
    let file2 = create_temp_file(r#"{"layer1": {"weights": [1.1, 2.1, 3.1]}}"#, ".json");

    let mut cmd = diffai_cmd();
    cmd.arg(file1.path()).arg(file2.path());
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("layer1"));

    Ok(())
}

/// Test case 10: diffai model1.safetensors model2.safetensors --output json > machine_readable.json
#[test]
fn test_machine_readable_output() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(r#"{"params": {"learning_rate": 0.001}}"#, ".json");
    let file2 = create_temp_file(r#"{"params": {"learning_rate": 0.01}}"#, ".json");

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

/// Test case 11: export DIFFAI_OUTPUT_FORMAT="json"
#[test]
fn test_env_var_json_format() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(r#"{"model_version": "1.0"}"#, ".json");
    let file2 = create_temp_file(r#"{"model_version": "2.0"}"#, ".json");

    let mut cmd = diffai_cmd();
    cmd.env("DIFFAI_OUTPUT_FORMAT", "json");
    cmd.arg(file1.path()).arg(file2.path());
    cmd.assert()
        .code(1)
        .stdout(predicates::str::is_match(r#"\[.*\]"#).unwrap());

    Ok(())
}

/// Test case 12: export DIFFAI_CLI_COLORS="true"
#[test]
fn test_env_var_cli_colors() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(r#"{"status": "active"}"#, ".json");
    let file2 = create_temp_file(r#"{"status": "inactive"}"#, ".json");

    let mut cmd = diffai_cmd();
    cmd.env("DIFFAI_CLI_COLORS", "true");
    cmd.arg(file1.path()).arg(file2.path());
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("status"));

    Ok(())
}

/// Test case 13: export DIFFAI_JSON_PRETTY="true"
#[test]
fn test_env_var_json_pretty() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(r#"{"data": {"value": 100}}"#, ".json");
    let file2 = create_temp_file(r#"{"data": {"value": 200}}"#, ".json");

    let mut cmd = diffai_cmd();
    cmd.env("DIFFAI_JSON_PRETTY", "true");
    cmd.arg(file1.path())
        .arg(file2.path())
        .arg("--output")
        .arg("json");
    cmd.assert()
        .code(1)
        .stdout(predicates::str::is_match(r#"\[.*\]"#).unwrap());

    Ok(())
}
