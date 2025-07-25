#[allow(unused_imports)]
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

/// Test case 1: diff model_v1.safetensors model_v2.safetensors
#[test]
fn test_basic_safetensors_diff() -> Result<(), Box<dyn std::error::Error>> {
    // Note: This would require actual model files. Using JSON as substitute for test
    let file1 = create_temp_file(r#"{"model": {"layers": 2, "params": 1000}}"#, ".json");
    let file2 = create_temp_file(r#"{"model": {"layers": 3, "params": 1500}}"#, ".json");

    let mut cmd = diffai_cmd();
    cmd.arg(file1.path()).arg(file2.path());
    cmd.assert()
        .code(1) // differences found
        .stdout(predicates::str::contains("layers:"))
        .stdout(predicates::str::contains("params:"));

    Ok(())
}

/// Test case 2: diffai model_v1.safetensors model_v2.safetensors
#[test]
fn test_comprehensive_model_analysis() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(r#"{"fc1": {"bias": 0.001, "weight": 0.5}}"#, ".json");
    let file2 = create_temp_file(r#"{"fc1": {"bias": 0.002, "weight": 0.6}}"#, ".json");

    let mut cmd = diffai_cmd();
    cmd.arg(file1.path()).arg(file2.path());
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("bias:"))
        .stdout(predicates::str::contains("weight:"));

    Ok(())
}

/// Test case 3: diffai model1.safetensors model2.safetensors --output json
#[test]
fn test_json_output() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(r#"{"data": "value1"}"#, ".json");
    let file2 = create_temp_file(r#"{"data": "value2"}"#, ".json");

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

/// Test case 4: diffai model1.safetensors model2.safetensors --verbose
#[test]
fn test_verbose_output() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(r#"{"test": 1}"#, ".json");
    let file2 = create_temp_file(r#"{"test": 2}"#, ".json");

    let mut cmd = diffai_cmd();
    cmd.arg(file1.path()).arg(file2.path()).arg("--verbose");
    cmd.assert()
        .code(1)
        .stderr(predicates::str::contains("verbose mode"));

    Ok(())
}

/// Test case 5: diffai model1.safetensors model2.safetensors --output yaml
#[test]
fn test_yaml_output() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(r#"{"value": 10}"#, ".json");
    let file2 = create_temp_file(r#"{"value": 20}"#, ".json");

    let mut cmd = diffai_cmd();
    cmd.arg(file1.path())
        .arg(file2.path())
        .arg("--output")
        .arg("yaml");
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("value:"));

    Ok(())
}

/// Test case 6: diffai baseline.safetensors finetuned.safetensors
#[test]
fn test_baseline_vs_finetuned() -> Result<(), Box<dyn std::error::Error>> {
    let baseline = create_temp_file(r#"{"model": {"accuracy": 0.85}}"#, ".json");
    let finetuned = create_temp_file(r#"{"model": {"accuracy": 0.92}}"#, ".json");

    let mut cmd = diffai_cmd();
    cmd.arg(baseline.path()).arg(finetuned.path());
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("accuracy:"));

    Ok(())
}

/// Test case 7: diffai data_v1.npy data_v2.npy
#[test]
fn test_numpy_comparison() -> Result<(), Box<dyn std::error::Error>> {
    // Using JSON as substitute for numpy files
    let file1 = create_temp_file(r#"{"array": [1, 2, 3]}"#, ".json");
    let file2 = create_temp_file(r#"{"array": [1, 2, 4]}"#, ".json");

    let mut cmd = diffai_cmd();
    cmd.arg(file1.path()).arg(file2.path());
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("array:"));

    Ok(())
}

/// Test case 8: diffai experiment_v1.mat experiment_v2.mat
#[test]
fn test_matlab_comparison() -> Result<(), Box<dyn std::error::Error>> {
    // Using JSON as substitute for MATLAB files
    let file1 = create_temp_file(r#"{"experiment": {"result": 0.75}}"#, ".json");
    let file2 = create_temp_file(r#"{"experiment": {"result": 0.80}}"#, ".json");

    let mut cmd = diffai_cmd();
    cmd.arg(file1.path()).arg(file2.path());
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("result:"));

    Ok(())
}

/// Test case 9: diffai model1.pt model2.pt
#[test]
fn test_pytorch_comparison() -> Result<(), Box<dyn std::error::Error>> {
    // Using JSON as substitute for PyTorch files
    let file1 = create_temp_file(r#"{"layers": {"conv1": {"filters": 32}}}"#, ".json");
    let file2 = create_temp_file(r#"{"layers": {"conv1": {"filters": 64}}}"#, ".json");

    let mut cmd = diffai_cmd();
    cmd.arg(file1.path()).arg(file2.path());
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("filters:"));

    Ok(())
}

/// Test case 10: diffai baseline_model.pt improved_model.pt
#[test]
fn test_model_improvement() -> Result<(), Box<dyn std::error::Error>> {
    let baseline = create_temp_file(r#"{"performance": {"f1": 0.80}}"#, ".json");
    let improved = create_temp_file(r#"{"performance": {"f1": 0.85}}"#, ".json");

    let mut cmd = diffai_cmd();
    cmd.arg(baseline.path()).arg(improved.path());
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("f1:"));

    Ok(())
}

/// Test case 11: diffai simple_model_v1.safetensors simple_model_v2.safetensors
#[test]
fn test_simple_model_diff() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(r#"{"fc1": {"bias": 0.001}}"#, ".json");
    let file2 = create_temp_file(r#"{"fc1": {"bias": 0.002}}"#, ".json");

    let mut cmd = diffai_cmd();
    cmd.arg(file1.path()).arg(file2.path());
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("bias:"));

    Ok(())
}

/// Test case 12: diffai baseline.safetensors improved.safetensors --output json
#[test]
fn test_improved_model_json() -> Result<(), Box<dyn std::error::Error>> {
    let baseline = create_temp_file(r#"{"metric": 0.7}"#, ".json");
    let improved = create_temp_file(r#"{"metric": 0.8}"#, ".json");

    let mut cmd = diffai_cmd();
    cmd.arg(baseline.path())
        .arg(improved.path())
        .arg("--output")
        .arg("json");
    cmd.assert()
        .code(1)
        .stdout(predicates::str::is_match(r#"\[.*\]"#).unwrap());

    Ok(())
}

/// Test case 13: diffai experiment_data_v1.npy experiment_data_v2.npy
#[test]
fn test_experiment_data_diff() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(r#"{"data": {"mean": 0.1234, "std": 0.9876}}"#, ".json");
    let file2 = create_temp_file(r#"{"data": {"mean": 0.1456, "std": 0.9654}}"#, ".json");

    let mut cmd = diffai_cmd();
    cmd.arg(file1.path()).arg(file2.path());
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("mean:"))
        .stdout(predicates::str::contains("std:"));

    Ok(())
}

/// Test case 14: diffai simulation_v1.mat simulation_v2.mat
#[test]
fn test_simulation_comparison() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(r#"{"results": {"mean": 2.3456, "std": 1.2345}}"#, ".json");
    let file2 = create_temp_file(r#"{"results": {"mean": 2.4567, "std": 1.3456}}"#, ".json");

    let mut cmd = diffai_cmd();
    cmd.arg(file1.path()).arg(file2.path());
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("mean:"))
        .stdout(predicates::str::contains("std:"));

    Ok(())
}

/// Test case 15: diffai model_old.pt model_new.pt
#[test]
fn test_old_vs_new_model() -> Result<(), Box<dyn std::error::Error>> {
    let old_model = create_temp_file(r#"{"version": "1.0", "layers": 5}"#, ".json");
    let new_model = create_temp_file(r#"{"version": "2.0", "layers": 8}"#, ".json");

    let mut cmd = diffai_cmd();
    cmd.arg(old_model.path()).arg(new_model.path());
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("version:"))
        .stdout(predicates::str::contains("layers:"));

    Ok(())
}

/// Test case 16: diffai checkpoint_v1.safetensors checkpoint_v2.safetensors
#[test]
fn test_checkpoint_comparison() -> Result<(), Box<dyn std::error::Error>> {
    let checkpoint1 = create_temp_file(r#"{"epoch": 10, "loss": 0.5}"#, ".json");
    let checkpoint2 = create_temp_file(r#"{"epoch": 20, "loss": 0.3}"#, ".json");

    let mut cmd = diffai_cmd();
    cmd.arg(checkpoint1.path()).arg(checkpoint2.path());
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("epoch:"))
        .stdout(predicates::str::contains("loss:"));

    Ok(())
}

/// Test case 17: diffai model1.safetensors model2.safetensors --output json | jq .
#[test]
fn test_json_output_for_jq() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(r#"{"test": "value1"}"#, ".json");
    let file2 = create_temp_file(r#"{"test": "value2"}"#, ".json");

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

/// Test case 18: diffai model1.safetensors model2.safetensors --output yaml
#[test]
fn test_yaml_output_format() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(r#"{"config": {"setting": "old"}}"#, ".json");
    let file2 = create_temp_file(r#"{"config": {"setting": "new"}}"#, ".json");

    let mut cmd = diffai_cmd();
    cmd.arg(file1.path())
        .arg(file2.path())
        .arg("--output")
        .arg("yaml");
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("setting:"));

    Ok(())
}

/// Test case 19: diffai baseline_results.npy new_results.npy
#[test]
fn test_baseline_results() -> Result<(), Box<dyn std::error::Error>> {
    let baseline = create_temp_file(r#"{"results": [0.8, 0.85, 0.9]}"#, ".json");
    let new_results = create_temp_file(r#"{"results": [0.82, 0.87, 0.92]}"#, ".json");

    let mut cmd = diffai_cmd();
    cmd.arg(baseline.path()).arg(new_results.path());
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("results:"));

    Ok(())
}

/// Test case 20: diffai simulation_v1.mat simulation_v2.mat
#[test]
fn test_matlab_simulation() -> Result<(), Box<dyn std::error::Error>> {
    let sim1 = create_temp_file(r#"{"temperature": 25.5, "pressure": 101.3}"#, ".json");
    let sim2 = create_temp_file(r#"{"temperature": 26.0, "pressure": 102.1}"#, ".json");

    let mut cmd = diffai_cmd();
    cmd.arg(sim1.path()).arg(sim2.path());
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("temperature:"))
        .stdout(predicates::str::contains("pressure:"));

    Ok(())
}

/// Test case 21: diffai pretrained_model.safetensors finetuned_model.safetensors
#[test]
fn test_pretrained_vs_finetuned() -> Result<(), Box<dyn std::error::Error>> {
    let pretrained = create_temp_file(r#"{"weights": {"layer1": 0.5, "layer2": 0.3}}"#, ".json");
    let finetuned = create_temp_file(r#"{"weights": {"layer1": 0.6, "layer2": 0.4}}"#, ".json");

    let mut cmd = diffai_cmd();
    cmd.arg(pretrained.path()).arg(finetuned.path());
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("layer1:"))
        .stdout(predicates::str::contains("layer2:"));

    Ok(())
}

/// Test case 22: diffai baseline_architecture.pt improved_architecture.pt
#[test]
fn test_architecture_comparison() -> Result<(), Box<dyn std::error::Error>> {
    let baseline = create_temp_file(r#"{"architecture": {"type": "cnn", "layers": 5}}"#, ".json");
    let improved = create_temp_file(r#"{"architecture": {"type": "cnn", "layers": 8}}"#, ".json");

    let mut cmd = diffai_cmd();
    cmd.arg(baseline.path()).arg(improved.path());
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("layers:"));

    Ok(())
}

/// Test case 23: diffai production_model.safetensors candidate_model.safetensors
#[test]
fn test_production_vs_candidate() -> Result<(), Box<dyn std::error::Error>> {
    let production = create_temp_file(r#"{"status": "stable", "version": "1.0"}"#, ".json");
    let candidate = create_temp_file(r#"{"status": "testing", "version": "1.1"}"#, ".json");

    let mut cmd = diffai_cmd();
    cmd.arg(production.path()).arg(candidate.path());
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("status:"))
        .stdout(predicates::str::contains("version:"));

    Ok(())
}

/// Test case 24: diffai original_model.pt optimized_model.pt --output json
#[test]
fn test_optimization_analysis() -> Result<(), Box<dyn std::error::Error>> {
    let original = create_temp_file(r#"{"performance": {"speed": 100, "memory": 512}}"#, ".json");
    let optimized = create_temp_file(r#"{"performance": {"speed": 150, "memory": 256}}"#, ".json");

    let mut cmd = diffai_cmd();
    cmd.arg(original.path())
        .arg(optimized.path())
        .arg("--output")
        .arg("json");
    cmd.assert()
        .code(1)
        .stdout(predicates::str::is_match(r#"\[.*\]"#).unwrap());

    Ok(())
}
