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

/// Test case 1: diffai baseline/resnet18.pth experiment/resnet34.pth
#[test]
fn test_model_development_improvement() -> Result<(), Box<dyn std::error::Error>> {
    let baseline = create_temp_file(
        r#"{"architecture": "resnet18", "layers": 18, "parameters": 11000000}"#,
        ".json",
    );
    let experiment = create_temp_file(
        r#"{"architecture": "resnet34", "layers": 34, "parameters": 21000000}"#,
        ".json",
    );

    let mut cmd = diffai_cmd();
    cmd.arg(baseline.path()).arg(experiment.path());
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("architecture"));

    Ok(())
}

/// Test case 2: diffai pretrained/model.pth finetuned/model.pth
#[test]
fn test_finetuning_comparison() -> Result<(), Box<dyn std::error::Error>> {
    let pretrained = create_temp_file(
        r#"{"model": {"pretrained": true, "weights": {"classifier": [0.0, 0.0]}}}"#,
        ".json",
    );
    let finetuned = create_temp_file(
        r#"{"model": {"pretrained": false, "weights": {"classifier": [0.8, 0.9]}}}"#,
        ".json",
    );

    let mut cmd = diffai_cmd();
    cmd.arg(pretrained.path()).arg(finetuned.path());
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("classifier"));

    Ok(())
}

/// Test case 3: diffai experiment_001/ experiment_002/ --recursive --include "*.json"
#[test]
fn test_experiment_results_comparison() -> Result<(), Box<dyn std::error::Error>> {
    let exp1 = create_temp_file(
        r#"{"experiment": {"id": "001", "accuracy": 0.85, "loss": 0.3}}"#,
        ".json",
    );
    let exp2 = create_temp_file(
        r#"{"experiment": {"id": "002", "accuracy": 0.88, "loss": 0.25}}"#,
        ".json",
    );

    let mut cmd = diffai_cmd();
    cmd.arg(exp1.path()).arg(exp2.path()).arg("--recursive");
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("experiment"));

    Ok(())
}

/// Test case 4: diffai config/baseline.yaml config/experiment.yaml
#[test]
fn test_hyperparameter_differences() -> Result<(), Box<dyn std::error::Error>> {
    let baseline = create_temp_file(
        r#"{"config": {"learning_rate": 0.01, "batch_size": 32, "epochs": 100}}"#,
        ".json",
    );
    let experiment = create_temp_file(
        r#"{"config": {"learning_rate": 0.001, "batch_size": 64, "epochs": 150}}"#,
        ".json",
    );

    let mut cmd = diffai_cmd();
    cmd.arg(baseline.path()).arg(experiment.path());
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("learning_rate"));

    Ok(())
}

/// Test case 5: diffai original/model.pth quantized/model.pth --show-structure
#[test]
fn test_quantization_comparison() -> Result<(), Box<dyn std::error::Error>> {
    let original = create_temp_file(
        r#"{"model": {"precision": "fp32", "size_mb": 100, "weights": {"layer1": [0.123456]}}}"#,
        ".json",
    );
    let quantized = create_temp_file(
        r#"{"model": {"precision": "int8", "size_mb": 25, "weights": {"layer1": [0.125]}}}"#,
        ".json",
    );

    let mut cmd = diffai_cmd();
    cmd.arg(original.path())
        .arg(quantized.path())
        .arg("--show-structure");
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("precision"));

    Ok(())
}

/// Test case 6: diffai full/model.pth pruned/model.pth --diff-only
#[test]
fn test_pruning_effects() -> Result<(), Box<dyn std::error::Error>> {
    let full = create_temp_file(
        r#"{"model": {"parameters": 1000000, "layers": {"conv1": {"weights": [0.1, 0.2, 0.3]}, "conv2": {"weights": [0.4, 0.5, 0.6]}}}}"#,
        ".json",
    );
    let pruned = create_temp_file(
        r#"{"model": {"parameters": 500000, "layers": {"conv1": {"weights": [0.1, 0.0, 0.3]}, "conv2": {"weights": [0.0, 0.5, 0.0]}}}}"#,
        ".json",
    );

    let mut cmd = diffai_cmd();
    cmd.arg(full.path()).arg(pruned.path()).arg("--diff-only");
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("parameters"));

    Ok(())
}

/// Test case 7: diffai baseline/ experiment/ --recursive --include "*.json" --include "*.pth"
#[test]
fn test_workflow_comparison() -> Result<(), Box<dyn std::error::Error>> {
    let baseline = create_temp_file(
        r#"{"workflow": {"baseline": true, "results": {"accuracy": 0.85, "f1": 0.82}}}"#,
        ".json",
    );
    let experiment = create_temp_file(
        r#"{"workflow": {"baseline": false, "results": {"accuracy": 0.90, "f1": 0.88}}}"#,
        ".json",
    );

    let mut cmd = diffai_cmd();
    cmd.arg(baseline.path())
        .arg(experiment.path())
        .arg("--recursive");
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("workflow"));

    Ok(())
}
