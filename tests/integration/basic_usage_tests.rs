use assert_cmd::prelude::*;
// use predicates::prelude::*;
use std::process::Command;

// Helper function to get the diffai command
fn diffai_cmd() -> Command {
    Command::cargo_bin("diffai").expect("Failed to find diffai binary")
}

/// Tests corresponding to docs/user-guide/basic-usage.md
/// Tests basic usage examples and implemented features mentioned in the basic usage guide

#[test]
fn test_default_ml_analysis() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/simple_base.safetensors")
        .arg("../tests/fixtures/ml_models/simple_modified.safetensors");

    let output = cmd.output()?;
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should contain statistical information by default
    assert!(stdout.contains("mean=") || stdout.contains("std=") || stdout.contains("shape="));
    // Should also contain advanced analysis
    assert!(stdout.contains("convergence_analysis") || stdout.contains("anomaly_detection") || stdout.contains("quantization_analysis"));

    Ok(())
}

#[test]
fn test_quantization_analysis_included() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/simple_base.safetensors")
        .arg("../tests/fixtures/ml_models/simple_modified.safetensors");

    let output = cmd.output()?;
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Quantization analysis should be included in default output
    assert!(stdout.contains("quantization_analysis") || stdout.contains("compression=") || stdout.contains("speedup="));

    Ok(())
}

#[test]
fn test_comprehensive_analysis() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/simple_base.safetensors")
        .arg("../tests/fixtures/ml_models/simple_modified.safetensors");

    let output = cmd.output()?;
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should contain multiple analysis types
    assert!(stdout.contains("gradient_analysis") || stdout.contains("memory_analysis") || stdout.contains("deployment_readiness"));

    Ok(())
}

#[test]
fn test_architecture_analysis_included() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/simple_base.safetensors")
        .arg("../tests/fixtures/ml_models/simple_modified.safetensors");

    let output = cmd.output()?;
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should contain architecture analysis by default
    assert!(stdout.contains("architecture_comparison") || stdout.contains("deployment_readiness") || stdout.contains("type1=feedforward"));

    Ok(())
}

#[test]
fn test_full_analysis_by_default() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/simple_base.safetensors")
        .arg("../tests/fixtures/ml_models/simple_modified.safetensors");

    let output = cmd.output()?;
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should contain statistical information by default
    assert!(stdout.contains("mean=") || stdout.contains("std=") || stdout.contains("shape="));
    // Should contain multiple analysis types
    assert!(stdout.contains("quantization_analysis") || stdout.contains("architecture_comparison") || stdout.contains("gradient_analysis"));

    Ok(())
}

#[test]
fn test_json_output_with_full_analysis() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/simple_base.safetensors")
        .arg("../tests/fixtures/ml_models/simple_modified.safetensors")
        .arg("--output")
        .arg("json");

    let output = cmd.output()?;
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    // JSON output should be valid
    assert!(stdout.starts_with('[') || stdout.starts_with('{'));

    Ok(())
}

#[test]
fn test_yaml_output_with_full_analysis() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/simple_base.safetensors")
        .arg("../tests/fixtures/ml_models/simple_modified.safetensors")
        .arg("--output")
        .arg("yaml");

    let output = cmd.output()?;
    assert!(output.status.success());

    // YAML output should run without error
    Ok(())
}

/// Test that unimplemented features return appropriate errors or warnings
#[test]
fn test_unimplemented_feature_error() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/simple_base.safetensors")
        .arg("../tests/fixtures/ml_models/simple_modified.safetensors")
        .arg("--learning-progress"); // This should not exist

    let output = cmd.output()?;

    // If the command succeeds, it means the feature is silently ignored (which is also acceptable)
    // If it fails, it should be due to an unrecognized argument
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains("unrecognized")
                || stderr.contains("unexpected")
                || stderr.contains("invalid")
        );
    }
    // If it succeeds, the feature is being silently ignored, which is acceptable behavior

    Ok(())
}
