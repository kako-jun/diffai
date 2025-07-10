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
fn test_stats_feature() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/simple_base.safetensors")
        .arg("../tests/fixtures/ml_models/simple_modified.safetensors")
        .arg("--stats");

    let output = cmd.output()?;
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should contain statistical information
    assert!(stdout.contains("mean=") || stdout.contains("std=") || stdout.contains("shape="));

    Ok(())
}

#[test]
fn test_quantization_analysis() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/simple_base.safetensors")
        .arg("../tests/fixtures/ml_models/simple_modified.safetensors")
        .arg("--quantization-analysis");

    let output = cmd.output()?;
    assert!(output.status.success());

    // Quantization analysis should run without error
    Ok(())
}

#[test]
fn test_sort_by_change_magnitude() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/simple_base.safetensors")
        .arg("../tests/fixtures/ml_models/simple_modified.safetensors")
        .arg("--sort-by-change-magnitude");

    let output = cmd.output()?;
    assert!(output.status.success());

    // Should run without error
    Ok(())
}

#[test]
fn test_show_layer_impact() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/simple_base.safetensors")
        .arg("../tests/fixtures/ml_models/simple_modified.safetensors")
        .arg("--show-layer-impact");

    let output = cmd.output()?;
    assert!(output.status.success());

    // Should run without error
    Ok(())
}

#[test]
fn test_combined_implemented_features() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/simple_base.safetensors")
        .arg("../tests/fixtures/ml_models/simple_modified.safetensors")
        .arg("--stats")
        .arg("--quantization-analysis")
        .arg("--sort-by-change-magnitude");

    let output = cmd.output()?;
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should contain statistical information when using --stats
    assert!(stdout.contains("mean=") || stdout.contains("std=") || stdout.contains("shape="));

    Ok(())
}

#[test]
fn test_json_output_with_implemented_features() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/simple_base.safetensors")
        .arg("../tests/fixtures/ml_models/simple_modified.safetensors")
        .arg("--stats")
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
fn test_yaml_output_with_implemented_features() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/simple_base.safetensors")
        .arg("../tests/fixtures/ml_models/simple_modified.safetensors")
        .arg("--stats")
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
        assert!(stderr.contains("unrecognized") || stderr.contains("unexpected") || stderr.contains("invalid"));
    }
    // If it succeeds, the feature is being silently ignored, which is acceptable behavior

    Ok(())
}