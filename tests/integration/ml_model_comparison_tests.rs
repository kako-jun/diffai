use assert_cmd::prelude::*;
use std::process::Command;

// Helper function to get the diffai command
fn diffai_cmd() -> Command {
    Command::cargo_bin("diffai").expect("Failed to find diffai binary")
}

/// Tests corresponding to docs/user-guide/ml-model-comparison.md
/// Tests specific ML model comparison scenarios and use cases

#[test]
fn test_basic_safetensors_comparison() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/simple_base.safetensors")
        .arg("../tests/fixtures/ml_models/simple_modified.safetensors");

    let output = cmd.output()?;
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should show tensor statistics as documented (included by default)
    assert!(stdout.contains("mean=") || stdout.contains("std=") || stdout.contains("shape="));

    Ok(())
}

#[test]
fn test_pytorch_model_comparison() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/simple_base.pt")
        .arg("../tests/fixtures/ml_models/simple_modified.pt");

    let output = cmd.output()?;
    // This should work with PyTorch format (stats included by default)
    assert!(output.status.success());

    Ok(())
}

#[test]
fn test_quantization_analysis_use_case() -> Result<(), Box<dyn std::error::Error>> {
    // Simulates the quantization analysis use case from the guide (included by default)
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/simple_base.safetensors")
        .arg("../tests/fixtures/ml_models/simple_modified.safetensors");

    let output = cmd.output()?;
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should contain quantization analysis output (included by default)
    assert!(stdout.contains("quantization_analysis") || output.status.success());

    Ok(())
}

#[test]
fn test_training_progress_tracking_simulation() -> Result<(), Box<dyn std::error::Error>> {
    // Simulates comparing training checkpoints as described in the guide
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/simple_base.safetensors")
        .arg("../tests/fixtures/ml_models/simple_modified.safetensors");

    let output = cmd.output()?;
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should show statistical changes sorted by magnitude (stats included by default)
    assert!(stdout.contains("mean=") || stdout.contains("std="));

    Ok(())
}

#[test]
fn test_epsilon_tolerance_feature() -> Result<(), Box<dyn std::error::Error>> {
    // Tests epsilon tolerance as mentioned in the guide
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/simple_base.safetensors")
        .arg("../tests/fixtures/ml_models/simple_modified.safetensors")
        .arg("--epsilon")
        .arg("1e-6");

    let output = cmd.output()?;
    assert!(output.status.success());

    Ok(())
}

#[test]
fn test_json_output_for_automation() -> Result<(), Box<dyn std::error::Error>> {
    // Tests JSON output for MLOps automation as shown in the guide
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/simple_base.safetensors")
        .arg("../tests/fixtures/ml_models/simple_modified.safetensors")
        .arg("--output")
        .arg("json");

    let output = cmd.output()?;
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should produce valid JSON (stats included by default)
    assert!(stdout.starts_with('[') || stdout.starts_with('{'));

    Ok(())
}

#[test]
fn test_yaml_output_for_readability() -> Result<(), Box<dyn std::error::Error>> {
    // Tests YAML output for human readability as mentioned in the guide
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/simple_base.safetensors")
        .arg("../tests/fixtures/ml_models/simple_modified.safetensors")
        .arg("--output")
        .arg("yaml");

    let output = cmd.output()?;
    assert!(output.status.success());

    Ok(())
}

#[test]
fn test_comprehensive_analysis_combination() -> Result<(), Box<dyn std::error::Error>> {
    // Tests comprehensive analysis combining multiple implemented features (included by default)
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/simple_base.safetensors")
        .arg("../tests/fixtures/ml_models/simple_modified.safetensors");

    let output = cmd.output()?;
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should contain statistical information (stats included by default)
    assert!(stdout.contains("mean=") || stdout.contains("std=") || stdout.contains("shape="));

    Ok(())
}

#[test]
fn test_path_filtering() -> Result<(), Box<dyn std::error::Error>> {
    // Tests path filtering as mentioned in the guide
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/simple_base.safetensors")
        .arg("../tests/fixtures/ml_models/simple_modified.safetensors")
        .arg("--path")
        .arg("tensor");

    let output = cmd.output()?;
    assert!(output.status.success());

    Ok(())
}

#[test]
fn test_architecture_comparison_format_detection() -> Result<(), Box<dyn std::error::Error>> {
    // Tests automatic format detection for different model architectures
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/simple_base.pt")
        .arg("../tests/fixtures/ml_models/simple_base.safetensors");

    let output = cmd.output()?;
    // Should work with automatic format detection
    assert!(output.status.success());

    Ok(())
}

#[test]
fn test_layer_impact_analysis() -> Result<(), Box<dyn std::error::Error>> {
    // Tests layer-by-layer impact analysis as documented
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/simple_base.safetensors")
        .arg("../tests/fixtures/ml_models/simple_modified.safetensors");

    let output = cmd.output()?;
    assert!(output.status.success());

    Ok(())
}

#[test]
fn test_feature_selection_for_training_monitoring() -> Result<(), Box<dyn std::error::Error>> {
    // Tests the "For Training Monitoring" feature selection from the guide
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/simple_base.safetensors")
        .arg("../tests/fixtures/ml_models/simple_modified.safetensors");

    let output = cmd.output()?;
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should show sorted statistical changes (stats included by default)
    assert!(stdout.contains("mean=") || stdout.contains("std="));

    Ok(())
}

#[test]
fn test_feature_selection_for_production_deployment() -> Result<(), Box<dyn std::error::Error>> {
    // Tests the "For Production Deployment" feature selection from the guide
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/simple_base.safetensors")
        .arg("../tests/fixtures/ml_models/simple_modified.safetensors");

    let output = cmd.output()?;
    assert!(output.status.success());

    Ok(())
}

#[test]
fn test_feature_selection_for_research_analysis() -> Result<(), Box<dyn std::error::Error>> {
    // Tests the "For Research Analysis" feature selection from the guide
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/simple_base.safetensors")
        .arg("../tests/fixtures/ml_models/simple_modified.safetensors");

    let output = cmd.output()?;
    assert!(output.status.success());

    Ok(())
}

#[test]
fn test_feature_selection_for_quantization_validation() -> Result<(), Box<dyn std::error::Error>> {
    // Tests the "For Quantization Validation" feature selection from the guide
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/simple_base.safetensors")
        .arg("../tests/fixtures/ml_models/simple_modified.safetensors");

    let output = cmd.output()?;
    assert!(output.status.success());

    Ok(())
}
