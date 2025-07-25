use assert_cmd::prelude::*;
use std::process::Command;

// Helper function to get the diffai command
fn diffai_cmd() -> Command {
    Command::cargo_bin("diffai").expect("Failed to find diffai binary")
}

/// Test basic ML model comparison
/// Corresponds to: docs/examples/test-results/ml_basic_comparison.md
#[test]
fn test_ml_analysis_default() -> Result<(), Box<dyn std::error::Error>> {
    // Test that ML analysis runs by default for ML model files
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/checkpoint_epoch_0.safetensors")
        .arg("../tests/fixtures/ml_models/checkpoint_epoch_10.safetensors");

    let output = cmd.output()?;

    // Should process successfully with real ML files
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Check that full ML analysis was performed (should contain multiple analysis types)
    assert!(stdout.contains("mean=") || stdout.contains("std="));
    assert!(
        stdout.contains("convergence_analysis")
            || stdout.contains("anomaly_detection")
            || stdout.contains("gradient_analysis")
    );

    Ok(())
}

/// Test full ML analysis by default
#[test]
fn test_full_ml_analysis_by_default() -> Result<(), Box<dyn std::error::Error>> {
    // Test that full ML analysis runs by default (no flags needed)
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/simple_base.safetensors")
        .arg("../tests/fixtures/ml_models/simple_modified.safetensors");

    let output = cmd.output()?;

    // Should process successfully with default full analysis
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Check that multiple analyses were performed (based on actual output format)
    assert!(stdout.contains("mean=") || stdout.contains("std="));
    assert!(
        stdout.contains("memory_analysis")
            || stdout.contains("inference_speed")
            || stdout.contains("🧠")
            || stdout.contains("⚡")
    );
    assert!(stdout.contains("fc1.") || stdout.contains("fc2.") || stdout.contains("fc3."));
    // Check for additional analysis types that should be included
    assert!(
        stdout.contains("convergence_analysis")
            || stdout.contains("anomaly_detection")
            || stdout.contains("gradient_analysis")
    );

    Ok(())
}

/// Test JSON output with full ML analysis
#[test]
fn test_json_output_with_full_ml_analysis() -> Result<(), Box<dyn std::error::Error>> {
    // Test JSON output format with full ML analysis (default)
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/simple_base.safetensors")
        .arg("../tests/fixtures/ml_models/simple_modified.safetensors")
        .arg("--output")
        .arg("json");

    let output = cmd.output()?;

    // Should process successfully
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should output valid JSON
    assert!(stdout.starts_with('[') && stdout.trim_end().ends_with(']'));
    // Should contain ML analysis results (TensorStatsChanged and multiple other analyses)
    assert!(stdout.contains("TensorStatsChanged") || stdout.contains("MemoryAnalysis"));
    // Should contain multiple analysis types in JSON output
    assert!(
        stdout.contains("ConvergenceAnalysis")
            || stdout.contains("AnomalyDetection")
            || stdout.contains("GradientAnalysis")
    );

    Ok(())
}
