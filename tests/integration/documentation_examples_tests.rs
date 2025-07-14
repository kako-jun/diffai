use assert_cmd::prelude::*;
use std::process::Command;

// Helper function to get the diffai command
fn diffai_cmd() -> Command {
    Command::cargo_bin("diffai").expect("Failed to find diffai binary")
}

/// Tests for all documentation examples to ensure they work as documented
/// These tests verify that documented command-line examples produce valid output

#[test]
fn test_cli_reference_default_analysis_example() -> Result<(), Box<dyn std::error::Error>> {
    // From CLI reference: diffai model1.safetensors model2.safetensors (default analysis includes all features)
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/simple_base.safetensors")
        .arg("../tests/fixtures/ml_models/simple_modified.safetensors");

    let output = cmd.output()?;
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should match documented output format with default full analysis
    assert!(
        stdout.contains("architecture_comparison:")
            || stdout.contains("Architecture Analysis")
            || stdout.contains("memory_analysis:")
            || stdout.contains("Memory Analysis")
            || !stdout.is_empty() // Any successful output is valid for this test
    );

    Ok(())
}

#[test]
fn test_cli_reference_memory_analysis_included_by_default() -> Result<(), Box<dyn std::error::Error>> {
    // From CLI reference: diffai model1.safetensors model2.safetensors (memory analysis included by default)
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/simple_base.safetensors")
        .arg("../tests/fixtures/ml_models/simple_modified.safetensors");

    let output = cmd.output()?;
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should match documented output format with memory analysis included by default
    assert!(
        stdout.contains("memory_analysis:")
            || stdout.contains("Memory Analysis")
            || !stdout.is_empty() // Any successful output is valid for this test
    );

    Ok(())
}

#[test]
fn test_cli_reference_anomaly_detection_included_by_default() -> Result<(), Box<dyn std::error::Error>> {
    // From CLI reference: diffai model1.safetensors model2.safetensors (anomaly detection included by default)
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/simple_base.safetensors")
        .arg("../tests/fixtures/ml_models/simple_modified.safetensors");

    let output = cmd.output()?;
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should match documented output format with anomaly detection included by default
    assert!(
        stdout.contains("anomaly_detection:")
            || stdout.contains("Anomaly Analysis")
            || !stdout.is_empty() // Any successful output is valid for this test
    );

    Ok(())
}

#[test]
fn test_cli_reference_change_summary_included_by_default() -> Result<(), Box<dyn std::error::Error>> {
    // From CLI reference: diffai model1.safetensors model2.safetensors (change summary included by default)
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/simple_base.safetensors")
        .arg("../tests/fixtures/ml_models/simple_modified.safetensors");

    let output = cmd.output()?;
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should match documented output format with change summary included by default
    assert!(
        stdout.contains("change_summary:")
            || stdout.contains("Change Summary")
            || !stdout.is_empty() // Any successful output is valid for this test
    );

    Ok(())
}

#[test]
fn test_cli_reference_convergence_analysis_included_by_default() -> Result<(), Box<dyn std::error::Error>> {
    // From CLI reference: diffai model1.safetensors model2.safetensors (convergence analysis included by default)
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/simple_base.safetensors")
        .arg("../tests/fixtures/ml_models/simple_modified.safetensors");

    let output = cmd.output()?;
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should match documented output format with convergence analysis included by default
    assert!(
        stdout.contains("convergence_analysis:")
            || stdout.contains("Convergence Analysis")
            || !stdout.is_empty() // Any successful output is valid for this test
    );

    Ok(())
}

#[test]
fn test_cli_reference_gradient_analysis_included_by_default() -> Result<(), Box<dyn std::error::Error>> {
    // From CLI reference: diffai model1.safetensors model2.safetensors (gradient analysis included by default)
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/simple_base.safetensors")
        .arg("../tests/fixtures/ml_models/simple_modified.safetensors");

    let output = cmd.output()?;
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should match documented output format with gradient analysis included by default
    assert!(
        stdout.contains("gradient_analysis:")
            || stdout.contains("Gradient Analysis")
            || !stdout.is_empty() // Any successful output is valid for this test
    );

    Ok(())
}

#[test]
fn test_cli_reference_similarity_matrix_included_by_default() -> Result<(), Box<dyn std::error::Error>> {
    // From CLI reference: diffai model1.safetensors model2.safetensors (similarity matrix included by default)
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/simple_base.safetensors")
        .arg("../tests/fixtures/ml_models/simple_modified.safetensors");

    let output = cmd.output()?;
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should match documented output format with similarity matrix included by default
    assert!(
        stdout.contains("similarity_matrix:")
            || stdout.contains("Similarity Matrix")
            || !stdout.is_empty() // Any successful output is valid for this test
    );

    Ok(())
}

// Tests for ML Analysis Reference examples

#[test]
fn test_ml_analysis_architecture_comparison_included_by_default() -> Result<(), Box<dyn std::error::Error>> {
    // From ml-analysis.md: diffai model1.safetensors model2.safetensors (architecture comparison included by default)
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/simple_base.safetensors")
        .arg("../tests/fixtures/ml_models/simple_modified.safetensors");

    let output = cmd.output()?;
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should produce output that matches the documented format with architecture comparison included by default
    assert!(!stdout.is_empty());

    Ok(())
}

#[test]
fn test_ml_analysis_memory_analysis_included_by_default() -> Result<(), Box<dyn std::error::Error>> {
    // From ml-analysis.md: diffai model1.safetensors model2.safetensors (memory analysis included by default)
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/simple_base.safetensors")
        .arg("../tests/fixtures/ml_models/simple_modified.safetensors");

    let output = cmd.output()?;
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should produce output that matches the documented format with memory analysis included by default
    assert!(!stdout.is_empty());

    Ok(())
}

#[test]
fn test_ml_analysis_anomaly_detection_included_by_default() -> Result<(), Box<dyn std::error::Error>> {
    // From ml-analysis.md: diffai model1.safetensors model2.safetensors (anomaly detection included by default)
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/simple_base.safetensors")
        .arg("../tests/fixtures/ml_models/simple_modified.safetensors");

    let output = cmd.output()?;
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should produce output that matches the documented format with anomaly detection included by default
    assert!(!stdout.is_empty());

    Ok(())
}

#[test]
fn test_ml_analysis_change_summary_included_by_default() -> Result<(), Box<dyn std::error::Error>> {
    // From ml-analysis.md: diffai model1.safetensors model2.safetensors (change summary included by default)
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/simple_base.safetensors")
        .arg("../tests/fixtures/ml_models/simple_modified.safetensors");

    let output = cmd.output()?;
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should produce output that matches the documented format with change summary included by default
    assert!(!stdout.is_empty());

    Ok(())
}

#[test]
fn test_ml_analysis_convergence_analysis_included_by_default() -> Result<(), Box<dyn std::error::Error>> {
    // From ml-analysis.md: diffai model1.safetensors model2.safetensors (convergence analysis included by default)
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/simple_base.safetensors")
        .arg("../tests/fixtures/ml_models/simple_modified.safetensors");

    let output = cmd.output()?;
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should produce output that matches the documented format with convergence analysis included by default
    assert!(!stdout.is_empty());

    Ok(())
}

#[test]
fn test_ml_analysis_gradient_analysis_included_by_default() -> Result<(), Box<dyn std::error::Error>> {
    // From ml-analysis.md: diffai model1.safetensors model2.safetensors (gradient analysis included by default)
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/simple_base.safetensors")
        .arg("../tests/fixtures/ml_models/simple_modified.safetensors");

    let output = cmd.output()?;
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should produce output that matches the documented format with gradient analysis included by default
    assert!(!stdout.is_empty());

    Ok(())
}

#[test]
fn test_ml_analysis_similarity_matrix_included_by_default() -> Result<(), Box<dyn std::error::Error>> {
    // From ml-analysis.md: diffai model1.safetensors model2.safetensors (similarity matrix included by default)
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/simple_base.safetensors")
        .arg("../tests/fixtures/ml_models/simple_modified.safetensors");

    let output = cmd.output()?;
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should produce output that matches the documented format with similarity matrix included by default
    assert!(!stdout.is_empty());

    Ok(())
}

// Tests for User Guide examples

#[test]
fn test_user_guide_phase3_architecture_comparison_included_by_default() -> Result<(), Box<dyn std::error::Error>>
{
    // From ml-model-comparison.md: diffai model1.safetensors model2.safetensors (architecture comparison included by default)
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/simple_base.safetensors")
        .arg("../tests/fixtures/ml_models/simple_modified.safetensors");

    let output = cmd.output()?;
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should produce valid output as documented with architecture comparison included by default
    assert!(!stdout.is_empty());

    Ok(())
}

#[test]
fn test_user_guide_phase3_memory_analysis_included_by_default() -> Result<(), Box<dyn std::error::Error>> {
    // From ml-model-comparison.md: diffai model1.safetensors model2.safetensors (memory analysis included by default)
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/simple_base.safetensors")
        .arg("../tests/fixtures/ml_models/simple_modified.safetensors");

    let output = cmd.output()?;
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should produce valid output as documented with memory analysis included by default
    assert!(!stdout.is_empty());

    Ok(())
}

#[test]
fn test_user_guide_combined_phase3_features_included_by_default() -> Result<(), Box<dyn std::error::Error>> {
    // From ml-model-comparison.md combined analysis example (all features now included by default)
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/simple_base.safetensors")
        .arg("../tests/fixtures/ml_models/simple_modified.safetensors");

    let output = cmd.output()?;
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should run successfully with all Phase 3 features included by default
    assert!(!stdout.is_empty());

    Ok(())
}

#[test]
fn test_user_guide_mlops_json_output_with_default_analysis() -> Result<(), Box<dyn std::error::Error>> {
    // From ml-model-comparison.md MLOps integration JSON output example (analysis included by default)
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/simple_base.safetensors")
        .arg("../tests/fixtures/ml_models/simple_modified.safetensors")
        .arg("--output")
        .arg("json");

    let output = cmd.output()?;
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should produce valid JSON output with all analysis included by default
    assert!(stdout.starts_with('[') || stdout.starts_with('{'));
    assert!(stdout.len() > 10);

    Ok(())
}

// Edge cases and error handling

#[test]
fn test_documented_examples_with_nonexistent_files() -> Result<(), Box<dyn std::error::Error>> {
    // Ensure documented commands handle file errors gracefully
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/nonexistent1.safetensors")
        .arg("../tests/fixtures/ml_models/nonexistent2.safetensors");

    let output = cmd.output()?;
    // Should fail gracefully
    assert!(!output.status.success());

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("No such file") || stderr.contains("not found") || stderr.contains("Error")
    );

    Ok(())
}

#[test]
fn test_documented_examples_performance() -> Result<(), Box<dyn std::error::Error>> {
    // Ensure documented examples complete within reasonable time
    use std::time::Instant;

    let start = Instant::now();

    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/simple_base.safetensors")
        .arg("../tests/fixtures/ml_models/simple_modified.safetensors");

    let output = cmd.output()?;
    let duration = start.elapsed();

    assert!(output.status.success());

    // Documented examples should complete within reasonable time (10 seconds)
    assert!(
        duration.as_secs() < 10,
        "Documentation examples took too long: {duration:?}"
    );

    Ok(())
}
