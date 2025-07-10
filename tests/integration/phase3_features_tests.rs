use assert_cmd::prelude::*;
use std::process::Command;

// Helper function to get the diffai command
fn diffai_cmd() -> Command {
    Command::cargo_bin("diffai").expect("Failed to find diffai binary")
}

/// Tests for Phase 3 features (TDD approach - these will initially fail)
/// These tests define the expected behavior before implementation

// Phase 3A: Core Features

#[test]
fn test_architecture_comparison_feature() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/simple_base.safetensors")
        .arg("../tests/fixtures/ml_models/simple_modified.safetensors")
        .arg("--architecture-comparison");

    let output = cmd.output()?;
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Expected output format: architecture comparison analysis
    assert!(
        stdout.contains("architecture_comparison:") || 
        stdout.contains("Architecture Analysis") ||
        stdout.contains("model_type") ||
        stdout.contains("parameter_count") ||
        stdout.contains("layer_structure")
    );

    Ok(())
}

#[test]
fn test_memory_analysis_feature() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/simple_base.safetensors")
        .arg("../tests/fixtures/ml_models/simple_modified.safetensors")
        .arg("--memory-analysis");

    let output = cmd.output()?;
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Expected output format: memory usage analysis
    assert!(
        stdout.contains("memory_analysis:") ||
        stdout.contains("Memory Analysis") ||
        stdout.contains("memory_usage") ||
        stdout.contains("size_diff") ||
        stdout.contains("efficiency")
    );

    Ok(())
}

#[test]
fn test_anomaly_detection_feature() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/simple_base.safetensors")
        .arg("../tests/fixtures/ml_models/simple_modified.safetensors")
        .arg("--anomaly-detection");

    let output = cmd.output()?;
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Expected output format: anomaly detection results
    assert!(
        stdout.contains("anomaly_detection:") ||
        stdout.contains("Anomaly Analysis") ||
        stdout.contains("anomalies_found") ||
        stdout.contains("suspicious_patterns") ||
        stdout.contains("normal") ||
        stdout.contains("outliers")
    );

    Ok(())
}

#[test]
fn test_change_summary_feature() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/simple_base.safetensors")
        .arg("../tests/fixtures/ml_models/simple_modified.safetensors")
        .arg("--change-summary");

    let output = cmd.output()?;
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Expected output format: comprehensive change summary
    assert!(
        stdout.contains("change_summary:") ||
        stdout.contains("Change Summary") ||
        stdout.contains("total_changes") ||
        stdout.contains("significant_changes") ||
        stdout.contains("summary")
    );

    Ok(())
}

// Phase 3B: Advanced Analysis

#[test]
fn test_convergence_analysis_feature() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/simple_base.safetensors")
        .arg("../tests/fixtures/ml_models/simple_modified.safetensors")
        .arg("--convergence-analysis");

    let output = cmd.output()?;
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Expected output format: convergence pattern analysis
    assert!(
        stdout.contains("convergence_analysis:") ||
        stdout.contains("Convergence Analysis") ||
        stdout.contains("convergence_status") ||
        stdout.contains("stability") ||
        stdout.contains("trend")
    );

    Ok(())
}

#[test]
fn test_gradient_analysis_feature() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/simple_base.safetensors")
        .arg("../tests/fixtures/ml_models/simple_modified.safetensors")
        .arg("--gradient-analysis");

    let output = cmd.output()?;
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Expected output format: gradient analysis
    assert!(
        stdout.contains("gradient_analysis:") ||
        stdout.contains("Gradient Analysis") ||
        stdout.contains("gradient_flow") ||
        stdout.contains("gradient_norm") ||
        stdout.contains("gradient_health")
    );

    Ok(())
}

#[test]
fn test_similarity_matrix_feature() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/simple_base.safetensors")
        .arg("../tests/fixtures/ml_models/simple_modified.safetensors")
        .arg("--similarity-matrix");

    let output = cmd.output()?;
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Expected output format: similarity matrix
    assert!(
        stdout.contains("similarity_matrix:") ||
        stdout.contains("Similarity Matrix") ||
        stdout.contains("similarity_score") ||
        stdout.contains("correlation") ||
        stdout.contains("matrix_size")
    );

    Ok(())
}

// Combined Phase 3 feature tests

#[test]
fn test_phase3a_core_features_combination() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/simple_base.safetensors")
        .arg("../tests/fixtures/ml_models/simple_modified.safetensors")
        .arg("--architecture-comparison")
        .arg("--memory-analysis")
        .arg("--anomaly-detection")
        .arg("--change-summary");

    let output = cmd.output()?;
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should contain output from all Phase 3A features
    assert!(
        (stdout.contains("architecture_comparison") || stdout.contains("Architecture")) &&
        (stdout.contains("memory_analysis") || stdout.contains("Memory")) &&
        (stdout.contains("anomaly_detection") || stdout.contains("Anomaly")) &&
        (stdout.contains("change_summary") || stdout.contains("Summary"))
    );

    Ok(())
}

#[test]
fn test_phase3b_advanced_features_combination() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/simple_base.safetensors")
        .arg("../tests/fixtures/ml_models/simple_modified.safetensors")
        .arg("--convergence-analysis")
        .arg("--gradient-analysis")
        .arg("--similarity-matrix");

    let output = cmd.output()?;
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should contain output from all Phase 3B features
    assert!(
        (stdout.contains("convergence_analysis") || stdout.contains("Convergence")) &&
        (stdout.contains("gradient_analysis") || stdout.contains("Gradient")) &&
        (stdout.contains("similarity_matrix") || stdout.contains("Similarity"))
    );

    Ok(())
}

#[test]
fn test_all_phase3_features_combined() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/simple_base.safetensors")
        .arg("../tests/fixtures/ml_models/simple_modified.safetensors")
        .arg("--architecture-comparison")
        .arg("--memory-analysis")
        .arg("--anomaly-detection")
        .arg("--change-summary")
        .arg("--convergence-analysis")
        .arg("--gradient-analysis")
        .arg("--similarity-matrix");

    let output = cmd.output()?;
    assert!(output.status.success());

    // Should run without crashing when all Phase 3 features are enabled
    Ok(())
}

// JSON output tests for Phase 3 features

#[test]
fn test_phase3_features_json_output() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/simple_base.safetensors")
        .arg("../tests/fixtures/ml_models/simple_modified.safetensors")
        .arg("--architecture-comparison")
        .arg("--memory-analysis")
        .arg("--output")
        .arg("json");

    let output = cmd.output()?;
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should produce valid JSON with Phase 3 analysis results
    assert!(stdout.starts_with('[') || stdout.starts_with('{'));
    
    // Should contain Phase 3 specific JSON structure
    assert!(
        stdout.contains("architecture_comparison") ||
        stdout.contains("memory_analysis") ||
        stdout.contains("ArchitectureComparison") ||
        stdout.contains("MemoryAnalysis")
    );

    Ok(())
}

// Error handling tests

#[test]
fn test_phase3_features_with_invalid_files() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/nonexistent1.safetensors")
        .arg("../tests/fixtures/ml_models/nonexistent2.safetensors")
        .arg("--architecture-comparison");

    let output = cmd.output()?;
    // Should handle missing files gracefully
    assert!(!output.status.success());

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("No such file") ||
        stderr.contains("not found") ||
        stderr.contains("Error")
    );

    Ok(())
}

// Performance tests for Phase 3 features

#[test]
fn test_phase3_features_performance() -> Result<(), Box<dyn std::error::Error>> {
    use std::time::Instant;

    let start = Instant::now();
    
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/simple_base.safetensors")
        .arg("../tests/fixtures/ml_models/simple_modified.safetensors")
        .arg("--architecture-comparison")
        .arg("--memory-analysis");

    let output = cmd.output()?;
    let duration = start.elapsed();

    assert!(output.status.success());
    
    // Phase 3 features should complete within reasonable time (10 seconds)
    assert!(duration.as_secs() < 10, "Phase 3 analysis took too long: {:?}", duration);

    Ok(())
}

// Edge case tests

#[test]
fn test_phase3_features_with_identical_files() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/simple_base.safetensors")
        .arg("../tests/fixtures/ml_models/simple_base.safetensors") // Same file
        .arg("--architecture-comparison")
        .arg("--change-summary");

    let output = cmd.output()?;
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should handle identical files appropriately
    assert!(
        stdout.contains("no changes") ||
        stdout.contains("identical") ||
        stdout.contains("same") ||
        stdout.len() == 0 // No output for identical files is also acceptable
    );

    Ok(())
}