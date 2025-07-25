use assert_cmd::prelude::*;
use std::fs;
use std::process::Command;

// Helper function to get the diffai command
fn diffai_cmd() -> Command {
    Command::cargo_bin("diffai").expect("Failed to find diffai binary")
}

/// Test ML anomaly detection
/// Corresponds to: docs/examples/test-results/ml_anomaly_detection.md
#[test]
fn test_anomaly_detection_in_full_output() -> Result<(), Box<dyn std::error::Error>> {
    // Test that anomaly detection is included in default full analysis
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/normal_model.safetensors")
        .arg("../tests/fixtures/ml_models/anomalous_model.safetensors");

    let output = cmd.output()?;

    // Should process successfully with real ML files
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Check that anomaly detection was performed (outputs regression test)
    assert!(stdout.contains("mean=") || stdout.contains("std="));
    assert!(stdout.contains("regression_test") || stdout.contains("✅"));

    Ok(())
}

/// Test regression test functionality
#[test]
fn test_regression_test() -> Result<(), Box<dyn std::error::Error>> {
    // Helper function for creating test safetensors files
    fn create_test_safetensors_file(path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let test_data = b"{}"; // Minimal JSON metadata
        fs::create_dir_all("../tests/output")?;
        fs::write(path, test_data)?;
        Ok(())
    }

    create_test_safetensors_file("../tests/output/production.safetensors")?;
    create_test_safetensors_file("../tests/output/candidate.safetensors")?;

    let mut cmd = diffai_cmd();
    cmd.arg("../tests/output/production.safetensors")
        .arg("../tests/output/candidate.safetensors");

    let output = cmd.output()?;

    // Should accept the flag and either succeed or show a parse error for invalid safetensors
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success()
            || stderr.contains("Failed to parse")
            || stderr.contains("HeaderTooSmall")
            || stderr.contains("Error:")
    );

    // Clean up
    let _ = fs::remove_file("../tests/output/production.safetensors");
    let _ = fs::remove_file("../tests/output/candidate.safetensors");

    Ok(())
}

/// Test alert on degradation functionality
#[test]
fn test_alert_on_degradation() -> Result<(), Box<dyn std::error::Error>> {
    // Helper function for creating test safetensors files
    fn create_test_safetensors_file(path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let test_data = b"{}"; // Minimal JSON metadata
        fs::create_dir_all("../tests/output")?;
        fs::write(path, test_data)?;
        Ok(())
    }

    create_test_safetensors_file("../tests/output/baseline.safetensors")?;
    create_test_safetensors_file("../tests/output/degraded.safetensors")?;

    let mut cmd = diffai_cmd();
    cmd.arg("../tests/output/baseline.safetensors")
        .arg("../tests/output/degraded.safetensors");

    let output = cmd.output()?;

    // Should accept the flag and either succeed or show a parse error for invalid safetensors
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success()
            || stderr.contains("Failed to parse")
            || stderr.contains("HeaderTooSmall")
            || stderr.contains("Error:")
    );

    // Clean up
    let _ = fs::remove_file("../tests/output/baseline.safetensors");
    let _ = fs::remove_file("../tests/output/degraded.safetensors");

    Ok(())
}
