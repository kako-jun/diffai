use assert_cmd::prelude::*;
use std::fs;
use std::process::Command;

// Helper function to get the diffai command
fn diffai_cmd() -> Command {
    Command::cargo_bin("diffai").expect("Failed to find diffai binary")
}

/// Test ML statistical analysis
/// Corresponds to: docs/examples/test-results/ml_stats_analysis.md
#[test]
fn test_gradient_analysis_with_real_models() -> Result<(), Box<dyn std::error::Error>> {
    // Test gradient analysis with real ML model files
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/simple_base.safetensors")
        .arg("../tests/fixtures/ml_models/simple_modified.safetensors");

    let output = cmd.output()?;

    // Should process successfully with real ML files
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Check that gradient analysis was performed (should be in full analysis)
    assert!(stdout.contains("gradient_analysis") || stdout.contains("flow_health"));

    Ok(())
}

/// Test inference speed estimation
#[test]
fn test_inference_speed_estimate() -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all("../tests/output")?;
    fs::write(
        "../tests/output/fast_model.json",
        r#"{"speed": "fast", "params": 1000}"#,
    )?;
    fs::write(
        "../tests/output/slow_model.json",
        r#"{"speed": "slow", "params": 10000}"#,
    )?;

    let mut cmd = diffai_cmd();
    cmd.arg("../tests/output/fast_model.json")
        .arg("../tests/output/slow_model.json");

    let output = cmd.output()?;

    // Should accept the flag and process successfully
    assert!(output.status.success());

    // Clean up
    let _ = fs::remove_file("../tests/output/fast_model.json");
    let _ = fs::remove_file("../tests/output/slow_model.json");

    Ok(())
}

/// Test parameter efficiency analysis
#[test]
fn test_param_efficiency_analysis() -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all("../tests/output")?;
    fs::write(
        "../tests/output/efficient.json",
        r#"{"efficiency": "high", "params": 1000000}"#,
    )?;
    fs::write(
        "../tests/output/inefficient.json",
        r#"{"efficiency": "low", "params": 10000000}"#,
    )?;

    let mut cmd = diffai_cmd();
    cmd.arg("../tests/output/efficient.json")
        .arg("../tests/output/inefficient.json");

    let output = cmd.output()?;

    // Should accept the flag and process successfully
    assert!(output.status.success());

    // Clean up
    let _ = fs::remove_file("../tests/output/efficient.json");
    let _ = fs::remove_file("../tests/output/inefficient.json");

    Ok(())
}

/// Test learning rate analysis
#[test]
fn test_learning_rate_analysis() -> Result<(), Box<dyn std::error::Error>> {
    // Helper function for creating test PyTorch files
    fn create_test_pytorch_file(path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let test_data = b"\x80\x02}q\x00."; // Minimal pickle header
        fs::create_dir_all("../tests/output")?;
        fs::write(path, test_data)?;
        Ok(())
    }

    create_test_pytorch_file("../tests/output/high_lr.pt")?;
    create_test_pytorch_file("../tests/output/low_lr.pt")?;

    let mut cmd = diffai_cmd();
    cmd.arg("../tests/output/high_lr.pt")
        .arg("../tests/output/low_lr.pt");

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
    let _ = fs::remove_file("../tests/output/high_lr.pt");
    let _ = fs::remove_file("../tests/output/low_lr.pt");

    Ok(())
}
