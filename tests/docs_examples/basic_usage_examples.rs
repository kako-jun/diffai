use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::process::Command;
use tempfile::NamedTempFile;
use std::io::Write;

// Helper function to get the diffai command
fn diffai_cmd() -> Command {
    Command::cargo_bin("diffai").expect("Failed to find diffai binary")
}

/// Test basic file comparison from basic-usage.md
#[test]
fn test_basic_pytorch_model_comparison() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/simple_base.pt")
        .arg("tests/fixtures/ml_models/simple_modified.pt");

    let output = cmd.output()?;
    
    // Should process without panic, exit code should be reasonable
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("panic"));
    assert!(matches!(output.status.code(), Some(0) | Some(1) | Some(2)));

    Ok(())
}

/// Test safetensors file comparison from basic-usage.md
#[test]
fn test_basic_safetensors_comparison() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/simple_base.safetensors")
        .arg("tests/fixtures/ml_models/simple_modified.safetensors");

    let output = cmd.output()?;
    
    // Should process without panic
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("panic"));

    Ok(())
}

/// Test numpy array comparison from basic-usage.md
#[test]
fn test_basic_numpy_comparison() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/numpy_data1.npy")
        .arg("tests/fixtures/ml_models/numpy_data2.npy");

    let output = cmd.output()?;
    
    // Should handle numpy files gracefully
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("panic"));

    Ok(())
}

/// Test MATLAB file comparison from basic-usage.md
#[test]
fn test_basic_matlab_comparison() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/matlab_data1.mat")
        .arg("tests/fixtures/ml_models/matlab_data2.mat");

    let output = cmd.output()?;
    
    // Should handle MATLAB files gracefully
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("panic"));

    Ok(())
}

/// Test JSON output format from basic-usage.md
#[test]
fn test_json_output_format() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/small_model.pt")
        .arg("tests/fixtures/ml_models/small_model.pt")  // Same file for predictable output
        .arg("--output")
        .arg("json");

    let output = cmd.output()?;
    
    // Should handle JSON output gracefully
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("panic"));
    
    // If successful, output should be JSON-like or empty
    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        if !stdout.trim().is_empty() {
            // Should look like JSON (starts with [ or {)
            let trimmed = stdout.trim();
            assert!(trimmed.starts_with("[") || trimmed.starts_with("{") || trimmed == "[]");
        }
    }

    Ok(())
}

/// Test YAML output format from basic-usage.md
#[test]
fn test_yaml_output_format() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/small_model.pt")
        .arg("tests/fixtures/ml_models/small_model.pt")  // Same file
        .arg("--output")
        .arg("yaml");

    let output = cmd.output()?;
    
    // Should handle YAML output gracefully
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("panic"));

    Ok(())
}

/// Test verbose mode from basic-usage.md
#[test]
fn test_verbose_mode_example() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/model1.pt")
        .arg("tests/fixtures/ml_models/model2.pt")
        .arg("--verbose");

    let output = cmd.output()?;
    
    // Should handle verbose mode without crashing
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("panic"));

    Ok(())
}

/// Test no-color option from basic-usage.md
#[test]
fn test_no_color_option_example() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/simple_base.pt")
        .arg("tests/fixtures/ml_models/simple_modified.pt")
        .arg("--no-color");

    let output = cmd.output()?;
    
    // Should produce output without ANSI color codes
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(!stdout.contains("\x1b["));

    Ok(())
}

/// Test brief mode from basic-usage.md
#[test]
fn test_brief_mode_example() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/model1.pt")
        .arg("tests/fixtures/ml_models/model2.pt")
        .arg("--brief");

    let output = cmd.output()?;
    
    // Should handle brief mode gracefully
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("panic"));
    
    // Brief mode should provide minimal output
    let stdout = String::from_utf8_lossy(&output.stdout);
    if output.status.success() || output.status.code() == Some(1) {
        // Should mention files differing or being identical, or be empty
        assert!(stdout.is_empty() || stdout.contains("differ") || stdout.contains("identical"));
    }

    Ok(())
}

/// Test quiet mode from basic-usage.md
#[test]
fn test_quiet_mode_example() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/small_model.pt")
        .arg("tests/fixtures/ml_models/small_model.pt")  // Same file
        .arg("--quiet");

    let output = cmd.output()?;
    
    // Quiet mode should produce no stdout
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.is_empty());
    
    // Should exit with code 0 for identical files
    assert_eq!(output.status.code(), Some(0));

    Ok(())
}

/// Test format specification from basic-usage.md
#[test]
fn test_format_specification_example() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/model1.pt")
        .arg("tests/fixtures/ml_models/model2.pt")
        .arg("--format")
        .arg("pytorch");

    let output = cmd.output()?;
    
    // Should handle explicit format specification
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("panic"));

    Ok(())
}

/// Test comprehensive ML analysis from basic-usage.md examples
#[test]
fn test_comprehensive_ml_analysis_example() -> Result<(), Box<dyn std::error::Error>> {
    // Test the example showing all 11 ML features
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/checkpoint_epoch_0.pt")
        .arg("tests/fixtures/ml_models/checkpoint_epoch_10.pt");

    let output = cmd.output()?;
    
    // Should perform comprehensive analysis without crashing
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("panic"));
    
    // Exit code should be reasonable
    assert!(matches!(output.status.code(), Some(0) | Some(1) | Some(2)));

    Ok(())
}

/// Test cross-format comparison stability
#[test]
fn test_cross_format_comparison_example() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/model1.pt")
        .arg("tests/fixtures/ml_models/small_model.safetensors");

    let output = cmd.output()?;
    
    // Should handle cross-format comparison gracefully
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("panic"));

    Ok(())
}