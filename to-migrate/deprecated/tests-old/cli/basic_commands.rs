use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::process::Command;

// Helper function to get the diffai command (following diffx pattern)
fn diffai_cmd() -> Command {
    Command::cargo_bin("diffai").expect("Failed to find diffai binary")
}

/// Test basic help command
#[test]
fn test_help_command() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("--help");

    cmd.assert()
        .success()
        .stdout(predicates::str::contains("diffai"))
        .stdout(predicates::str::contains("AI/ML specialized diff tool"));

    Ok(())
}

/// Test version command
#[test]
fn test_version_command() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("--version");

    cmd.assert()
        .success()
        .stdout(predicates::str::contains("diffai"));

    Ok(())
}

/// Test no-color option basic functionality
#[test]
fn test_no_color_option_basic() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/simple_base.pt")
        .arg("tests/fixtures/ml_models/simple_modified.pt")
        .arg("--no-color");

    let output = cmd.output()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    
    // Should not contain ANSI color codes
    assert!(!stdout.contains("\x1b["));
    
    Ok(())
}

/// Test insufficient arguments error
#[test]
fn test_insufficient_arguments() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/model1.pt");

    cmd.assert()
        .failure()
        .stderr(predicates::str::contains("required"));

    Ok(())
}

/// Test nonexistent file error
#[test]
fn test_nonexistent_file() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("nonexistent_file.pt")
        .arg("tests/fixtures/ml_models/model1.pt");

    cmd.assert()
        .failure()
        .stderr(predicates::str::contains("No such file"));

    Ok(())
}

/// Test basic PyTorch file comparison
#[test]
fn test_basic_pytorch_comparison() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/simple_base.pt")
        .arg("tests/fixtures/ml_models/simple_modified.pt");

    let output = cmd.output()?;
    
    // Should process files without panic
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("panic"));
    
    // Exit code should be consistent (0 if no differences, 1 if differences found)
    assert!(output.status.code() == Some(0) || output.status.code() == Some(1));

    Ok(())
}

/// Test basic Safetensors file comparison
#[test]
fn test_basic_safetensors_comparison() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/simple_base.safetensors")
        .arg("tests/fixtures/ml_models/simple_modified.safetensors");

    let output = cmd.output()?;
    
    // Should process files without panic
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("panic"));
    
    // Exit code should be consistent
    assert!(output.status.code() == Some(0) || output.status.code() == Some(1));

    Ok(())
}

/// Test format specification
#[test]
fn test_format_specification() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/model1.pt")
        .arg("tests/fixtures/ml_models/model2.pt")
        .arg("--format")
        .arg("pytorch");

    let output = cmd.output()?;
    
    // Should handle format specification gracefully
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("panic"));

    Ok(())
}

/// Test verbose mode
#[test]
fn test_verbose_mode() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/small_model.pt")
        .arg("tests/fixtures/ml_models/small_model.pt")  // Same file
        .arg("--verbose");

    let output = cmd.output()?;
    
    // Should provide verbose output or handle gracefully
    let combined_output = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(!combined_output.contains("panic"));

    Ok(())
}