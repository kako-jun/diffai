use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::process::Command;

// Helper function to get the diffai command (following diffx pattern)
fn diffai_cmd() -> Command {
    Command::cargo_bin("diffai").expect("Failed to find diffai binary")
}

/// Test help command output consistency with documentation
#[test]
fn test_help_command_documentation_consistency() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("--help");

    cmd.assert()
        .success()
        .stdout(predicates::str::contains("diffai"))
        .stdout(predicates::str::contains("AI/ML specialized diff tool"))
        .stdout(predicates::str::contains("FILE1"))
        .stdout(predicates::str::contains("FILE2"))
        .stdout(predicates::str::contains("--format"))
        .stdout(predicates::str::contains("pytorch, safetensors, numpy, matlab"));

    Ok(())
}

/// Test version command output
#[test]
fn test_version_command_output() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("--version");

    cmd.assert()
        .success()
        .stdout(predicates::str::contains("diffai"));

    Ok(())
}

/// Test that help includes all documented format options
#[test]
fn test_help_includes_all_formats() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("--help");

    let output = cmd.output()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    
    // Verify all supported formats are mentioned in help
    assert!(stdout.contains("pytorch"));
    assert!(stdout.contains("safetensors"));
    assert!(stdout.contains("numpy"));
    assert!(stdout.contains("matlab"));

    Ok(())
}

/// Test that help includes essential ML-specific options
#[test]
fn test_help_includes_ml_options() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("--help");

    let output = cmd.output()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    
    // Verify essential options are documented
    assert!(stdout.contains("--format"));
    assert!(stdout.contains("--output"));
    assert!(stdout.contains("--verbose"));
    assert!(stdout.contains("--no-color"));

    Ok(())
}

/// Test short help option works
#[test]
fn test_short_help_option() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("-h");

    cmd.assert()
        .success()
        .stdout(predicates::str::contains("diffai"))
        .stdout(predicates::str::contains("Usage:"));

    Ok(())
}

/// Test short version option works
#[test]
fn test_short_version_option() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("-V");

    cmd.assert()
        .success()
        .stdout(predicates::str::contains("diffai"));

    Ok(())
}