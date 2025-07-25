use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::process::Command;

// Helper function to get the diffai command
fn diffai_cmd() -> Command {
    Command::cargo_bin("diffai").expect("Failed to find diffai binary")
}

/// Test examples from user guide basic usage
/// Corresponds to: docs/user-guide/basic-usage.md
#[test]
fn test_basic_usage_examples() -> Result<(), Box<dyn std::error::Error>> {
    // Test the most basic usage example from docs
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/file1.json")
        .arg("../tests/fixtures/file2.json");

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("~").or(predicate::str::contains("+")));

    Ok(())
}

/// Test output format examples from user guide
#[test]
fn test_output_format_examples() -> Result<(), Box<dyn std::error::Error>> {
    // Test JSON output example from docs
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/file1.json")
        .arg("../tests/fixtures/file2.json")
        .arg("--output")
        .arg("json");

    cmd.assert()
        .success()
        .stdout(predicate::str::starts_with("["));

    Ok(())
}
