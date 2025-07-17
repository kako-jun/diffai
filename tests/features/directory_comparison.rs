use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::process::Command;

// Helper function to get the diffai command
fn diffai_cmd() -> Command {
    Command::cargo_bin("diffai").expect("Failed to find diffai binary")
}

/// Test directory comparison functionality
/// Corresponds to: docs/examples/test-results/advanced_directory_comparison.md
#[test]
fn test_directory_comparison() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/dir1")
        .arg("../tests/fixtures/dir2")
        .arg("--recursive");
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("--- Comparing b.json ---"))
        .stdout(predicate::str::contains(
            "~ key3: \"value3\" -> \"new_value3\"",
        ));
    Ok(())
}