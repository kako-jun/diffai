#[allow(unused_imports)]
#[allow(unused_imports)]
use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::process::Command;

// Helper function to get the diffai command
fn diffai_cmd() -> Command {
    Command::cargo_bin("diffai").expect("Failed to find diffai binary")
}

/// Test JSON format comparison
/// Corresponds to: docs/examples/test-results/format_json_comparison.md
#[test]
fn test_json_format() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/file1.json")
        .arg("../tests/fixtures/file2.json");
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("~ age: 30 -> 31"))
        .stdout(predicate::str::contains(
            "~ city: \"New York\" -> \"Boston\"",
        ))
        .stdout(predicate::str::contains("  + items[2]: \"orange\""));
    Ok(())
}
