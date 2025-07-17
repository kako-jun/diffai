use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::process::Command;

// Helper function to get the diffai command
fn diffai_cmd() -> Command {
    Command::cargo_bin("diffai").expect("Failed to find diffai binary")
}

/// Test CSV format comparison  
/// Corresponds to: docs/examples/test-results/format_csv_comparison.md
#[test]
fn test_csv_format() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/file1.csv")
        .arg("../tests/fixtures/file2.csv");
    cmd.assert()
        .success()
        .stdout(predicate::str::contains(
            "~ [0].header2: \"valueB\" -> \"new_valueB\"",
        ))
        .stdout(
            predicate::str::contains("+ [2]: ")
                .and(predicate::str::contains("\"header1\":\"valueE\""))
                .and(predicate::str::contains("\"header2\":\"valueF\"")),
        );
    Ok(())
}