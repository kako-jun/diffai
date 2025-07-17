use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::process::Command;

// Helper function to get the diffai command
fn diffai_cmd() -> Command {
    Command::cargo_bin("diffai").expect("Failed to find diffai binary")
}

/// Test XML format comparison
/// Corresponds to: docs/examples/test-results/format_xml_comparison.md
#[test]
fn test_xml_format() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/file1.xml")
        .arg("../tests/fixtures/file2.xml");
    cmd.assert()
        .success()
        .stdout(predicate::str::contains(
            "~ item.$text: \"value2\" -> \"value3\"",
        ))
        .stdout(predicate::str::contains("~ item.@id: \"2\" -> \"3\""));
    Ok(())
}