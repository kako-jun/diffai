use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::process::Command;

// Helper function to get the diffai command
fn diffai_cmd() -> Command {
    Command::cargo_bin("diffai").expect("Failed to find diffai binary")
}

/// Test JSON output format
/// Corresponds to: docs/examples/test-results/basic_json_output.md
#[test]
fn test_json_output_format() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/file1.json")
        .arg("../tests/fixtures/file2.json")
        .arg("--output")
        .arg("json");
    cmd.assert()
        .success()
        .stdout(predicate::str::contains(r#""Modified""#))
        .stdout(predicate::str::contains(r#""age""#))
        .stdout(predicate::str::contains(r#""city""#))
        .stdout(predicate::str::contains(r#""New York""#))
        .stdout(predicate::str::contains(r#""Boston""#))
        .stdout(predicate::str::contains(r#""Added""#))
        .stdout(predicate::str::contains(r#""items[2]""#))
        .stdout(predicate::str::contains(r#""orange""#));
    Ok(())
}

/// Test YAML output format
#[test]
fn test_yaml_output_format() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/file1.json")
        .arg("../tests/fixtures/file2.json")
        .arg("--output")
        .arg("yaml");
    cmd.assert()
        .success()
        .stdout(predicate::str::contains(
            r#"- Modified:
  - age
  - 30
  - 31"#,
        ))
        .stdout(predicate::str::contains(
            r#"- Modified:
  - city
  - New York
  - Boston"#,
        ))
        .stdout(predicate::str::contains(
            r#"- Added:
  - items[2]
  - orange"#,
        ));
    Ok(())
}

/// Test unified output format
#[test]
fn test_unified_output_format() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/file1.json")
        .arg("../tests/fixtures/file2.json")
        .arg("--output")
        .arg("unified");
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("-  \"age\": 30,"))
        .stdout(predicate::str::contains("+  \"age\": 31,"))
        .stdout(predicate::str::contains("-  \"city\": \"New York\","));
    Ok(())
}