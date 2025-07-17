use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::process::Command;

// Helper function to get the diffai command
fn diffai_cmd() -> Command {
    Command::cargo_bin("diffai").expect("Failed to find diffai binary")
}

/// Test basic JSON diff functionality
/// Corresponds to: docs/examples/test-results/basic_json_diff.md
#[test]
fn test_basic_json_diff() -> Result<(), Box<dyn std::error::Error>> {
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

/// Test epsilon comparison with JSON files
#[test]
fn test_epsilon_comparison() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/data1.json")
        .arg("../tests/fixtures/data2.json")
        .arg("--epsilon")
        .arg("0.00001");
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("No differences found.")); // No differences expected within epsilon
    Ok(())
}

/// Test ignore keys regex functionality
#[test]
fn test_ignore_keys_regex() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/file1.json")
        .arg("../tests/fixtures/file2.json")
        .arg("--ignore-keys-regex")
        .arg("^age$");
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("~ age:").not())
        .stdout(predicate::str::contains(
            r#"~ city: "New York" -> "Boston""#,
        ))
        .stdout(predicate::str::contains("+ items[2]: \"orange\""));
    Ok(())
}

/// Test array ID key functionality
#[test]
fn test_array_id_key() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/users1.json")
        .arg("../tests/fixtures/users2.json")
        .arg("--array-id-key")
        .arg("id");
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("~ [id=1].age: 25 -> 26"))
        .stdout(
            predicate::str::contains("+ [id=3]: ")
                .and(predicate::str::contains(r#""id":3"#))
                .and(predicate::str::contains(r#""name":"Charlie""#))
                .and(predicate::str::contains(r#""age":28"#)),
        )
        .stdout(predicate::str::contains("~ [0].").not()); // Ensure not comparing by index
    Ok(())
}