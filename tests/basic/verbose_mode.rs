use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::process::Command;
use std::io::Write;

// Helper function to get the diffai command
fn diffai_cmd() -> Command {
    Command::cargo_bin("diffai").expect("Failed to find diffai binary")
}

/// Test verbose mode output
/// Corresponds to: docs/examples/test-results/basic_verbose_mode.md
#[test]
fn test_verbose_mode() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/file1.json")
        .arg("../tests/fixtures/file2.json")
        .arg("--verbose");
    
    let output = cmd.output()?;
    assert!(output.status.success());
    
    let stdout = String::from_utf8_lossy(&output.stdout);
    // Verbose mode should provide more detailed output
    assert!(stdout.len() > 0);
    
    Ok(())
}

/// Test stdin input with format specification
#[test]
fn test_specify_input_format() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    let mut child = cmd
        .arg("-")
        .arg("../tests/fixtures/file2.json")
        .arg("--format")
        .arg("json")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()?;
    {
        let stdin = child.stdin.as_mut().ok_or("Failed to open stdin")?;
        stdin.write_all(
            r#"{
  "name": "Alice",
  "age": 30,
  "city": "New York",
  "config": {
    "users": [
      {"id": 1, "name": "Alice"},
      {"id": 2, "name": "Bob"}
    ],
    "settings": {"theme": "dark"}
  }
}"#
            .as_bytes(),
        )?;
    } // stdin is dropped here, closing the pipe
    let output = child.wait_with_output()?;
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(predicate::str::contains("~ age: 30 -> 31").eval(&stdout));
    assert!(predicate::str::contains("~ city: \"New York\" -> \"Boston\"").eval(&stdout));
    assert!(predicate::str::contains("~ name: \"Alice\" -> \"John\"").eval(&stdout));
    assert!(predicate::str::contains("+ items:").eval(&stdout));
    Ok(())
}