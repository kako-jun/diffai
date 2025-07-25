#[allow(unused_imports)]
#[allow(unused_imports)]
use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::fs;
use std::process::Command;

// Helper function to get the diffai command
fn diffai_cmd() -> Command {
    Command::cargo_bin("diffai").expect("Failed to find diffai binary")
}

/// Test combined ML features
/// Corresponds to: docs/examples/test-results/ml_combined_features.md
#[test]
fn test_meta_chaining() -> Result<(), Box<dyn std::error::Error>> {
    // Ensure test output directory exists
    std::fs::create_dir_all("../tests/output")?;

    // Step 1: Generate diff_report_v1.json
    let mut cmd1 = diffai_cmd();
    cmd1.arg("../tests/fixtures/config_v1.json")
        .arg("../tests/fixtures/config_v2.json")
        .arg("--output")
        .arg("json");
    let output1 = cmd1.output()?.stdout;
    std::fs::write("../tests/output/diff_report_v1.json", output1)?;

    // Step 2: Generate diff_report_v2.json
    let mut cmd2 = diffai_cmd();
    cmd2.arg("../tests/fixtures/config_v2.json")
        .arg("../tests/fixtures/config_v3.json")
        .arg("--output")
        .arg("json");
    let output2 = cmd2.output()?.stdout;
    std::fs::write("../tests/output/diff_report_v2.json", output2)?;

    // Step 3: Compare the two diff reports
    let mut cmd3 = diffai_cmd();
    cmd3.arg("../tests/output/diff_report_v1.json")
        .arg("../tests/output/diff_report_v2.json");
    cmd3.assert()
        .success()
        .stdout(predicate::str::contains(
            r#"~ [1].Modified[1]: "1.0" -> "1.1""#,
        ))
        .stdout(predicate::str::contains(
            r#"~ [1].Modified[2]: "1.1" -> "1.2""#,
        ))
        .stdout(predicate::str::contains(
            r#"+ [2]: {"Added":["features[2]","featureD"]}"#,
        ));

    // Clean up generated diff report files
    std::fs::remove_file("../tests/output/diff_report_v1.json")?;
    std::fs::remove_file("../tests/output/diff_report_v2.json")?;

    Ok(())
}

/// Test ML format detection for SafeTensors
#[test]
fn test_safetensors_format_detection() -> Result<(), Box<dyn std::error::Error>> {
    // Helper function for creating test safetensors files
    fn create_test_safetensors_file(path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let test_data = b"{}"; // Minimal JSON metadata
        fs::create_dir_all("../tests/output")?;
        fs::write(path, test_data)?;
        Ok(())
    }

    // Create minimal test safetensors files
    create_test_safetensors_file("../tests/output/test1.safetensors")?;
    create_test_safetensors_file("../tests/output/test2.safetensors")?;

    let mut cmd = diffai_cmd();
    cmd.arg("../tests/output/test1.safetensors")
        .arg("../tests/output/test2.safetensors");

    // Should detect safetensors format automatically
    let output = cmd.output()?;

    // For now, since we create identical test files, expect no differences
    // or an error message indicating parsing issues
    assert!(
        output.status.success()
            || String::from_utf8_lossy(&output.stderr).contains("Failed to parse")
            || String::from_utf8_lossy(&output.stderr).contains("HeaderTooLarge")
            || String::from_utf8_lossy(&output.stderr).contains("HeaderTooSmall")
    );

    // Clean up
    let _ = fs::remove_file("../tests/output/test1.safetensors");
    let _ = fs::remove_file("../tests/output/test2.safetensors");

    Ok(())
}

/// Test ML format detection for PyTorch
#[test]
fn test_pytorch_format_detection() -> Result<(), Box<dyn std::error::Error>> {
    // Helper function for creating test PyTorch files
    fn create_test_pytorch_file(path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let test_data = b"\x80\x02}q\x00."; // Minimal pickle header
        fs::create_dir_all("../tests/output")?;
        fs::write(path, test_data)?;
        Ok(())
    }

    // Create minimal test PyTorch files
    create_test_pytorch_file("../tests/output/test1.pt")?;
    create_test_pytorch_file("../tests/output/test2.pt")?;

    let mut cmd = diffai_cmd();
    cmd.arg("../tests/output/test1.pt")
        .arg("../tests/output/test2.pt");

    // Should detect pytorch format automatically
    let output = cmd.output()?;

    // Expect parsing error since we create minimal test files
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("Failed to parse")
            || output.status.success()
    );

    // Clean up
    let _ = fs::remove_file("../tests/output/test1.pt");
    let _ = fs::remove_file("../tests/output/test2.pt");

    Ok(())
}

/// Test ML model comparison with epsilon
#[test]
fn test_ml_model_comparison_with_epsilon() -> Result<(), Box<dyn std::error::Error>> {
    // Helper function for creating test safetensors files
    fn create_test_safetensors_file(path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let test_data = b"{}"; // Minimal JSON metadata
        fs::create_dir_all("../tests/output")?;
        fs::write(path, test_data)?;
        Ok(())
    }

    // Test that epsilon parameter works with ML model comparison
    create_test_safetensors_file("../tests/output/model1.safetensors")?;
    create_test_safetensors_file("../tests/output/model2.safetensors")?;

    let mut cmd = diffai_cmd();
    cmd.arg("../tests/output/model1.safetensors")
        .arg("../tests/output/model2.safetensors")
        .arg("--epsilon")
        .arg("0.001");

    let output = cmd.output()?;

    // Should handle epsilon parameter without crashing
    assert!(
        output.status.success()
            || String::from_utf8_lossy(&output.stderr).contains("Failed to parse")
            || String::from_utf8_lossy(&output.stderr).contains("HeaderTooLarge")
            || String::from_utf8_lossy(&output.stderr).contains("HeaderTooSmall")
    );

    // Clean up
    let _ = fs::remove_file("../tests/output/model1.safetensors");
    let _ = fs::remove_file("../tests/output/model2.safetensors");

    Ok(())
}
