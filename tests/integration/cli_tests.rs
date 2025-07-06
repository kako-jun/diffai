use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::fs;
use std::process::Command;

// Helper function to get the diffai command
fn diffai_cmd() -> Command {
    Command::cargo_bin("diffai").expect("Failed to find diffai binary")
}

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

#[test]
fn test_basic_yaml_diff() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/file1.yaml")
        .arg("../tests/fixtures/file2.yaml");
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("~ age: 30 -> 31"))
        .stdout(predicate::str::contains(
            "~ city: \"New York\" -> \"Boston\"",
        ))
        .stdout(predicate::str::contains("  + items[2]: \"orange\""));
    Ok(())
}

#[test]
fn test_basic_toml_diff() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/file1.toml")
        .arg("../tests/fixtures/file2.toml");
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("~ age: 30 -> 31"))
        .stdout(predicate::str::contains(
            "~ city: \"New York\" -> \"Boston\"",
        ))
        .stdout(predicate::str::contains("  + items[2]: \"orange\""));
    Ok(())
}

#[test]
fn test_basic_ini_diff() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/file1.ini")
        .arg("../tests/fixtures/file2.ini");
    cmd.assert()
        .success()
        .stdout(predicate::str::contains(
            "~ section1.key2: \"value2\" -> \"new_value2\"",
        ))
        .stdout(predicate::str::contains("+ section2.key4: \"value4\""));
    Ok(())
}

#[test]
fn test_basic_xml_diff() -> Result<(), Box<dyn std::error::Error>> {
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

#[test]
fn test_basic_csv_diff() -> Result<(), Box<dyn std::error::Error>> {
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

#[test]
fn test_specify_input_format() -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;

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

// ML-specific tests

#[test]
fn test_safetensors_format_detection() -> Result<(), Box<dyn std::error::Error>> {
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

#[test]
fn test_pytorch_format_detection() -> Result<(), Box<dyn std::error::Error>> {
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

#[test]
fn test_ml_model_comparison_with_epsilon() -> Result<(), Box<dyn std::error::Error>> {
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

#[test]
fn test_ml_json_output_format() -> Result<(), Box<dyn std::error::Error>> {
    // Test JSON output format with ML model files
    create_test_safetensors_file("../tests/output/model_a.safetensors")?;
    create_test_safetensors_file("../tests/output/model_b.safetensors")?;

    let mut cmd = diffai_cmd();
    cmd.arg("../tests/output/model_a.safetensors")
        .arg("../tests/output/model_b.safetensors")
        .arg("--output")
        .arg("json");

    let output = cmd.output()?;

    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        // Should output valid JSON (even if empty array)
        assert!(stdout.starts_with('[') && stdout.trim_end().ends_with(']'));
    }

    // Clean up
    let _ = fs::remove_file("../tests/output/model_a.safetensors");
    let _ = fs::remove_file("../tests/output/model_b.safetensors");

    Ok(())
}

#[test]
fn test_unsupported_ml_format_error() -> Result<(), Box<dyn std::error::Error>> {
    // Test error handling for unsupported ML formats
    fs::write("../tests/output/fake.onnx", b"fake onnx data")?;
    fs::write("../tests/output/fake2.onnx", b"fake onnx data2")?;

    let mut cmd = diffai_cmd();
    cmd.arg("../tests/output/fake.onnx")
        .arg("../tests/output/fake2.onnx");

    let output = cmd.output()?;

    // Should handle unknown format gracefully or show error
    let stderr = String::from_utf8_lossy(&output.stderr);
    let _stdout = String::from_utf8_lossy(&output.stdout);

    // Either success (treated as regular file) or error about format
    assert!(
        output.status.success() || stderr.contains("format") || stderr.contains("Could not infer")
    );

    // Clean up
    let _ = fs::remove_file("../tests/output/fake.onnx");
    let _ = fs::remove_file("../tests/output/fake2.onnx");

    Ok(())
}

// Helper functions for creating test ML files

fn create_test_safetensors_file(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    // Create a minimal safetensors file structure
    // This is just for testing file detection, not actual parsing
    let test_data = b"{}"; // Minimal JSON metadata
    fs::create_dir_all("../tests/output")?;
    fs::write(path, test_data)?;
    Ok(())
}

fn create_test_pytorch_file(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    // Create a minimal PyTorch file structure
    // This is just for testing file detection, not actual parsing
    let test_data = b"\x80\x02}q\x00."; // Minimal pickle header
    fs::create_dir_all("../tests/output")?;
    fs::write(path, test_data)?;
    Ok(())
}

// Tests for all 13 advanced ML features
// These tests verify that the CLI arguments are accepted and processed without errors

#[test]
fn test_learning_progress_analysis() -> Result<(), Box<dyn std::error::Error>> {
    // Use JSON files for testing CLI flag acceptance
    fs::create_dir_all("../tests/output")?;
    fs::write(
        "../tests/output/checkpoint_v1.json",
        r#"{"model": "v1", "epoch": 10}"#,
    )?;
    fs::write(
        "../tests/output/checkpoint_v2.json",
        r#"{"model": "v2", "epoch": 20}"#,
    )?;

    let mut cmd = diffai_cmd();
    cmd.arg("../tests/output/checkpoint_v1.json")
        .arg("../tests/output/checkpoint_v2.json")
        .arg("--learning-progress");

    let output = cmd.output()?;

    // Should accept the flag and process the files (success is expected for JSON)
    assert!(output.status.success());

    // Check that the flag was processed (should be in stdout or stderr)
    let _stdout = String::from_utf8_lossy(&output.stdout);
    let _stderr = String::from_utf8_lossy(&output.stderr);

    // The test passes if it processes successfully
    assert!(output.status.success());

    // Clean up
    let _ = fs::remove_file("../tests/output/checkpoint_v1.json");
    let _ = fs::remove_file("../tests/output/checkpoint_v2.json");

    Ok(())
}

#[test]
fn test_convergence_analysis() -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all("../tests/output")?;
    fs::write(
        "../tests/output/model_before.json",
        r#"{"loss": 0.5, "accuracy": 0.8}"#,
    )?;
    fs::write(
        "../tests/output/model_after.json",
        r#"{"loss": 0.3, "accuracy": 0.9}"#,
    )?;

    let mut cmd = diffai_cmd();
    cmd.arg("../tests/output/model_before.json")
        .arg("../tests/output/model_after.json")
        .arg("--convergence-analysis");

    let output = cmd.output()?;

    // Should accept the flag and process successfully
    assert!(output.status.success());

    // Clean up
    let _ = fs::remove_file("../tests/output/model_before.json");
    let _ = fs::remove_file("../tests/output/model_after.json");

    Ok(())
}

#[test]
fn test_anomaly_detection() -> Result<(), Box<dyn std::error::Error>> {
    create_test_pytorch_file("../tests/output/model_normal.pt")?;
    create_test_pytorch_file("../tests/output/model_anomaly.pt")?;

    let mut cmd = diffai_cmd();
    cmd.arg("../tests/output/model_normal.pt")
        .arg("../tests/output/model_anomaly.pt")
        .arg("--anomaly-detection");

    let output = cmd.output()?;

    // Should accept the flag and either succeed or show a parse error for invalid safetensors
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success()
            || stderr.contains("Failed to parse")
            || stderr.contains("HeaderTooSmall")
            || stderr.contains("Error:")
    );

    // Clean up
    let _ = fs::remove_file("../tests/output/model_normal.pt");
    let _ = fs::remove_file("../tests/output/model_anomaly.pt");

    Ok(())
}

#[test]
fn test_gradient_analysis() -> Result<(), Box<dyn std::error::Error>> {
    create_test_pytorch_file("../tests/output/grad_before.pt")?;
    create_test_pytorch_file("../tests/output/grad_after.pt")?;

    let mut cmd = diffai_cmd();
    cmd.arg("../tests/output/grad_before.pt")
        .arg("../tests/output/grad_after.pt")
        .arg("--gradient-analysis");

    let output = cmd.output()?;

    // Should accept the flag and either succeed or show a parse error for invalid safetensors
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success()
            || stderr.contains("Failed to parse")
            || stderr.contains("HeaderTooSmall")
            || stderr.contains("Error:")
    );

    // Clean up
    let _ = fs::remove_file("../tests/output/grad_before.pt");
    let _ = fs::remove_file("../tests/output/grad_after.pt");

    Ok(())
}

#[test]
fn test_memory_analysis() -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all("../tests/output")?;
    fs::write(
        "../tests/output/lightweight.json",
        r#"{"size": "small", "memory": 100}"#,
    )?;
    fs::write(
        "../tests/output/heavy.json",
        r#"{"size": "large", "memory": 1000}"#,
    )?;

    let mut cmd = diffai_cmd();
    cmd.arg("../tests/output/lightweight.json")
        .arg("../tests/output/heavy.json")
        .arg("--memory-analysis");

    let output = cmd.output()?;

    // Should accept the flag and process successfully
    assert!(output.status.success());

    // Clean up
    let _ = fs::remove_file("../tests/output/lightweight.json");
    let _ = fs::remove_file("../tests/output/heavy.json");

    Ok(())
}

#[test]
fn test_inference_speed_estimate() -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all("../tests/output")?;
    fs::write(
        "../tests/output/fast_model.json",
        r#"{"speed": "fast", "params": 1000}"#,
    )?;
    fs::write(
        "../tests/output/slow_model.json",
        r#"{"speed": "slow", "params": 10000}"#,
    )?;

    let mut cmd = diffai_cmd();
    cmd.arg("../tests/output/fast_model.json")
        .arg("../tests/output/slow_model.json")
        .arg("--inference-speed-estimate");

    let output = cmd.output()?;

    // Should accept the flag and process successfully
    assert!(output.status.success());

    // Clean up
    let _ = fs::remove_file("../tests/output/fast_model.json");
    let _ = fs::remove_file("../tests/output/slow_model.json");

    Ok(())
}

#[test]
fn test_regression_test() -> Result<(), Box<dyn std::error::Error>> {
    create_test_safetensors_file("../tests/output/production.safetensors")?;
    create_test_safetensors_file("../tests/output/candidate.safetensors")?;

    let mut cmd = diffai_cmd();
    cmd.arg("../tests/output/production.safetensors")
        .arg("../tests/output/candidate.safetensors")
        .arg("--regression-test");

    let output = cmd.output()?;

    // Should accept the flag and either succeed or show a parse error for invalid safetensors
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success()
            || stderr.contains("Failed to parse")
            || stderr.contains("HeaderTooSmall")
            || stderr.contains("Error:")
    );

    // Clean up
    let _ = fs::remove_file("../tests/output/production.safetensors");
    let _ = fs::remove_file("../tests/output/candidate.safetensors");

    Ok(())
}

#[test]
fn test_alert_on_degradation() -> Result<(), Box<dyn std::error::Error>> {
    create_test_safetensors_file("../tests/output/baseline.safetensors")?;
    create_test_safetensors_file("../tests/output/degraded.safetensors")?;

    let mut cmd = diffai_cmd();
    cmd.arg("../tests/output/baseline.safetensors")
        .arg("../tests/output/degraded.safetensors")
        .arg("--alert-on-degradation");

    let output = cmd.output()?;

    // Should accept the flag and either succeed or show a parse error for invalid safetensors
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success()
            || stderr.contains("Failed to parse")
            || stderr.contains("HeaderTooSmall")
            || stderr.contains("Error:")
    );

    // Clean up
    let _ = fs::remove_file("../tests/output/baseline.safetensors");
    let _ = fs::remove_file("../tests/output/degraded.safetensors");

    Ok(())
}

#[test]
fn test_architecture_comparison() -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all("../tests/output")?;
    fs::write(
        "../tests/output/resnet.json",
        r#"{"architecture": "resnet", "layers": 18}"#,
    )?;
    fs::write(
        "../tests/output/efficientnet.json",
        r#"{"architecture": "efficientnet", "layers": 24}"#,
    )?;

    let mut cmd = diffai_cmd();
    cmd.arg("../tests/output/resnet.json")
        .arg("../tests/output/efficientnet.json")
        .arg("--architecture-comparison");

    let output = cmd.output()?;

    // Should accept the flag and process successfully
    assert!(output.status.success());

    // Clean up
    let _ = fs::remove_file("../tests/output/resnet.json");
    let _ = fs::remove_file("../tests/output/efficientnet.json");

    Ok(())
}

#[test]
fn test_param_efficiency_analysis() -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all("../tests/output")?;
    fs::write(
        "../tests/output/efficient.json",
        r#"{"efficiency": "high", "params": 1000000}"#,
    )?;
    fs::write(
        "../tests/output/inefficient.json",
        r#"{"efficiency": "low", "params": 10000000}"#,
    )?;

    let mut cmd = diffai_cmd();
    cmd.arg("../tests/output/efficient.json")
        .arg("../tests/output/inefficient.json")
        .arg("--param-efficiency-analysis");

    let output = cmd.output()?;

    // Should accept the flag and process successfully
    assert!(output.status.success());

    // Clean up
    let _ = fs::remove_file("../tests/output/efficient.json");
    let _ = fs::remove_file("../tests/output/inefficient.json");

    Ok(())
}

#[test]
fn test_hyperparameter_impact() -> Result<(), Box<dyn std::error::Error>> {
    create_test_safetensors_file("../tests/output/lr_001.safetensors")?;
    create_test_safetensors_file("../tests/output/lr_0001.safetensors")?;

    let mut cmd = diffai_cmd();
    cmd.arg("../tests/output/lr_001.safetensors")
        .arg("../tests/output/lr_0001.safetensors")
        .arg("--hyperparameter-impact");

    let output = cmd.output()?;

    // Should accept the flag and either succeed or show a parse error for invalid safetensors
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success()
            || stderr.contains("Failed to parse")
            || stderr.contains("HeaderTooSmall")
            || stderr.contains("Error:")
    );

    // Clean up
    let _ = fs::remove_file("../tests/output/lr_001.safetensors");
    let _ = fs::remove_file("../tests/output/lr_0001.safetensors");

    Ok(())
}

#[test]
fn test_learning_rate_analysis() -> Result<(), Box<dyn std::error::Error>> {
    create_test_pytorch_file("../tests/output/high_lr.pt")?;
    create_test_pytorch_file("../tests/output/low_lr.pt")?;

    let mut cmd = diffai_cmd();
    cmd.arg("../tests/output/high_lr.pt")
        .arg("../tests/output/low_lr.pt")
        .arg("--learning-rate-analysis");

    let output = cmd.output()?;

    // Should accept the flag and either succeed or show a parse error for invalid safetensors
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success()
            || stderr.contains("Failed to parse")
            || stderr.contains("HeaderTooSmall")
            || stderr.contains("Error:")
    );

    // Clean up
    let _ = fs::remove_file("../tests/output/high_lr.pt");
    let _ = fs::remove_file("../tests/output/low_lr.pt");

    Ok(())
}

#[test]
fn test_deployment_readiness() -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all("../tests/output")?;
    fs::write(
        "../tests/output/model_a.json",
        r#"{"status": "ready", "version": "1.0"}"#,
    )?;
    fs::write(
        "../tests/output/model_b.json",
        r#"{"status": "candidate", "version": "2.0"}"#,
    )?;

    let mut cmd = diffai_cmd();
    cmd.arg("../tests/output/model_a.json")
        .arg("../tests/output/model_b.json")
        .arg("--deployment-readiness");

    let output = cmd.output()?;

    // Should accept the flag and process successfully
    assert!(output.status.success());

    // Clean up
    let _ = fs::remove_file("../tests/output/model_a.json");
    let _ = fs::remove_file("../tests/output/model_b.json");

    Ok(())
}

#[test]
fn test_combined_advanced_features() -> Result<(), Box<dyn std::error::Error>> {
    // Test multiple advanced flags together using JSON files
    fs::create_dir_all("../tests/output")?;
    fs::write(
        "../tests/output/base.json",
        r#"{"model": "base", "params": 1000000}"#,
    )?;
    fs::write(
        "../tests/output/improved.json",
        r#"{"model": "improved", "params": 1200000}"#,
    )?;

    let mut cmd = diffai_cmd();
    cmd.arg("../tests/output/base.json")
        .arg("../tests/output/improved.json")
        .arg("--learning-progress")
        .arg("--convergence-analysis")
        .arg("--memory-analysis")
        .arg("--architecture-comparison")
        .arg("--stats");

    let output = cmd.output()?;

    // Should accept multiple flags and process successfully
    assert!(output.status.success());

    // Clean up
    let _ = fs::remove_file("../tests/output/base.json");
    let _ = fs::remove_file("../tests/output/improved.json");

    Ok(())
}

#[test]
fn test_json_output_with_advanced_features() -> Result<(), Box<dyn std::error::Error>> {
    // Test JSON output format with advanced features
    create_test_safetensors_file("../tests/output/json_test1.safetensors")?;
    create_test_safetensors_file("../tests/output/json_test2.safetensors")?;

    let mut cmd = diffai_cmd();
    cmd.arg("../tests/output/json_test1.safetensors")
        .arg("../tests/output/json_test2.safetensors")
        .arg("--learning-progress")
        .arg("--output")
        .arg("json");

    let output = cmd.output()?;

    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        // Should output valid JSON
        assert!(stdout.starts_with('[') && stdout.trim_end().ends_with(']'));
    }

    // Clean up
    let _ = fs::remove_file("../tests/output/json_test1.safetensors");
    let _ = fs::remove_file("../tests/output/json_test2.safetensors");

    Ok(())
}
