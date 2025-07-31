use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::process::Command;

// Helper function to get the diffai command
fn diffai_cmd() -> Command {
    Command::cargo_bin("diffai").expect("Failed to find diffai binary")
}

/// Test default CLI output format
#[test]
fn test_default_output_format() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/simple_base.pt")
        .arg("tests/fixtures/ml_models/model1.pt");

    let output = cmd.output()?;
    
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    
    // Must not crash
    assert!(!stderr.contains("panic"), "Should not panic with default format");
    assert!(!stderr.contains("unwrap"), "Should not have unwrap errors");
    
    // Must succeed - exit code 1 means differences found, which is expected
    let exit_code = output.status.code().unwrap_or(-1);
    assert!(exit_code == 0 || exit_code == 1, "Should succeed with default format (exit code: {})", exit_code);
    
    // Must have output for different models
    assert!(!stdout.trim().is_empty(), "Should produce output with default format");
    
    // Default format should be human-readable, not JSON/YAML
    assert!(!stdout.starts_with("{"), "Default format should not be JSON");
    assert!(!stdout.starts_with("["), "Default format should not be JSON array");
    assert!(!stdout.starts_with("---"), "Default format should not be YAML");
    
    // Should contain diffai-style symbols (~, +, -)
    let contains_diffai_symbols = stdout.contains("  ~") || stdout.contains("  +") || stdout.contains("  -");
    assert!(contains_diffai_symbols, "Default format should use diffai symbols (~, +, -)");
    
    // Should contain ML analysis (ModelArchitectureChanged in JSON parts is OK)
    let has_ml_analysis = stdout.contains("memory_analysis") || 
                         stdout.contains("gradient_distributions") ||
                         stdout.contains("ModelArchitectureChanged");
    assert!(has_ml_analysis, "Default format should contain ML analysis results");

    Ok(())
}

/// Test JSON output format
#[test]
fn test_json_output_format() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/simple_base.pt")
        .arg("tests/fixtures/ml_models/simple_modified.pt")
        .arg("--output")
        .arg("json");

    let output = cmd.output()?;
    
    // Should handle JSON output format gracefully
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("panic"));
    
    // If successful and has output, should be valid JSON format
    let stdout = String::from_utf8_lossy(&output.stdout);
    if output.status.success() && !stdout.trim().is_empty() {
        // Basic JSON format check
        let trimmed = stdout.trim();
        assert!(trimmed.starts_with("[") || trimmed.starts_with("{"));
    }

    Ok(())
}

/// Test YAML output format
#[test]
fn test_yaml_output_format() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/simple_base.pt")
        .arg("tests/fixtures/ml_models/simple_modified.pt")
        .arg("--output")
        .arg("yaml");

    let output = cmd.output()?;
    
    // Should handle YAML output format gracefully
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("panic"));
    
    Ok(())
}

/// Test invalid output format
#[test]
fn test_invalid_output_format() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/model1.pt")
        .arg("tests/fixtures/ml_models/model2.pt")
        .arg("--output")
        .arg("invalid_format");

    cmd.assert()
        .failure()
        .stderr(predicates::str::contains("invalid").or(predicates::str::contains("Invalid")));

    Ok(())
}

/// Test output format with no differences
#[test]
fn test_output_format_no_differences() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/small_model.pt")
        .arg("tests/fixtures/ml_models/small_model.pt")  // Same file
        .arg("--output")
        .arg("json");

    let output = cmd.output()?;
    
    // Should handle identical files gracefully
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("panic"));
    
    // Exit code should be 0 for identical files
    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        if !stdout.trim().is_empty() {
            // Should be empty JSON array or empty object
            let trimmed = stdout.trim();
            assert!(trimmed == "[]" || trimmed == "{}" || trimmed.contains("[]"));
        }
    }

    Ok(())
}

/// Test output format with ML analysis results
#[test]
fn test_output_format_with_ml_analysis() -> Result<(), Box<dyn std::error::Error>> {
    let test_pairs = [
        ("json", "tests/fixtures/ml_models/model1.pt", "tests/fixtures/ml_models/model2.pt"),
        ("yaml", "tests/fixtures/ml_models/normal_model.safetensors", "tests/fixtures/ml_models/anomalous_model.safetensors"),
        ("json", "tests/fixtures/ml_models/model_fp32.pt", "tests/fixtures/ml_models/model_quantized.pt"),
    ];

    for (format, file1, file2) in test_pairs.iter() {
        let mut cmd = diffai_cmd();
        cmd.arg(file1)
            .arg(file2)
            .arg("--output")
            .arg(format);

        let output = cmd.output()?;
        
        // Should process ML analysis in structured format without crashing
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(!stderr.contains("panic"), "Failed for {} format with {} vs {}", format, file1, file2);
        
        // Should have reasonable exit code
        let code = output.status.code().unwrap_or(-1);
        assert!(code >= 0 && code <= 2, "Invalid exit code {} for {} format", code, format);
    }

    Ok(())
}

/// Test output format consistency across file types
#[test]
fn test_output_format_consistency() -> Result<(), Box<dyn std::error::Error>> {
    let formats = ["json", "yaml"];
    let file_pairs = [
        ("tests/fixtures/ml_models/simple_base.pt", "tests/fixtures/ml_models/simple_modified.pt"),
        ("tests/fixtures/ml_models/simple_base.safetensors", "tests/fixtures/ml_models/simple_modified.safetensors"),
    ];

    for format in formats.iter() {
        for (file1, file2) in file_pairs.iter() {
            let mut cmd = diffai_cmd();
            cmd.arg(file1)
                .arg(file2)
                .arg("--output")
                .arg(format);

            let output = cmd.output()?;
            
            // Should handle all combinations consistently
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(!stderr.contains("panic"), "Inconsistent behavior for {} format with {} vs {}", format, file1, file2);
        }
    }

    Ok(())
}

/// Test brief mode output
#[test]
fn test_brief_mode_output() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/model1.pt")
        .arg("tests/fixtures/ml_models/model2.pt")
        .arg("--brief");

    let output = cmd.output()?;
    
    // Should provide brief output without crashing
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("panic"));
    
    // Brief mode should provide minimal output
    let stdout = String::from_utf8_lossy(&output.stdout);
    if output.status.success() || output.status.code() == Some(1) {
        // Should either say files differ or are identical
        assert!(stdout.contains("differ") || stdout.contains("identical") || stdout.is_empty());
    }

    Ok(())
}

/// Test quiet mode (no output)
#[test]
fn test_quiet_mode_output() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/small_model.pt")
        .arg("tests/fixtures/ml_models/small_model.pt")  // Same file
        .arg("--quiet");

    let output = cmd.output()?;
    
    // Quiet mode should produce no stdout
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.is_empty());
    
    // Should exit with code 0 for identical files
    assert_eq!(output.status.code(), Some(0));

    Ok(())
}