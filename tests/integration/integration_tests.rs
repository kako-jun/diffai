use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::process::Command;
use tempfile::NamedTempFile;
use std::io::Write;

// Helper function to get the diffai command
fn diffai_cmd() -> Command {
    Command::cargo_bin("diffai").expect("Failed to find diffai binary")
}

/// Integration test: Complete ML model comparison workflow
#[test]
fn test_complete_ml_model_comparison_workflow() -> Result<(), Box<dyn std::error::Error>> {
    // Test comprehensive ML analysis across different model types
    let test_scenarios = [
        // Training progression analysis
        ("tests/fixtures/ml_models/checkpoint_epoch_0.pt", "tests/fixtures/ml_models/checkpoint_epoch_10.pt"),
        ("tests/fixtures/ml_models/checkpoint_epoch_10.pt", "tests/fixtures/ml_models/checkpoint_epoch_50.pt"),
        
        // Model architecture comparison
        ("tests/fixtures/ml_models/normal_model.pt", "tests/fixtures/ml_models/transformer.pt"),
        
        // Quantization analysis
        ("tests/fixtures/ml_models/model_fp32.pt", "tests/fixtures/ml_models/model_quantized.pt"),
        
        // Anomaly detection
        ("tests/fixtures/ml_models/normal_model.pt", "tests/fixtures/ml_models/anomalous_model.pt"),
    ];

    for (file1, file2) in test_scenarios.iter() {
        let mut cmd = diffai_cmd();
        cmd.arg(file1).arg(file2);

        let output = cmd.output()?;
        
        // Should handle all ML scenarios without crashing
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(!stderr.contains("panic"), "Failed ML workflow for {} vs {}", file1, file2);
        
        // Exit code should be reasonable
        let code = output.status.code().unwrap_or(-1);
        assert!(code >= 0 && code <= 2, "Invalid exit code {} for {} vs {}", code, file1, file2);
    }

    Ok(())
}

/// Integration test: Multi-format analysis workflow  
#[test]
fn test_multi_format_analysis_workflow() -> Result<(), Box<dyn std::error::Error>> {
    let format_combinations = [
        // Same model, different formats
        ("tests/fixtures/ml_models/simple_base.pt", "tests/fixtures/ml_models/simple_base.safetensors"),
        ("tests/fixtures/ml_models/transformer.pt", "tests/fixtures/ml_models/transformer.safetensors"),
        
        // Cross-format scientific data
        ("tests/fixtures/ml_models/numpy_data1.npy", "tests/fixtures/ml_models/matlab_data1.mat"),
        
        // Model vs scientific data (should handle gracefully)
        ("tests/fixtures/ml_models/small_model.pt", "tests/fixtures/ml_models/numpy_data1.npy"),
    ];

    for (file1, file2) in format_combinations.iter() {
        let mut cmd = diffai_cmd();
        cmd.arg(file1).arg(file2);

        let output = cmd.output()?;
        
        // Should handle cross-format comparisons gracefully
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(!stderr.contains("panic"), "Failed cross-format for {} vs {}", file1, file2);
    }

    Ok(())
}

/// Integration test: Output format consistency workflow
#[test]
fn test_output_format_consistency_workflow() -> Result<(), Box<dyn std::error::Error>> {
    let output_formats = ["json", "yaml"];
    let test_pairs = [
        ("tests/fixtures/ml_models/model1.pt", "tests/fixtures/ml_models/model2.pt"),
        ("tests/fixtures/ml_models/simple_base.safetensors", "tests/fixtures/ml_models/simple_modified.safetensors"),
        ("tests/fixtures/ml_models/numpy_data1.npy", "tests/fixtures/ml_models/numpy_data2.npy"),
    ];

    for format in output_formats.iter() {
        for (file1, file2) in test_pairs.iter() {
            let mut cmd = diffai_cmd();
            cmd.arg(file1)
                .arg(file2)
                .arg("--output")
                .arg(format);

            let output = cmd.output()?;
            
            // Should handle all format/file combinations consistently
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(!stderr.contains("panic"), "Failed {} format for {} vs {}", format, file1, file2);
        }
    }

    Ok(())
}

/// Integration test: Option combination workflow
#[test]
fn test_option_combination_workflow() -> Result<(), Box<dyn std::error::Error>> {
    let option_combinations = [
        vec!["--verbose", "--no-color"],
        vec!["--output", "json", "--no-color"],
        vec!["--brief", "--quiet"],
        vec!["--format", "pytorch", "--verbose"],
        vec!["--output", "yaml", "--verbose"],
    ];

    for options in option_combinations.iter() {
        let mut cmd = diffai_cmd();
        cmd.arg("tests/fixtures/ml_models/model1.pt")
            .arg("tests/fixtures/ml_models/model2.pt");
        
        for option in options {
            cmd.arg(option);
        }

        let output = cmd.output()?;
        
        // Should handle option combinations gracefully
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(!stderr.contains("panic"), "Failed option combination: {:?}", options);
    }

    Ok(())
}

/// Integration test: Large model analysis workflow
#[test]
fn test_large_model_analysis_workflow() -> Result<(), Box<dyn std::error::Error>> {
    // Test memory efficiency with larger models
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/large_model.pt")
        .arg("tests/fixtures/ml_models/transformer.pt")
        .arg("--memory-optimization");

    let output = cmd.output()?;
    
    // Should handle large models efficiently
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("panic"));
    assert!(!stderr.contains("out of memory"));

    Ok(())
}

/// Integration test: Scientific computing workflow
#[test]
fn test_scientific_computing_workflow() -> Result<(), Box<dyn std::error::Error>> {
    let scientific_scenarios = [
        // NumPy array analysis
        ("tests/fixtures/ml_models/numpy_data1.npy", "tests/fixtures/ml_models/numpy_data2.npy"),
        ("tests/fixtures/ml_models/numpy_model1.npz", "tests/fixtures/ml_models/numpy_model2.npz"),
        
        // MATLAB file analysis  
        ("tests/fixtures/ml_models/matlab_data1.mat", "tests/fixtures/ml_models/matlab_data2.mat"),
        ("tests/fixtures/ml_models/matlab_simple1.mat", "tests/fixtures/ml_models/matlab_simple2.mat"),
    ];

    for (file1, file2) in scientific_scenarios.iter() {
        let mut cmd = diffai_cmd();
        cmd.arg(file1).arg(file2);

        let output = cmd.output()?;
        
        // Should handle scientific data formats
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(!stderr.contains("panic"), "Failed scientific workflow for {} vs {}", file1, file2);
    }

    Ok(())
}

/// Integration test: Error handling workflow
#[test]
fn test_error_handling_workflow() -> Result<(), Box<dyn std::error::Error>> {
    // Test various error conditions
    let error_scenarios = [
        // Nonexistent files
        ("nonexistent1.pt", "tests/fixtures/ml_models/model1.pt"),
        ("tests/fixtures/ml_models/model1.pt", "nonexistent2.pt"),
        
        // Invalid formats
        // (We'll test these programmatically rather than with files)
    ];

    for (file1, file2) in error_scenarios.iter() {
        let mut cmd = diffai_cmd();
        cmd.arg(file1).arg(file2);

        let output = cmd.output()?;
        
        // Should fail gracefully with meaningful error messages
        assert!(!output.status.success());
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(!stderr.contains("panic"));
        // Should contain helpful error information
        assert!(stderr.contains("No such file") || stderr.contains("not found") || stderr.contains("error"));
    }

    Ok(())
}

/// Integration test: Identical files workflow
#[test]
fn test_identical_files_workflow() -> Result<(), Box<dyn std::error::Error>> {
    let identical_files = [
        "tests/fixtures/ml_models/small_model.pt",
        "tests/fixtures/ml_models/simple_base.safetensors", 
        "tests/fixtures/ml_models/numpy_data1.npy",
        "tests/fixtures/ml_models/matlab_data1.mat",
    ];

    for file in identical_files.iter() {
        let mut cmd = diffai_cmd();
        cmd.arg(file).arg(file);  // Compare file with itself

        let output = cmd.output()?;
        
        // Should handle identical files correctly
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(!stderr.contains("panic"), "Failed identical file test for {}", file);
        
        // Exit code should be 0 or 1 (depending on implementation)
        assert!(matches!(output.status.code(), Some(0) | Some(1)));
    }

    Ok(())
}

/// Integration test: Comprehensive JSON output workflow
#[test]
fn test_comprehensive_json_output_workflow() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/checkpoint_epoch_0.pt")
        .arg("tests/fixtures/ml_models/checkpoint_epoch_10.pt")
        .arg("--output")
        .arg("json");

    let output = cmd.output()?;
    
    // Should produce valid JSON for comprehensive analysis
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("panic"));
    
    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        if !stdout.trim().is_empty() {
            // Should be valid JSON format
            let trimmed = stdout.trim();
            assert!(trimmed.starts_with("[") || trimmed.starts_with("{") || trimmed == "[]");
        }
    }

    Ok(())
}

/// Integration test: No-color consistency workflow
#[test]
fn test_no_color_consistency_workflow() -> Result<(), Box<dyn std::error::Error>> {
    let test_files = [
        ("tests/fixtures/ml_models/model1.pt", "tests/fixtures/ml_models/model2.pt"),
        ("tests/fixtures/ml_models/simple_base.safetensors", "tests/fixtures/ml_models/simple_modified.safetensors"),
    ];

    for (file1, file2) in test_files.iter() {
        let mut cmd = diffai_cmd();
        cmd.arg(file1)
            .arg(file2)
            .arg("--no-color");

        let output = cmd.output()?;
        
        // Should produce consistent output without color codes
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(!stdout.contains("\x1b["), "Found ANSI codes in no-color output for {} vs {}", file1, file2);
    }

    Ok(())
}

/// Integration test: Format specification consistency workflow
#[test]
fn test_format_specification_consistency_workflow() -> Result<(), Box<dyn std::error::Error>> {
    let format_tests = [
        ("pytorch", "tests/fixtures/ml_models/model1.pt", "tests/fixtures/ml_models/model2.pt"),
        ("safetensors", "tests/fixtures/ml_models/simple_base.safetensors", "tests/fixtures/ml_models/simple_modified.safetensors"),
        ("numpy", "tests/fixtures/ml_models/numpy_data1.npy", "tests/fixtures/ml_models/numpy_data2.npy"),
        ("matlab", "tests/fixtures/ml_models/matlab_data1.mat", "tests/fixtures/ml_models/matlab_data2.mat"),
    ];

    for (format, file1, file2) in format_tests.iter() {
        let mut cmd = diffai_cmd();
        cmd.arg(file1)
            .arg(file2)
            .arg("--format")
            .arg(format);

        let output = cmd.output()?;
        
        // Should handle explicit format specification consistently
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(!stderr.contains("panic"), "Failed format specification {} for {} vs {}", format, file1, file2);
    }

    Ok(())
}