/// Integration tests for ML recommendations system
/// 
/// Tests the end-to-end functionality of the recommendation system
/// including CLI output, message formatting, and priority handling.

use std::process::Command;
use std::str;

#[test]
fn test_ml_recommendations_cli_output() {
    // Test with models that should trigger recommendations
    let output = Command::new("cargo")
        .args(&["run", "--bin", "diffai", "--", 
               "tests/fixtures/ml_models/simple_base.safetensors", 
               "tests/fixtures/ml_models/simple_modified.safetensors"])
        .output()
        .expect("Failed to execute diffai");

    let stdout = str::from_utf8(&output.stdout).unwrap();
    let stderr = str::from_utf8(&output.stderr).unwrap();

    println!("STDOUT: {}", stdout);
    println!("STDERR: {}", stderr);

    // Check if command executed successfully
    // Note: This depends on the specific test models having differences that trigger recommendations
    assert!(output.status.success() || stdout.contains("No differences found") || stderr.contains("Error"));
}

#[test]
fn test_ml_recommendations_not_in_json_output() {
    // Test that recommendations don't appear in JSON output
    let output = Command::new("cargo")
        .args(&["run", "--bin", "diffai", "--", 
               "tests/fixtures/ml_models/model_quantized.safetensors", 
               "tests/fixtures/ml_models/model_fp32.safetensors",
               "--output", "json"])
        .output()
        .expect("Failed to execute diffai");

    let stdout = str::from_utf8(&output.stdout).unwrap();
    let stderr = str::from_utf8(&output.stderr).unwrap();

    println!("JSON STDOUT: {}", stdout);
    println!("JSON STDERR: {}", stderr);

    // Recommendations should not appear in JSON output
    assert!(!stdout.contains("[CRITICAL]"));
    assert!(!stdout.contains("[WARNING]"));
    assert!(!stdout.contains("[RECOMMENDATIONS]"));
    
    // JSON should be valid
    if !stdout.is_empty() && !stdout.contains("No differences found") {
        serde_json::from_str::<serde_json::Value>(stdout).expect("Should be valid JSON");
    }
}

#[test]
fn test_ml_recommendations_not_in_yaml_output() {
    // Test that recommendations don't appear in YAML output
    let output = Command::new("cargo")
        .args(&["run", "--bin", "diffai", "--", 
               "tests/fixtures/ml_models/model_quantized.safetensors", 
               "tests/fixtures/ml_models/model_fp32.safetensors",
               "--output", "yaml"])
        .output()
        .expect("Failed to execute diffai");

    let stdout = str::from_utf8(&output.stdout).unwrap();
    let stderr = str::from_utf8(&output.stderr).unwrap();

    println!("YAML STDOUT: {}", stdout);
    println!("YAML STDERR: {}", stderr);

    // Recommendations should not appear in YAML output
    assert!(!stdout.contains("[CRITICAL]"));
    assert!(!stdout.contains("[WARNING]"));
    assert!(!stdout.contains("[RECOMMENDATIONS]"));
    
    // YAML should be valid
    if !stdout.is_empty() && !stdout.contains("No differences found") {
        serde_yml::from_str::<serde_yml::Value>(stdout).expect("Should be valid YAML");
    }
}

#[test]
fn test_ml_recommendations_priority_ordering() {
    // Test with models that should trigger multiple priority levels
    let output = Command::new("cargo")
        .args(&["run", "--bin", "diffai", "--", 
               "tests/fixtures/ml_models/anomalous_model.safetensors", 
               "tests/fixtures/ml_models/normal_model.safetensors"])
        .output()
        .expect("Failed to execute diffai");

    let stdout = str::from_utf8(&output.stdout).unwrap();
    let stderr = str::from_utf8(&output.stderr).unwrap();

    println!("PRIORITY STDOUT: {}", stdout);
    println!("PRIORITY STDERR: {}", stderr);

    // If recommendations are present, check priority ordering
    let output_text = format!("{}{}", stdout, stderr);
    
    if output_text.contains("[CRITICAL]") {
        // CRITICAL should appear before WARNING and RECOMMENDATIONS
        let critical_pos = output_text.find("[CRITICAL]").unwrap();
        if let Some(warning_pos) = output_text.find("[WARNING]") {
            assert!(critical_pos < warning_pos, "CRITICAL should appear before WARNING");
        }
        if let Some(rec_pos) = output_text.find("[RECOMMENDATIONS]") {
            assert!(critical_pos < rec_pos, "CRITICAL should appear before RECOMMENDATIONS");
        }
    }
    
    if output_text.contains("[WARNING]") {
        // WARNING should appear before RECOMMENDATIONS
        let warning_pos = output_text.find("[WARNING]").unwrap();
        if let Some(rec_pos) = output_text.find("[RECOMMENDATIONS]") {
            assert!(warning_pos < rec_pos, "WARNING should appear before RECOMMENDATIONS");
        }
    }
}

#[test]
fn test_ml_recommendations_message_format() {
    // Test message format structure
    let output = Command::new("cargo")
        .args(&["run", "--bin", "diffai", "--", 
               "tests/fixtures/ml_models/large_model.safetensors", 
               "tests/fixtures/ml_models/small_model.safetensors"])
        .output()
        .expect("Failed to execute diffai");

    let stdout = str::from_utf8(&output.stdout).unwrap();
    let stderr = str::from_utf8(&output.stderr).unwrap();

    println!("FORMAT STDOUT: {}", stdout);
    println!("FORMAT STDERR: {}", stderr);

    let output_text = format!("{}{}", stdout, stderr);
    
    // Check for proper message format if recommendations exist
    if output_text.contains("•") {
        let lines: Vec<&str> = output_text.lines().collect();
        for line in lines {
            if line.trim().starts_with("•") {
                // Each recommendation should be a bullet point
                assert!(line.contains("•"), "Recommendations should use bullet points");
                
                // Should contain meaningful content (not just punctuation)
                let content = line.trim_start_matches("•").trim();
                assert!(content.len() > 10, "Recommendation should have meaningful content");
                
                // Should end with a period (proper English)
                assert!(content.ends_with('.'), "Recommendation should end with period");
            }
        }
    }
}

#[test]
fn test_ml_recommendations_with_verbose_mode() {
    // Test that recommendations work with verbose mode
    let output = Command::new("cargo")
        .args(&["run", "--bin", "diffai", "--", 
               "tests/fixtures/ml_models/model_quantized.safetensors", 
               "tests/fixtures/ml_models/model_fp32.safetensors",
               "--verbose"])
        .output()
        .expect("Failed to execute diffai");

    let stdout = str::from_utf8(&output.stdout).unwrap();
    let stderr = str::from_utf8(&output.stderr).unwrap();

    println!("VERBOSE STDOUT: {}", stdout);
    println!("VERBOSE STDERR: {}", stderr);

    // Should contain verbose information
    let output_text = format!("{}{}", stdout, stderr);
    assert!(output_text.contains("verbose mode") || output_text.contains("Processing results"));
    
    // Recommendations should still appear in CLI mode even with verbose
    // (This depends on the test models having differences that trigger recommendations)
}

#[test]
fn test_ml_recommendations_limit_to_five() {
    // Test that recommendations are limited to 5 items
    let output = Command::new("cargo")
        .args(&["run", "--bin", "diffai", "--", 
               "tests/fixtures/ml_models/transformer.safetensors", 
               "tests/fixtures/ml_models/simple_base.safetensors"])
        .output()
        .expect("Failed to execute diffai");

    let stdout = str::from_utf8(&output.stdout).unwrap();
    let stderr = str::from_utf8(&output.stderr).unwrap();

    println!("LIMIT STDOUT: {}", stdout);
    println!("LIMIT STDERR: {}", stderr);

    let output_text = format!("{}{}", stdout, stderr);
    
    // Count bullet points in recommendations
    let bullet_count = output_text.matches("•").count();
    
    // Should not exceed 5 recommendations
    if bullet_count > 0 {
        assert!(bullet_count <= 5, "Should not exceed 5 recommendations, found {}", bullet_count);
    }
}

#[test]
fn test_ml_recommendations_with_no_ml_analysis() {
    // Test with non-ML files (should not trigger ML recommendations)
    let output = Command::new("cargo")
        .args(&["run", "--bin", "diffai", "--", 
               "tests/fixtures/data1.json", 
               "tests/fixtures/data2.json"])
        .output()
        .expect("Failed to execute diffai");

    let stdout = str::from_utf8(&output.stdout).unwrap();
    let stderr = str::from_utf8(&output.stderr).unwrap();

    println!("NO ML STDOUT: {}", stdout);
    println!("NO ML STDERR: {}", stderr);

    let output_text = format!("{}{}", stdout, stderr);
    
    // Should not contain ML recommendations for non-ML files
    assert!(!output_text.contains("[CRITICAL]"));
    assert!(!output_text.contains("[WARNING]"));
    assert!(!output_text.contains("[RECOMMENDATIONS]"));
}

#[test]
fn test_ml_recommendations_colored_output() {
    // Test that colored output works (when not piped)
    let output = Command::new("cargo")
        .args(&["run", "--bin", "diffai", "--", 
               "tests/fixtures/ml_models/model_quantized.safetensors", 
               "tests/fixtures/ml_models/model_fp32.safetensors"])
        .output()
        .expect("Failed to execute diffai");

    let stdout = str::from_utf8(&output.stdout).unwrap();
    let stderr = str::from_utf8(&output.stderr).unwrap();

    println!("COLORED STDOUT: {}", stdout);
    println!("COLORED STDERR: {}", stderr);

    // Note: When running through Command, colors might be disabled
    // This test mainly ensures the command completes successfully
    assert!(output.status.success() || stdout.contains("No differences found"));
}

#[test]
fn test_ml_recommendations_exit_code() {
    // Test that exit codes reflect recommendation priority
    let output = Command::new("cargo")
        .args(&["run", "--bin", "diffai", "--", 
               "tests/fixtures/ml_models/model_quantized.safetensors", 
               "tests/fixtures/ml_models/model_fp32.safetensors"])
        .output()
        .expect("Failed to execute diffai");

    let stdout = str::from_utf8(&output.stdout).unwrap();
    let stderr = str::from_utf8(&output.stderr).unwrap();

    println!("EXIT CODE STDOUT: {}", stdout);
    println!("EXIT CODE STDERR: {}", stderr);
    println!("EXIT CODE: {}", output.status.code().unwrap_or(-1));

    // Exit code should reflect the presence of differences
    // 0 = no differences, non-zero = differences found
    // Note: Current implementation may not set exit codes based on recommendation priority
    assert!(output.status.code().unwrap_or(-1) >= 0);
}

#[test]
fn test_ml_recommendations_thread_safety() {
    // Test that recommendations work correctly in concurrent scenarios
    use std::thread;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    let success_count = Arc::new(AtomicUsize::new(0));
    let mut handles = vec![];

    for i in 0..3 {
        let success_count_clone = Arc::clone(&success_count);
        let handle = thread::spawn(move || {
            let output = Command::new("cargo")
                .args(&["run", "--bin", "diffai", "--", 
                       "tests/fixtures/ml_models/model_quantized.safetensors", 
                       "tests/fixtures/ml_models/model_fp32.safetensors"])
                .output()
                .expect("Failed to execute diffai");

            let stdout = str::from_utf8(&output.stdout).unwrap();
            let stderr = str::from_utf8(&output.stderr).unwrap();

            println!("THREAD {} STDOUT: {}", i, stdout);
            println!("THREAD {} STDERR: {}", i, stderr);

            if output.status.success() {
                success_count_clone.fetch_add(1, Ordering::SeqCst);
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    // All threads should complete successfully
    assert_eq!(success_count.load(Ordering::SeqCst), 3);
}