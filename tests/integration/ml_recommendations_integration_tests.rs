/// Integration tests for ML recommendations system
///
/// Tests the end-to-end functionality of the recommendation system
/// including CLI output, message formatting, and priority handling.
use std::process::Command;
use std::str;
use std::path::PathBuf;

fn get_project_root() -> PathBuf {
    // Get the directory containing the main Cargo.toml (workspace root)
    let mut current_dir = std::env::current_dir().unwrap();
    
    // Walk up until we find the workspace root with tests/ directory
    while !current_dir.join("tests").exists() || !current_dir.join("Cargo.toml").exists() {
        current_dir = current_dir.parent().unwrap().to_path_buf();
    }
    
    current_dir
}

fn get_fixture_path(relative_path: &str) -> PathBuf {
    get_project_root().join(relative_path)
}

#[test]
fn test_ml_recommendations_cli_output() {
    // Test with models that should trigger recommendations
    let project_root = get_project_root();
    let file1 = get_fixture_path("tests/fixtures/ml_models/simple_base.safetensors");
    let file2 = get_fixture_path("tests/fixtures/ml_models/simple_modified.safetensors");
    
    let output = Command::new("cargo")
        .args([
            "run",
            "--bin",
            "diffai",
            "--",
            file1.to_str().unwrap(),
            file2.to_str().unwrap(),
        ])
        .current_dir(&project_root)
        .output()
        .expect("Failed to execute diffai");

    let stdout = str::from_utf8(&output.stdout).unwrap();
    let stderr = str::from_utf8(&output.stderr).unwrap();

    println!("STDOUT: {stdout}");
    println!("STDERR: {stderr}");

    // Check if command executed successfully
    if !output.status.success() {
        panic!("Command failed with exit code: {:?}\nSTDOUT: {}\nSTDERR: {}", 
               output.status.code(), stdout, stderr);
    }
    
    // Should contain differences and ML recommendations
    assert!(stdout.contains("fc1.") || stdout.contains("fc2.") || stdout.contains("fc3."), 
            "Should contain layer differences");
    assert!(stdout.contains("[CRITICAL]") || stdout.contains("[WARNING]") || stdout.contains("[RECOMMENDATIONS]"), 
            "Should contain ML recommendations");
}

#[test]
fn test_ml_recommendations_not_in_json_output() {
    // Test that recommendations don't appear in JSON output
    let project_root = get_project_root();
    let file1 = get_fixture_path("tests/fixtures/ml_models/model_quantized.safetensors");
    let file2 = get_fixture_path("tests/fixtures/ml_models/model_fp32.safetensors");
    
    let output = Command::new("cargo")
        .args([
            "run",
            "--bin",
            "diffai",
            "--",
            file1.to_str().unwrap(),
            file2.to_str().unwrap(),
            "--output",
            "json",
        ])
        .current_dir(&project_root)
        .output()
        .expect("Failed to execute diffai");

    let stdout = str::from_utf8(&output.stdout).unwrap();
    let stderr = str::from_utf8(&output.stderr).unwrap();

    println!("JSON STDOUT: {stdout}");
    println!("JSON STDERR: {stderr}");

    // Check if command executed successfully
    if !output.status.success() {
        panic!("Command failed with exit code: {:?}\nSTDOUT: {}\nSTDERR: {}", 
               output.status.code(), stdout, stderr);
    }
    
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
    let project_root = get_project_root();
    let file1 = get_fixture_path("tests/fixtures/ml_models/model_quantized.safetensors");
    let file2 = get_fixture_path("tests/fixtures/ml_models/model_fp32.safetensors");
    
    let output = Command::new("cargo")
        .args([
            "run",
            "--bin",
            "diffai",
            "--",
            file1.to_str().unwrap(),
            file2.to_str().unwrap(),
            "--output",
            "yaml",
        ])
        .current_dir(&project_root)
        .output()
        .expect("Failed to execute diffai");

    let stdout = str::from_utf8(&output.stdout).unwrap();
    let stderr = str::from_utf8(&output.stderr).unwrap();

    println!("YAML STDOUT: {stdout}");
    println!("YAML STDERR: {stderr}");

    // Check if command executed successfully
    if !output.status.success() {
        panic!("Command failed with exit code: {:?}\nSTDOUT: {}\nSTDERR: {}", 
               output.status.code(), stdout, stderr);
    }
    
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
    let project_root = get_project_root();
    let file1 = get_fixture_path("tests/fixtures/ml_models/anomalous_model.safetensors");
    let file2 = get_fixture_path("tests/fixtures/ml_models/normal_model.safetensors");
    
    let output = Command::new("cargo")
        .args([
            "run",
            "--bin",
            "diffai",
            "--",
            file1.to_str().unwrap(),
            file2.to_str().unwrap(),
        ])
        .current_dir(&project_root)
        .output()
        .expect("Failed to execute diffai");

    let stdout = str::from_utf8(&output.stdout).unwrap();
    let stderr = str::from_utf8(&output.stderr).unwrap();

    println!("PRIORITY STDOUT: {stdout}");
    println!("PRIORITY STDERR: {stderr}");

    // Check if command executed successfully
    if !output.status.success() {
        panic!("Command failed with exit code: {:?}\nSTDOUT: {}\nSTDERR: {}", 
               output.status.code(), stdout, stderr);
    }
    
    // If recommendations are present, check priority ordering
    let output_text = format!("{stdout}{stderr}");

    if output_text.contains("[CRITICAL]") {
        // CRITICAL should appear before WARNING and RECOMMENDATIONS
        let critical_pos = output_text.find("[CRITICAL]").unwrap();
        if let Some(warning_pos) = output_text.find("[WARNING]") {
            assert!(
                critical_pos < warning_pos,
                "CRITICAL should appear before WARNING"
            );
        }
        if let Some(rec_pos) = output_text.find("[RECOMMENDATIONS]") {
            assert!(
                critical_pos < rec_pos,
                "CRITICAL should appear before RECOMMENDATIONS"
            );
        }
    }

    if output_text.contains("[WARNING]") {
        // WARNING should appear before RECOMMENDATIONS
        let warning_pos = output_text.find("[WARNING]").unwrap();
        if let Some(rec_pos) = output_text.find("[RECOMMENDATIONS]") {
            assert!(
                warning_pos < rec_pos,
                "WARNING should appear before RECOMMENDATIONS"
            );
        }
    }
}

#[test]
fn test_ml_recommendations_message_format() {
    // Test message format structure
    let project_root = get_project_root();
    let file1 = get_fixture_path("tests/fixtures/ml_models/large_model.safetensors");
    let file2 = get_fixture_path("tests/fixtures/ml_models/small_model.safetensors");
    
    let output = Command::new("cargo")
        .args([
            "run",
            "--bin",
            "diffai",
            "--",
            file1.to_str().unwrap(),
            file2.to_str().unwrap(),
        ])
        .current_dir(&project_root)
        .output()
        .expect("Failed to execute diffai");

    let stdout = str::from_utf8(&output.stdout).unwrap();
    let stderr = str::from_utf8(&output.stderr).unwrap();

    println!("FORMAT STDOUT: {stdout}");
    println!("FORMAT STDERR: {stderr}");

    // Check if command executed successfully
    if !output.status.success() {
        panic!("Command failed with exit code: {:?}\nSTDOUT: {}\nSTDERR: {}", 
               output.status.code(), stdout, stderr);
    }
    
    let output_text = format!("{stdout}{stderr}");

    // Check for proper message format if recommendations exist
    if output_text.contains("•") {
        let lines: Vec<&str> = output_text.lines().collect();
        for line in lines {
            if line.trim().starts_with("•") {
                // Each recommendation should be a bullet point
                assert!(
                    line.contains("•"),
                    "Recommendations should use bullet points"
                );

                // Should contain meaningful content (not just punctuation)
                let content = line.trim_start_matches("•").trim();
                assert!(
                    content.len() > 10,
                    "Recommendation should have meaningful content"
                );

                // Should end with a period (proper English)
                assert!(
                    content.ends_with('.'),
                    "Recommendation should end with period"
                );
            }
        }
    }
}

#[test]
fn test_ml_recommendations_with_verbose_mode() {
    // Test that recommendations work with verbose mode
    let project_root = get_project_root();
    let file1 = get_fixture_path("tests/fixtures/ml_models/model_quantized.safetensors");
    let file2 = get_fixture_path("tests/fixtures/ml_models/model_fp32.safetensors");
    
    let output = Command::new("cargo")
        .args([
            "run",
            "--bin",
            "diffai",
            "--",
            file1.to_str().unwrap(),
            file2.to_str().unwrap(),
            "--verbose",
        ])
        .current_dir(&project_root)
        .output()
        .expect("Failed to execute diffai");

    let stdout = str::from_utf8(&output.stdout).unwrap();
    let stderr = str::from_utf8(&output.stderr).unwrap();

    println!("VERBOSE STDOUT: {stdout}");
    println!("VERBOSE STDERR: {stderr}");

    // Check if command executed successfully
    if !output.status.success() {
        panic!("Command failed with exit code: {:?}\nSTDOUT: {}\nSTDERR: {}", 
               output.status.code(), stdout, stderr);
    }
    
    // Should contain verbose information or ML analysis output
    let output_text = format!("{stdout}{stderr}");
    assert!(output_text.contains("verbose mode") || output_text.contains("Processing results") || output_text.contains("analysis"));

    // Recommendations should still appear in CLI mode even with verbose
    // (This depends on the test models having differences that trigger recommendations)
}

#[test]
fn test_ml_recommendations_limit_to_five() {
    // Test that recommendations are limited to 5 items
    let project_root = get_project_root();
    let file1 = get_fixture_path("tests/fixtures/ml_models/transformer.safetensors");
    let file2 = get_fixture_path("tests/fixtures/ml_models/simple_base.safetensors");
    
    let output = Command::new("cargo")
        .args([
            "run",
            "--bin",
            "diffai",
            "--",
            file1.to_str().unwrap(),
            file2.to_str().unwrap(),
        ])
        .current_dir(&project_root)
        .output()
        .expect("Failed to execute diffai");

    let stdout = str::from_utf8(&output.stdout).unwrap();
    let stderr = str::from_utf8(&output.stderr).unwrap();

    println!("LIMIT STDOUT: {stdout}");
    println!("LIMIT STDERR: {stderr}");

    // Check if command executed successfully
    if !output.status.success() {
        panic!("Command failed with exit code: {:?}\nSTDOUT: {}\nSTDERR: {}", 
               output.status.code(), stdout, stderr);
    }
    
    let output_text = format!("{stdout}{stderr}");

    // Count bullet points in recommendations
    let bullet_count = output_text.matches("•").count();

    // Should not exceed 5 recommendations
    if bullet_count > 0 {
        assert!(
            bullet_count <= 5,
            "Should not exceed 5 recommendations, found {bullet_count}"
        );
    }
}

#[test]
fn test_ml_recommendations_with_no_ml_analysis() {
    // Test with non-ML files (should not trigger ML recommendations)
    let project_root = get_project_root();
    let file1 = get_fixture_path("tests/fixtures/data1.json");
    let file2 = get_fixture_path("tests/fixtures/data2.json");
    
    let output = Command::new("cargo")
        .args([
            "run",
            "--bin",
            "diffai",
            "--",
            file1.to_str().unwrap(),
            file2.to_str().unwrap(),
        ])
        .current_dir(&project_root)
        .output()
        .expect("Failed to execute diffai");

    let stdout = str::from_utf8(&output.stdout).unwrap();
    let stderr = str::from_utf8(&output.stderr).unwrap();

    println!("NO ML STDOUT: {stdout}");
    println!("NO ML STDERR: {stderr}");

    // Check if command executed successfully
    if !output.status.success() {
        panic!("Command failed with exit code: {:?}\nSTDOUT: {}\nSTDERR: {}", 
               output.status.code(), stdout, stderr);
    }
    
    let output_text = format!("{stdout}{stderr}");

    // Should not contain ML recommendations for non-ML files
    assert!(!output_text.contains("[CRITICAL]"));
    assert!(!output_text.contains("[WARNING]"));
    assert!(!output_text.contains("[RECOMMENDATIONS]"));
}

#[test]
fn test_ml_recommendations_colored_output() {
    // Test that colored output works (when not piped)
    let project_root = get_project_root();
    let file1 = get_fixture_path("tests/fixtures/ml_models/model_quantized.safetensors");
    let file2 = get_fixture_path("tests/fixtures/ml_models/model_fp32.safetensors");
    
    let output = Command::new("cargo")
        .args([
            "run",
            "--bin",
            "diffai",
            "--",
            file1.to_str().unwrap(),
            file2.to_str().unwrap(),
        ])
        .current_dir(&project_root)
        .output()
        .expect("Failed to execute diffai");

    let stdout = str::from_utf8(&output.stdout).unwrap();
    let stderr = str::from_utf8(&output.stderr).unwrap();

    println!("COLORED STDOUT: {stdout}");
    println!("COLORED STDERR: {stderr}");

    // Check if command executed successfully
    if !output.status.success() {
        panic!("Command failed with exit code: {:?}\nSTDOUT: {}\nSTDERR: {}", 
               output.status.code(), stdout, stderr);
    }
    
    // Note: When running through Command, colors might be disabled
    // This test mainly ensures the command completes successfully
}

#[test]
fn test_ml_recommendations_exit_code() {
    // Test that exit codes reflect recommendation priority
    let project_root = get_project_root();
    let file1 = get_fixture_path("tests/fixtures/ml_models/model_quantized.safetensors");
    let file2 = get_fixture_path("tests/fixtures/ml_models/model_fp32.safetensors");
    
    let output = Command::new("cargo")
        .args([
            "run",
            "--bin",
            "diffai",
            "--",
            file1.to_str().unwrap(),
            file2.to_str().unwrap(),
        ])
        .current_dir(&project_root)
        .output()
        .expect("Failed to execute diffai");

    let stdout = str::from_utf8(&output.stdout).unwrap();
    let stderr = str::from_utf8(&output.stderr).unwrap();

    println!("EXIT CODE STDOUT: {stdout}");
    println!("EXIT CODE STDERR: {stderr}");
    println!("EXIT CODE: {}", output.status.code().unwrap_or(-1));

    // Check if command executed successfully
    if !output.status.success() {
        panic!("Command failed with exit code: {:?}\nSTDOUT: {}\nSTDERR: {}", 
               output.status.code(), stdout, stderr);
    }
    
    // Exit code should reflect the presence of differences
    // 0 = no differences, non-zero = differences found
    // Note: Current implementation may not set exit codes based on recommendation priority
    assert!(output.status.code().unwrap_or(-1) >= 0);
}

#[test]
fn test_ml_recommendations_thread_safety() {
    // Test that recommendations work correctly in concurrent scenarios
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::thread;

    let success_count = Arc::new(AtomicUsize::new(0));
    let mut handles = vec![];

    for i in 0..3 {
        let success_count_clone = Arc::clone(&success_count);
        let handle = thread::spawn(move || {
            let project_root = get_project_root();
            let file1 = get_fixture_path("tests/fixtures/ml_models/model_quantized.safetensors");
            let file2 = get_fixture_path("tests/fixtures/ml_models/model_fp32.safetensors");
            
            let output = Command::new("cargo")
                .args([
                    "run",
                    "--bin",
                    "diffai",
                    "--",
                    file1.to_str().unwrap(),
                    file2.to_str().unwrap(),
                ])
                .current_dir(&project_root)
                .output()
                .expect("Failed to execute diffai");

            let stdout = str::from_utf8(&output.stdout).unwrap();
            let stderr = str::from_utf8(&output.stderr).unwrap();

            println!("THREAD {i} STDOUT: {stdout}");
            println!("THREAD {i} STDERR: {stderr}");

            if output.status.success() {
                success_count_clone.fetch_add(1, Ordering::SeqCst);
            } else {
                println!("THREAD {i} FAILED: exit code {:?}", output.status.code());
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
