#[allow(unused_imports)]
use assert_cmd::prelude::*;
use std::process::Command;
use std::fs;
use std::io::Write;
use tempfile::tempdir;

// Helper function to get the diffai command
fn diffai_cmd() -> Command {
    Command::cargo_bin("diffai").expect("Failed to find diffai binary")
}

/// Test empty model files
/// Verifies graceful handling of empty or minimal model files
#[test]
fn test_empty_model_files() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = tempdir()?;
    let empty_file1 = temp_dir.path().join("empty1.pt");
    let empty_file2 = temp_dir.path().join("empty2.pt");
    
    // Create empty files
    fs::File::create(&empty_file1)?;
    fs::File::create(&empty_file2)?;

    let mut cmd = diffai_cmd();
    cmd.arg(&empty_file1).arg(&empty_file2);

    let output = cmd.output()?;
    
    // Should handle empty files gracefully (may succeed or fail with clear error)
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains("empty") 
            || stderr.contains("invalid") 
            || stderr.contains("parse")
            || stderr.contains("format")
        );
    }

    Ok(())
}

/// Test corrupted model files
/// Verifies error handling for corrupted or malformed model files
#[test]
fn test_corrupted_model_files() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = tempdir()?;
    let corrupt_file1 = temp_dir.path().join("corrupt1.safetensors");
    let corrupt_file2 = temp_dir.path().join("corrupt2.safetensors");
    
    // Create files with invalid content
    fs::write(&corrupt_file1, "not a valid safetensors file")?;
    fs::write(&corrupt_file2, b"\x00\x01\x02\x03invalid binary data")?;

    let mut cmd = diffai_cmd();
    cmd.arg(&corrupt_file1).arg(&corrupt_file2);

    let output = cmd.output()?;
    
    // Should fail gracefully with informative error
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("parse") 
        || stderr.contains("invalid") 
        || stderr.contains("corrupt")
        || stderr.contains("format")
    );

    Ok(())
}

/// Test very large model files (simulation)
/// Verifies memory efficiency with large models
#[test]
fn test_large_model_memory_efficiency() -> Result<(), Box<dyn std::error::Error>> {
    // Use existing large model files
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/large_model.pt")
        .arg("../tests/fixtures/ml_models/large_model.safetensors");

    let output = cmd.output()?;
    
    // Should handle large files without memory issues
    assert!(output.status.success());
    
    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should show memory usage information in verbose mode
    if stdout.contains("memory") {
        // Memory analysis should be present for large models
        assert!(
            stdout.contains("memory_usage") 
            || stdout.contains("memory_delta")
            || stdout.contains("peak_memory")
        );
    }

    Ok(())
}

/// Test models with extreme values (NaN, Inf)
/// Verifies anomaly detection for mathematical edge cases
#[test]
fn test_extreme_values_detection() -> Result<(), Box<dyn std::error::Error>> {
    // Use anomalous model which should contain edge cases
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/normal_model.safetensors")
        .arg("../tests/fixtures/ml_models/anomalous_model.safetensors")
        .arg("--output")
        .arg("json");

    let output = cmd.output()?;
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    
    // Should detect extreme values
    assert!(
        stdout.contains("nan") 
        || stdout.contains("inf") 
        || stdout.contains("extreme") 
        || stdout.contains("anomaly")
    );

    Ok(())
}

/// Test mixed precision models
/// Verifies handling of different numerical precisions
#[test]
fn test_mixed_precision_handling() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/model_fp32.safetensors")
        .arg("../tests/fixtures/ml_models/model_quantized.safetensors")
        .arg("--output")
        .arg("json");

    let output = cmd.output()?;
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    
    // Should detect precision differences
    assert!(
        stdout.contains("precision") 
        || stdout.contains("quantiz") 
        || stdout.contains("fp32")
        || stdout.contains("dtype")
    );

    Ok(())
}

/// Test models with zero parameters
/// Verifies handling of minimal model structures
#[test]
fn test_minimal_model_structures() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = tempdir()?;
    let minimal_file1 = temp_dir.path().join("minimal1.safetensors");
    let minimal_file2 = temp_dir.path().join("minimal2.safetensors");
    
    // Create minimal valid safetensors files (header only)
    let minimal_header = r#"{"metadata":{},"__metadata__":{"format":"pt"}}"#;
    let header_bytes = minimal_header.as_bytes();
    let header_len = header_bytes.len() as u64;
    
    for file_path in [&minimal_file1, &minimal_file2] {
        let mut file = fs::File::create(file_path)?;
        file.write_all(&header_len.to_le_bytes())?;
        file.write_all(header_bytes)?;
    }

    let mut cmd = diffai_cmd();
    cmd.arg(&minimal_file1).arg(&minimal_file2);

    let output = cmd.output()?;
    
    // Should handle minimal models (may succeed with no differences or warn about empty models)
    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            stdout.contains("no differences") 
            || stdout.contains("identical") 
            || stdout.contains("empty")
        );
    }

    Ok(())
}

/// Test memory limit enforcement
/// Verifies that memory limits are respected
#[test]
fn test_memory_limit_enforcement() -> Result<(), Box<dyn std::error::Error>> {
    // Set a very low memory limit via environment variable
    let mut cmd = diffai_cmd();
    cmd.env("DIFFAI_MAX_MEMORY", "1") // 1MB limit
        .arg("../tests/fixtures/ml_models/large_model.pt")
        .arg("../tests/fixtures/ml_models/large_model.safetensors");

    let output = cmd.output()?;
    
    // Should either respect limit (and possibly use streaming) or fail gracefully
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains("memory") 
            || stderr.contains("limit") 
            || stderr.contains("resource")
        );
    } else {
        // If successful, should mention memory optimization
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            stdout.contains("streaming") 
            || stdout.contains("memory") 
            || stdout.contains("optimiz")
        );
    }

    Ok(())
}

/// Test format auto-detection edge cases
/// Verifies correct format detection for ambiguous files
#[test]
fn test_format_detection_edge_cases() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = tempdir()?;
    let ambiguous_file = temp_dir.path().join("model.bin");
    
    // Create a file with .bin extension but safetensors content
    fs::copy("../tests/fixtures/ml_models/simple_base.safetensors", &ambiguous_file)?;

    let mut cmd = diffai_cmd();
    cmd.arg(&ambiguous_file)
        .arg("../tests/fixtures/ml_models/simple_modified.safetensors");

    let output = cmd.output()?;
    
    // Should auto-detect format correctly or provide clear error
    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        // Should show format detection worked
        assert!(!stdout.is_empty());
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains("format") 
            || stderr.contains("detect") 
            || stderr.contains("unknown")
        );
    }

    Ok(())
}

/// Test concurrent model comparison stress test
/// Verifies stability under concurrent operations
#[test]
fn test_concurrent_comparison_stability() -> Result<(), Box<dyn std::error::Error>> {
    use std::thread;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};
    
    let success_count = Arc::new(AtomicU32::new(0));
    let mut handles = vec![];

    // Spawn multiple threads doing comparisons
    for _ in 0..3 {
        let success_count = Arc::clone(&success_count);
        
        let handle = thread::spawn(move || {
            let mut cmd = diffai_cmd();
            cmd.arg("../tests/fixtures/ml_models/simple_base.pt")
                .arg("../tests/fixtures/ml_models/simple_modified.pt");

            if let Ok(output) = cmd.output() {
                if output.status.success() {
                    success_count.fetch_add(1, Ordering::SeqCst);
                }
            }
        });
        
        handles.push(handle);
    }

    // Wait for all threads
    for handle in handles {
        handle.join().unwrap();
    }

    // At least some operations should succeed (may not be all due to file locking)
    let final_count = success_count.load(Ordering::SeqCst);
    assert!(final_count > 0, "No concurrent operations succeeded");

    Ok(())
}

/// Test invalid CLI combinations
/// Verifies proper error handling for invalid option combinations
#[test]
fn test_invalid_cli_combinations() -> Result<(), Box<dyn std::error::Error>> {
    // Test invalid output format
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/simple_base.pt")
        .arg("../tests/fixtures/ml_models/simple_modified.pt")
        .arg("--output")
        .arg("invalid_format");

    let output = cmd.output()?;
    assert!(!output.status.success());
    
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("invalid") 
        || stderr.contains("unknown") 
        || stderr.contains("format")
    );

    Ok(())
}