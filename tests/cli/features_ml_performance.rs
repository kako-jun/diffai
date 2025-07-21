use assert_cmd::prelude::*;
use std::process::Command;
use std::time::Instant;

// Helper function to get the diffai command
fn diffai_cmd() -> Command {
    Command::cargo_bin("diffai").expect("Failed to find diffai binary")
}

/// Test performance with large models
/// Verifies reasonable performance characteristics
#[test]
fn test_large_model_performance() -> Result<(), Box<dyn std::error::Error>> {
    let start = Instant::now();
    
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/large_model.pt")
        .arg("../tests/fixtures/ml_models/large_model.safetensors");

    let output = cmd.output()?;
    let duration = start.elapsed();
    
    assert!(output.status.success());
    
    // Should complete large model comparison within reasonable time (30 seconds)
    assert!(
        duration.as_secs() < 30, 
        "Large model comparison took too long: {:?}", 
        duration
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should provide some output for large models
    assert!(!stdout.trim().is_empty());

    Ok(())
}

/// Test memory usage scaling
/// Verifies memory usage remains reasonable with different model sizes
#[test]
fn test_memory_usage_scaling() -> Result<(), Box<dyn std::error::Error>> {
    // Test small model first
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/small_model.pt")
        .arg("../tests/fixtures/ml_models/small_model.safetensors")
        .arg("--verbose");

    let output = cmd.output()?;
    assert!(output.status.success());

    // Test large model
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/large_model.pt")
        .arg("../tests/fixtures/ml_models/large_model.safetensors")
        .arg("--verbose");

    let output = cmd.output()?;
    assert!(output.status.success());

    // Both should succeed, indicating proper memory management
    let stdout = String::from_utf8_lossy(&output.stdout);
    
    // In verbose mode, should show memory information
    if stdout.contains("memory") {
        assert!(
            stdout.contains("usage") 
            || stdout.contains("peak") 
            || stdout.contains("allocation")
        );
    }

    Ok(())
}

/// Test streaming processing efficiency
/// Verifies streaming mode works correctly for large files
#[test]
fn test_streaming_processing() -> Result<(), Box<dyn std::error::Error>> {
    // Force streaming mode with memory limit
    let mut cmd = diffai_cmd();
    cmd.env("DIFFAI_MAX_MEMORY", "100") // 100MB limit
        .arg("../tests/fixtures/ml_models/large_model.pt")
        .arg("../tests/fixtures/ml_models/transformer.pt")
        .arg("--verbose");

    let output = cmd.output()?;
    
    // Should succeed with streaming
    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        // Should indicate streaming was used
        assert!(
            stdout.contains("streaming") 
            || stdout.contains("batch") 
            || stdout.contains("chunk")
            || stdout.contains("memory")
        );
    } else {
        // If streaming not available, should give clear error
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains("memory") 
            || stderr.contains("limit") 
            || stderr.contains("resource")
        );
    }

    Ok(())
}

/// Test concurrent comparison performance
/// Verifies system handles multiple simultaneous comparisons
#[test]
fn test_concurrent_performance() -> Result<(), Box<dyn std::error::Error>> {
    use std::thread;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::time::Instant;
    
    let start = Instant::now();
    let success_count = Arc::new(AtomicU32::new(0));
    let mut handles = vec![];

    // Spawn multiple comparison threads
    for i in 0..3 {
        let success_count = Arc::clone(&success_count);
        
        let handle = thread::spawn(move || {
            let mut cmd = diffai_cmd();
            
            // Use different model pairs to avoid file locking
            match i {
                0 => {
                    cmd.arg("../tests/fixtures/ml_models/simple_base.pt")
                        .arg("../tests/fixtures/ml_models/simple_modified.pt");
                }
                1 => {
                    cmd.arg("../tests/fixtures/ml_models/checkpoint_epoch_0.pt")
                        .arg("../tests/fixtures/ml_models/checkpoint_epoch_10.pt");
                }
                _ => {
                    cmd.arg("../tests/fixtures/ml_models/model1.pt")
                        .arg("../tests/fixtures/ml_models/model2.pt");
                }
            }

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

    let duration = start.elapsed();
    let final_count = success_count.load(Ordering::SeqCst);
    
    // At least 2 out of 3 should succeed
    assert!(final_count >= 2, "Too few concurrent operations succeeded: {}", final_count);
    
    // Concurrent operations should not take excessively long
    assert!(duration.as_secs() < 60, "Concurrent operations took too long: {:?}", duration);

    Ok(())
}

/// Test analysis performance scaling
/// Verifies analysis complexity scales reasonably with model size
#[test]
fn test_analysis_performance_scaling() -> Result<(), Box<dyn std::error::Error>> {
    // Measure small model analysis time
    let start = Instant::now();
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/small_model.pt")
        .arg("../tests/fixtures/ml_models/small_model.safetensors")
        .arg("--output")
        .arg("json");

    let output = cmd.output()?;
    let small_duration = start.elapsed();
    assert!(output.status.success());

    // Measure large model analysis time
    let start = Instant::now();
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/large_model.pt")
        .arg("../tests/fixtures/ml_models/large_model.safetensors")
        .arg("--output")
        .arg("json");

    let output = cmd.output()?;
    let large_duration = start.elapsed();
    assert!(output.status.success());

    // Large model should not take disproportionately longer
    // Allow up to 10x longer for large models (reasonable scaling)
    let ratio = large_duration.as_millis() as f64 / small_duration.as_millis() as f64;
    assert!(
        ratio < 10.0, 
        "Large model analysis scaling is poor: {}x slower", 
        ratio
    );

    Ok(())
}

/// Test caching effectiveness
/// Verifies intelligent caching improves repeated comparisons
#[test]
fn test_caching_effectiveness() -> Result<(), Box<dyn std::error::Error>> {
    // First comparison (cold cache)
    let start = Instant::now();
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/transformer.pt")
        .arg("../tests/fixtures/ml_models/transformer.safetensors");

    let output = cmd.output()?;
    let first_duration = start.elapsed();
    assert!(output.status.success());

    // Second identical comparison (warm cache)
    let start = Instant::now();
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/transformer.pt")
        .arg("../tests/fixtures/ml_models/transformer.safetensors");

    let output = cmd.output()?;
    let second_duration = start.elapsed();
    assert!(output.status.success());

    // Second run should be faster (or at least not significantly slower)
    // Allow some variance but should show caching benefit
    let ratio = second_duration.as_millis() as f64 / first_duration.as_millis() as f64;
    assert!(
        ratio <= 1.5, 
        "Caching does not seem effective: second run was {}x of first", 
        ratio
    );

    Ok(())
}

/// Test output generation performance
/// Verifies different output formats have reasonable performance
#[test]
fn test_output_format_performance() -> Result<(), Box<dyn std::error::Error>> {
    let formats = ["cli", "json", "yaml"];
    let mut durations = Vec::new();

    for format in &formats {
        let start = Instant::now();
        let mut cmd = diffai_cmd();
        cmd.arg("../tests/fixtures/ml_models/simple_base.pt")
            .arg("../tests/fixtures/ml_models/simple_modified.pt")
            .arg("--output")
            .arg(format);

        let output = cmd.output()?;
        let duration = start.elapsed();
        
        assert!(output.status.success(), "Format {} failed", format);
        durations.push(duration);
        
        // Each format should complete within reasonable time
        assert!(
            duration.as_secs() < 10, 
            "Format {} took too long: {:?}", 
            format, 
            duration
        );
    }

    // No format should be dramatically slower than others
    let max_duration = durations.iter().max().unwrap();
    let min_duration = durations.iter().min().unwrap();
    let ratio = max_duration.as_millis() as f64 / min_duration.as_millis() as f64;
    
    assert!(
        ratio < 5.0, 
        "Output format performance varies too much: {}x difference", 
        ratio
    );

    Ok(())
}

/// Test memory cleanup effectiveness
/// Verifies proper memory cleanup after operations
#[test]
fn test_memory_cleanup() -> Result<(), Box<dyn std::error::Error>> {
    // Run multiple sequential comparisons
    for i in 0..5 {
        let mut cmd = diffai_cmd();
        cmd.arg("../tests/fixtures/ml_models/simple_base.pt")
            .arg("../tests/fixtures/ml_models/simple_modified.pt")
            .arg("--verbose");

        let output = cmd.output()?;
        assert!(output.status.success(), "Iteration {} failed", i);
    }

    // All iterations should complete successfully
    // If memory leaks exist, later iterations might fail
    Ok(())
}

/// Test optimization auto-trigger
/// Verifies automatic optimizations kick in for appropriate workloads
#[test]
fn test_optimization_auto_trigger() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/large_model.pt")
        .arg("../tests/fixtures/ml_models/large_model.safetensors")
        .arg("--verbose");

    let output = cmd.output()?;
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    
    // Should show optimization was triggered for large models
    assert!(
        stdout.contains("optimiz") 
        || stdout.contains("streaming") 
        || stdout.contains("batch")
        || stdout.contains("efficient")
    );

    Ok(())
}

/// Test graceful degradation under resource pressure
/// Verifies system remains stable under resource constraints
#[test]
fn test_graceful_degradation() -> Result<(), Box<dyn std::error::Error>> {
    // Simulate resource pressure with very low memory limit
    let mut cmd = diffai_cmd();
    cmd.env("DIFFAI_MAX_MEMORY", "10") // Very low 10MB limit
        .arg("../tests/fixtures/ml_models/large_model.pt")
        .arg("../tests/fixtures/ml_models/transformer.pt");

    let output = cmd.output()?;
    
    // Should either succeed with degraded performance or fail gracefully
    if output.status.success() {
        // If succeeded, should mention resource limitations
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            stdout.contains("limited") 
            || stdout.contains("reduced") 
            || stdout.contains("streaming")
        );
    } else {
        // If failed, should provide clear error about resources
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains("memory") 
            || stderr.contains("resource") 
            || stderr.contains("limit")
        );
    }

    Ok(())
}