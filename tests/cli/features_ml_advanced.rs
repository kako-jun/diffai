#[allow(unused_imports)]
#[allow(unused_imports)]
use assert_cmd::prelude::*;
use std::process::Command;

// Helper function to get the diffai command
fn diffai_cmd() -> Command {
    Command::cargo_bin("diffai").expect("Failed to find diffai binary")
}

/// Test transfer learning analysis
/// Verifies detection of transfer learning patterns and pre-trained layer analysis
#[test]
fn test_transfer_learning_analysis() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/checkpoint_epoch_0.safetensors")
        .arg("../tests/fixtures/ml_models/checkpoint_epoch_50.safetensors")
        .arg("--output")
        .arg("json");

    let output = cmd.output()?;
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    
    // Check for transfer learning analysis indicators
    assert!(
        stdout.contains("transfer_learning") 
        || stdout.contains("pretrained_layers")
        || stdout.contains("fine_tuning")
        || stdout.contains("frozen_layers")
    );

    Ok(())
}

/// Test ensemble analysis capabilities
/// Verifies detection of ensemble patterns and multi-model compatibility
#[test] 
fn test_ensemble_analysis() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/model1.pt")
        .arg("../tests/fixtures/ml_models/model2.pt")
        .arg("--output")
        .arg("json");

    let output = cmd.output()?;
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    
    // Check for ensemble analysis indicators
    assert!(
        stdout.contains("ensemble_compatibility")
        || stdout.contains("model_diversity")  
        || stdout.contains("voting_readiness")
        || stdout.contains("bagging_potential")
    );

    Ok(())
}

/// Test embedding analysis
/// Verifies semantic drift detection and embedding space analysis
#[test]
fn test_embedding_analysis() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/transformer.safetensors")
        .arg("../tests/fixtures/ml_models/simple_modified.safetensors")
        .arg("--output")
        .arg("json");

    let output = cmd.output()?;
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    
    // Check for embedding analysis indicators
    assert!(
        stdout.contains("embedding_drift")
        || stdout.contains("semantic_shift")
        || stdout.contains("embedding_space")
        || stdout.contains("vector_similarity")
    );

    Ok(())
}

/// Test attention analysis for transformer models
/// Verifies attention pattern analysis and head importance scoring
#[test]
fn test_attention_analysis() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/transformer.pt")
        .arg("../tests/fixtures/ml_models/transformer.safetensors")
        .arg("--output")
        .arg("json");

    let output = cmd.output()?;
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    
    // Check for attention analysis indicators
    assert!(
        stdout.contains("attention_patterns")
        || stdout.contains("attention_heads")
        || stdout.contains("self_attention")
        || stdout.contains("attention_weights")
    );

    Ok(())
}

/// Test statistical significance analysis
/// Verifies statistical tests for model differences and confidence intervals
#[test]
fn test_statistical_significance() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/normal_model.safetensors")
        .arg("../tests/fixtures/ml_models/anomalous_model.safetensors")
        .arg("--output")
        .arg("json");

    let output = cmd.output()?;
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    
    // Check for statistical significance indicators
    assert!(
        stdout.contains("p_value")
        || stdout.contains("confidence_interval")
        || stdout.contains("statistical_test")
        || stdout.contains("significance_level")
    );

    Ok(())
}

/// Test risk assessment analysis
/// Verifies comprehensive risk evaluation for model changes
#[test]
fn test_risk_assessment_analysis() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/large_model.safetensors")
        .arg("../tests/fixtures/ml_models/small_model.safetensors") 
        .arg("--output")
        .arg("json");

    let output = cmd.output()?;
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    
    // Check for risk assessment indicators
    assert!(
        stdout.contains("risk_level")
        || stdout.contains("risk_factors")
        || stdout.contains("change_impact")
        || stdout.contains("deployment_risk")
    );

    Ok(())
}

/// Test performance impact analysis
/// Verifies inference speed and efficiency impact estimation
#[test]
fn test_performance_impact_analysis() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/model_fp32.safetensors")
        .arg("../tests/fixtures/ml_models/model_quantized.safetensors")
        .arg("--output")
        .arg("json");

    let output = cmd.output()?;
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    
    // Check for performance impact indicators
    assert!(
        stdout.contains("inference_speed")
        || stdout.contains("memory_usage")
        || stdout.contains("compute_efficiency")
        || stdout.contains("performance_delta")
    );

    Ok(())
}

/// Test learning progress analysis
/// Verifies training progression and milestone detection
#[test]
fn test_learning_progress_analysis() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/checkpoint_epoch_0.pt")
        .arg("../tests/fixtures/ml_models/checkpoint_epoch_50.pt")
        .arg("--output")
        .arg("json");

    let output = cmd.output()?;
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    
    // Check for learning progress indicators
    assert!(
        stdout.contains("learning_progress")
        || stdout.contains("training_milestone")
        || stdout.contains("epoch_comparison")
        || stdout.contains("convergence_trend")
    );

    Ok(())
}

/// Test comprehensive ML analysis integration
/// Verifies all ML analyses work together without conflicts
#[test]
fn test_comprehensive_ml_analysis_integration() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/simple_base.safetensors")
        .arg("../tests/fixtures/ml_models/simple_modified.safetensors")
        .arg("--output")
        .arg("json")
        .arg("--verbose");

    let output = cmd.output()?;
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    
    // Verify multiple analysis types are present
    let analysis_types = [
        "anomaly_detection",
        "architecture_comparison", 
        "memory_analysis",
        "convergence_analysis",
        "gradient_analysis",
        "risk_assessment",
        "performance_impact"
    ];

    let mut found_analyses = 0;
    for analysis_type in &analysis_types {
        if stdout.contains(analysis_type) {
            found_analyses += 1;
        }
    }

    // Should find at least 3 different analysis types in comprehensive mode
    assert!(found_analyses >= 3, "Expected at least 3 analysis types, found {}", found_analyses);

    Ok(())
}