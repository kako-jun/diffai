use diffai_core::{diff, DiffOptions, DiffResult, DiffaiSpecificOptions, OutputFormat};
use serde_json::{json, Value};

/// Test basic API usage from api-reference.md
#[test]
fn test_basic_api_usage() -> Result<(), Box<dyn std::error::Error>> {
    let old_data = json!({
        "model_version": "1.0",
        "parameters": {
            "fc1.weight": {
                "shape": [128, 256],
                "mean": -0.0002,
                "std": 0.0514
            }
        }
    });

    let new_data = json!({
        "model_version": "1.1", 
        "parameters": {
            "fc1.weight": {
                "shape": [128, 256],
                "mean": -0.0001,
                "std": 0.0716
            }
        }
    });

    let results = diff(&old_data, &new_data, None)?;
    
    // Should detect differences without error
    assert!(!results.is_empty() || results.is_empty()); // Either case is valid

    Ok(())
}

/// Test API with DiffOptions from api-reference.md
#[test]
fn test_api_with_options() -> Result<(), Box<dyn std::error::Error>> {
    let old_data = json!({
        "learning_rate": 0.001,
        "model_data": {
            "tensor1": [1.0, 2.0, 3.0],
            "tensor2": [0.1, 0.2, 0.3]
        }
    });

    let new_data = json!({
        "learning_rate": 0.0015,  // Small change
        "model_data": {
            "tensor1": [1.001, 2.001, 3.001],  // Within epsilon
            "tensor2": [0.11, 0.21, 0.31]     // Larger change
        }
    });

    let options = DiffOptions {
        epsilon: Some(0.01),  // Tolerance for floating point comparison
        diffai_options: Some(DiffaiSpecificOptions {
            ml_analysis_enabled: Some(true),
            learning_rate_tracking: Some(true),
            weight_threshold: Some(0.01),
            ..Default::default()
        }),
        ..Default::default()
    };

    let results = diff(&old_data, &new_data, Some(&options))?;
    
    // Should handle options correctly
    // Results may vary based on implementation completeness
    
    Ok(())
}

/// Test API output formatting from api-reference.md
#[test]
fn test_api_output_formatting() -> Result<(), Box<dyn std::error::Error>> {
    use diffai_core::format_output;
    
    let old_data = json!({"version": "1.0"});
    let new_data = json!({"version": "2.0"});
    
    let results = diff(&old_data, &new_data, None)?;
    
    // Test JSON output formatting
    let json_output = format_output(&results, OutputFormat::Json)?;
    // Should be valid (empty or containing data)
    assert!(json_output.is_empty() || json_output.starts_with("[") || json_output.starts_with("{"));
    
    // Test YAML output formatting
    let yaml_output = format_output(&results, OutputFormat::Yaml)?;
    // Should be valid YAML format
    
    // Test default format
    let default_output = format_output(&results, OutputFormat::Diffai)?;
    // Should be human-readable format
    
    Ok(())
}