use serde_json::Value;

use super::types::QuantizationMethods;

// Analyze quantization method changes
pub(crate) fn analyze_quantization_methods(
    old_obj: &serde_json::Map<String, Value>,
    new_obj: &serde_json::Map<String, Value>,
) -> Option<(String, String)> {
    let old_methods = extract_quantization_methods(old_obj)?;
    let new_methods = extract_quantization_methods(new_obj)?;

    let mut method_analysis = Vec::new();

    // Compare quantization strategy
    if old_methods.strategy != new_methods.strategy {
        method_analysis.push(format!(
            "strategy: {} -> {}",
            old_methods.strategy, new_methods.strategy
        ));
    }

    // Compare calibration method
    if old_methods.calibration_method != new_methods.calibration_method {
        method_analysis.push(format!(
            "calibration: {} -> {}",
            old_methods.calibration_method, new_methods.calibration_method
        ));
    }

    // Compare symmetric vs asymmetric quantization
    if old_methods.symmetric != new_methods.symmetric {
        method_analysis.push(format!(
            "symmetric: {} -> {}",
            old_methods.symmetric, new_methods.symmetric
        ));
    }

    // Compare per-channel vs per-tensor quantization
    if old_methods.per_channel != new_methods.per_channel {
        method_analysis.push(format!(
            "per_channel: {} -> {}",
            old_methods.per_channel, new_methods.per_channel
        ));
    }

    if method_analysis.is_empty() {
        return None;
    }

    let old_info = format!(
        "strategy: {}, calibration: {}, symmetric: {}, per_channel: {}",
        old_methods.strategy,
        old_methods.calibration_method,
        old_methods.symmetric,
        old_methods.per_channel
    );
    let new_info = method_analysis.join(", ");

    Some((old_info, new_info))
}

// Enhanced quantization methods extraction with advanced technique detection
pub(crate) fn extract_quantization_methods(
    obj: &serde_json::Map<String, Value>,
) -> Option<QuantizationMethods> {
    let mut strategy = "post_training".to_string();
    let mut calibration_method = "minmax".to_string();
    let mut symmetric = true;
    let mut per_channel = false;
    let mut advanced_techniques = Vec::new();
    let mut optimization_level = "basic".to_string();
    let mut hardware_compatibility = Vec::new();
    let mut calibration_dataset_size = None;

    // Enhanced quantization method detection (lawkit comprehensive analysis)
    for (key, value) in obj {
        let is_quantization_related = key.contains("quant")
            || key.contains("precision")
            || key.contains("optim")
            || key.contains("compress");

        if is_quantization_related {
            // Strategy detection
            if key.contains("strategy") || key.contains("method") {
                if let Value::String(strat) = value {
                    strategy = strat.clone();
                }
            } else if key.contains("calibration") {
                if let Value::String(calib) = value {
                    calibration_method = calib.clone();
                }
            } else if key.contains("symmetric") {
                if let Value::Bool(sym) = value {
                    symmetric = *sym;
                }
            } else if key.contains("per_channel") || key.contains("channel_wise") {
                if let Value::Bool(per_ch) = value {
                    per_channel = *per_ch;
                } else {
                    per_channel = true;
                }
            }

            // Advanced technique detection
            if key.contains("pruning") || key.contains("sparsity") {
                advanced_techniques.push("structured_pruning".to_string());
            }
            if key.contains("distillation") || key.contains("teacher") {
                advanced_techniques.push("knowledge_distillation".to_string());
            }
            if key.contains("smoothquant") || key.contains("smooth") {
                advanced_techniques.push("smoothquant".to_string());
            }
            if key.contains("gptq") || key.contains("group_wise") {
                advanced_techniques.push("gptq".to_string());
            }
            if key.contains("awq") || key.contains("activation_aware") {
                advanced_techniques.push("awq".to_string());
            }
            if key.contains("bnb") || key.contains("bitsandbytes") {
                advanced_techniques.push("bitsandbytes".to_string());
            }

            // Hardware compatibility detection
            if key.contains("cuda") || key.contains("gpu") {
                hardware_compatibility.push("cuda".to_string());
            }
            if key.contains("tensorrt") || key.contains("trt") {
                hardware_compatibility.push("tensorrt".to_string());
            }
            if key.contains("onnx") {
                hardware_compatibility.push("onnx".to_string());
            }
            if key.contains("openvino") {
                hardware_compatibility.push("openvino".to_string());
            }
            if key.contains("coreml") {
                hardware_compatibility.push("coreml".to_string());
            }

            // Calibration dataset size
            if key.contains("calibration") && key.contains("size") {
                if let Value::Number(size) = value {
                    calibration_dataset_size = size.as_u64().map(|s| s as usize);
                }
            }
        }
    }

    // Enhanced strategy inference from model structure
    if obj.contains_key("quantization_aware_training") || obj.contains_key("qat") {
        strategy = "quantization_aware_training".to_string();
        optimization_level = "advanced".to_string();
    } else if obj.contains_key("dynamic_quantization") {
        strategy = "dynamic".to_string();
        optimization_level = "intermediate".to_string();
    } else if obj.contains_key("static_quantization") {
        strategy = "static".to_string();
        optimization_level = "intermediate".to_string();
    } else if !advanced_techniques.is_empty() {
        optimization_level = "expert".to_string();
    }

    // Enhanced calibration method inference
    if obj.contains_key("entropy_calibration") || obj.contains_key("kl_divergence") {
        calibration_method = "entropy".to_string();
    } else if obj.contains_key("percentile_calibration") {
        calibration_method = "percentile".to_string();
    } else if obj.contains_key("mse_calibration") {
        calibration_method = "mse".to_string();
    } else if obj.contains_key("sqnr_calibration") {
        calibration_method = "sqnr".to_string();
    }

    // Deduplicate and sort techniques
    advanced_techniques.sort();
    advanced_techniques.dedup();
    hardware_compatibility.sort();
    hardware_compatibility.dedup();

    Some(QuantizationMethods {
        strategy,
        calibration_method,
        symmetric,
        per_channel,
        advanced_techniques,
        optimization_level,
        hardware_compatibility,
        calibration_dataset_size,
    })
}
