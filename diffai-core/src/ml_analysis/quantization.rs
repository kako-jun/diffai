use serde_json::Value;

use crate::types::DiffResult;

// A5-3: QUANTIZATION ANALYSIS - Low Priority ML Feature
// ============================================================================

// A5-3: QuantizationAnalysis - Model quantization and precision analysis
pub fn analyze_quantization_patterns(
    old_model: &Value,
    new_model: &Value,
    results: &mut Vec<DiffResult>,
) {
    if let (Value::Object(old_obj), Value::Object(new_obj)) = (old_model, new_model) {
        // Quantization precision analysis
        if let Some((old_prec, new_prec)) = analyze_quantization_precision(old_obj, new_obj) {
            results.push(DiffResult::ModelArchitectureChanged(
                "quantization_precision".to_string(),
                old_prec,
                new_prec,
            ));
        }
        
        // Quantization method analysis
        if let Some((old_method, new_method)) = analyze_quantization_methods(old_obj, new_obj) {
            results.push(DiffResult::ModelArchitectureChanged(
                "quantization_methods".to_string(),
                old_method,
                new_method,
            ));
        }
        
        // Quantization impact analysis
        if let Some((old_impact, new_impact)) = analyze_quantization_impact(old_obj, new_obj) {
            results.push(DiffResult::ModelArchitectureChanged(
                "quantization_impact".to_string(),
                old_impact,
                new_impact,
            ));
        }
    }
}

// Enhanced quantization precision analysis with mixed precision focus
fn analyze_quantization_precision(old_obj: &serde_json::Map<String, Value>, new_obj: &serde_json::Map<String, Value>) -> Option<(String, String)> {
    let old_quant = extract_quantization_info(old_obj)?;
    let new_quant = extract_quantization_info(new_obj)?;
    
    let mut precision_analysis = Vec::new();
    
    // Enhanced bit width comparison
    if old_quant.bit_width != new_quant.bit_width {
        precision_analysis.push(format!(
            "bit_width: {} -> {}",
            old_quant.bit_width, new_quant.bit_width
        ));
    }
    
    // Enhanced data type comparison
    if old_quant.data_type != new_quant.data_type {
        precision_analysis.push(format!(
            "data_type: {} -> {}",
            old_quant.data_type, new_quant.data_type
        ));
    }
    
    // Enhanced quantization coverage comparison
    if old_quant.quantization_coverage != new_quant.quantization_coverage {
        let coverage_change = (new_quant.quantization_coverage - old_quant.quantization_coverage) * 100.0;
        precision_analysis.push(format!(
            "coverage: {:.1}% ({:+.1}%)",
            new_quant.quantization_coverage * 100.0, coverage_change
        ));
    }
    
    // Enhanced mixed precision analysis
    if old_quant.mixed_precision != new_quant.mixed_precision {
        precision_analysis.push(format!(
            "mixed_precision: {} -> {}",
            old_quant.mixed_precision, new_quant.mixed_precision
        ));
    }
    
    // Precision distribution comparison
    let old_dist = &old_quant.precision_distribution;
    let new_dist = &new_quant.precision_distribution;
    
    if old_dist.fp32_layers != new_dist.fp32_layers {
        let change = new_dist.fp32_layers as i32 - old_dist.fp32_layers as i32;
        precision_analysis.push(format!("fp32_layers: {} ({:+})", new_dist.fp32_layers, change));
    }
    
    if old_dist.fp16_layers != new_dist.fp16_layers {
        let change = new_dist.fp16_layers as i32 - old_dist.fp16_layers as i32;
        precision_analysis.push(format!("fp16_layers: {} ({:+})", new_dist.fp16_layers, change));
    }
    
    if old_dist.int8_layers != new_dist.int8_layers {
        let change = new_dist.int8_layers as i32 - old_dist.int8_layers as i32;
        precision_analysis.push(format!("int8_layers: {} ({:+})", new_dist.int8_layers, change));
    }
    
    if old_dist.int4_layers != new_dist.int4_layers {
        let change = new_dist.int4_layers as i32 - old_dist.int4_layers as i32;
        precision_analysis.push(format!("int4_layers: {} ({:+})", new_dist.int4_layers, change));
    }
    
    // Precision efficiency score comparison
    if (old_dist.precision_efficiency_score - new_dist.precision_efficiency_score).abs() > 0.01 {
        let score_change = new_dist.precision_efficiency_score - old_dist.precision_efficiency_score;
        precision_analysis.push(format!(
            "efficiency_score: {:.3} ({:+.3})",
            new_dist.precision_efficiency_score, score_change
        ));
    }
    
    // Dynamic range analysis comparison
    if let (Some(old_range), Some(new_range)) = (&old_quant.dynamic_range_analysis, &new_quant.dynamic_range_analysis) {
        if (old_range.range_utilization - new_range.range_utilization).abs() > 0.05 {
            precision_analysis.push(format!(
                "range_utilization: {:.2} -> {:.2}",
                old_range.range_utilization, new_range.range_utilization
            ));
        }
        
        if (old_range.saturation_risk - new_range.saturation_risk).abs() > 0.05 {
            precision_analysis.push(format!(
                "saturation_risk: {:.2} -> {:.2}",
                old_range.saturation_risk, new_range.saturation_risk
            ));
        }
    }
    
    if precision_analysis.is_empty() {
        return None;
    }
    
    // Enhanced summary format
    let old_info = format!(
        "{}bit {}, coverage: {:.1}%, mixed: {}, efficiency: {:.3}",
        old_quant.bit_width,
        old_quant.data_type,
        old_quant.quantization_coverage * 100.0,
        old_quant.mixed_precision,
        old_quant.precision_distribution.precision_efficiency_score
    );
    let new_info = precision_analysis.join(", ");
    
    Some((old_info, new_info))
}

// Analyze quantization method changes
fn analyze_quantization_methods(old_obj: &serde_json::Map<String, Value>, new_obj: &serde_json::Map<String, Value>) -> Option<(String, String)> {
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

// Analyze quantization impact on model
fn analyze_quantization_impact(old_obj: &serde_json::Map<String, Value>, new_obj: &serde_json::Map<String, Value>) -> Option<(String, String)> {
    let old_impact = extract_quantization_impact(old_obj)?;
    let new_impact = extract_quantization_impact(new_obj)?;
    
    let mut impact_analysis = Vec::new();
    
    // Compare model size reduction
    if let (Some(old_size), Some(new_size)) = (old_impact.size_reduction, new_impact.size_reduction) {
        let size_change = new_size - old_size;
        if size_change.abs() > 0.01 {
            impact_analysis.push(format!(
                "size_reduction: {:.1}% ({:+.1}%)",
                new_size * 100.0, size_change * 100.0
            ));
        }
    }
    
    // Compare accuracy impact
    if let (Some(old_acc), Some(new_acc)) = (old_impact.accuracy_impact, new_impact.accuracy_impact) {
        let acc_change = new_acc - old_acc;
        if acc_change.abs() > 0.001 {
            let impact_trend = if acc_change > 0.0 {
                "degraded"
            } else {
                "improved"
            };
            impact_analysis.push(format!(
                "accuracy_impact: {:.3} ({:+.3}, {})",
                new_acc, acc_change, impact_trend
            ));
        }
    }
    
    // Compare speed improvement
    if let (Some(old_speed), Some(new_speed)) = (old_impact.speed_improvement, new_impact.speed_improvement) {
        let speed_change = new_speed - old_speed;
        if speed_change.abs() > 0.01 {
            impact_analysis.push(format!(
                "speed_improvement: {:.1}x ({:+.1}x)",
                new_speed, speed_change
            ));
        }
    }
    
    // Compare memory efficiency
    if let (Some(old_mem), Some(new_mem)) = (old_impact.memory_efficiency, new_impact.memory_efficiency) {
        let mem_change = new_mem - old_mem;
        if mem_change.abs() > 0.01 {
            impact_analysis.push(format!(
                "memory_efficiency: {:.1}% ({:+.1}%)",
                new_mem * 100.0, mem_change * 100.0
            ));
        }
    }
    
    if impact_analysis.is_empty() {
        return None;
    }
    
    let old_info = format!(
        "size: {:.1}%, acc_impact: {:.3}, speed: {:.1}x, mem: {:.1}%",
        old_impact.size_reduction.unwrap_or(0.0) * 100.0,
        old_impact.accuracy_impact.unwrap_or(0.0),
        old_impact.speed_improvement.unwrap_or(1.0),
        old_impact.memory_efficiency.unwrap_or(0.0) * 100.0
    );
    let new_info = impact_analysis.join(", ");
    
    Some((old_info, new_info))
}

// Enhanced quantization analysis structures with lawkit memory patterns
#[derive(Debug, Clone)]
struct QuantizationInfo {
    bit_width: u8,
    data_type: String,
    quantized_layers: usize,
    mixed_precision: bool,
    // Enhanced mixed precision analysis
    precision_distribution: PrecisionDistribution,
    quantization_coverage: f64,
    dynamic_range_analysis: Option<DynamicRangeInfo>,
}

#[derive(Debug, Clone)]
struct PrecisionDistribution {
    fp32_layers: usize,
    fp16_layers: usize,
    int8_layers: usize,
    int4_layers: usize,
    custom_precision_layers: usize,
    precision_efficiency_score: f64,
}

#[derive(Debug, Clone)]
struct DynamicRangeInfo {
    min_value: f64,
    max_value: f64,
    range_utilization: f64,
    saturation_risk: f64,
    precision_loss_estimate: f64,
}

#[derive(Debug, Clone)]
struct QuantizationMethods {
    strategy: String,
    calibration_method: String,
    symmetric: bool,
    per_channel: bool,
    // Enhanced method analysis
    advanced_techniques: Vec<String>,
    optimization_level: String,
    hardware_compatibility: Vec<String>,
    calibration_dataset_size: Option<usize>,
}

#[derive(Debug, Clone)]
struct QuantizationImpact {
    size_reduction: Option<f64>,
    accuracy_impact: Option<f64>,
    speed_improvement: Option<f64>,
    memory_efficiency: Option<f64>,
    // Enhanced impact analysis using lawkit patterns
    inference_latency_reduction: Option<f64>,
    bandwidth_savings: Option<f64>,
    energy_efficiency_gain: Option<f64>,
    compression_ratio: Option<f64>,
    quality_degradation_risk: f64,
}

// Enhanced quantization information extraction with lawkit streaming analysis
fn extract_quantization_info(obj: &serde_json::Map<String, Value>) -> Option<QuantizationInfo> {
    let mut bit_width = 32u8; // Default FP32
    let mut data_type = "float32".to_string();
    let mut quantized_layers = 0;
    let mut mixed_precision = false;
    
    // Enhanced precision distribution analysis (lawkit incremental pattern)
    let mut precision_dist = PrecisionDistribution {
        fp32_layers: 0,
        fp16_layers: 0,
        int8_layers: 0,
        int4_layers: 0,
        custom_precision_layers: 0,
        precision_efficiency_score: 0.0,
    };
    
    let mut dynamic_range_values = Vec::new();
    let mut total_layers = 0;
    
    // First pass: comprehensive precision analysis (diffx optimization pattern)
    for (key, value) in obj {
        let is_quantization_related = key.contains("quant") || key.contains("precision") || 
                                     key.contains("bit") || key.contains("weight") ||
                                     key.contains("bias") || key.contains("param");
        
        if is_quantization_related {
            // Extract explicit quantization metadata
            if key.contains("quant") || key.contains("precision") || key.contains("bit") {
                if key.contains("bit") || key.contains("width") {
                    if let Value::Number(bits) = value {
                        if let Some(bits_val) = bits.as_u64() {
                            bit_width = bits_val as u8;
                        }
                    }
                }
                
                if key.contains("dtype") || key.contains("type") {
                    if let Value::String(dtype) = value {
                        data_type = dtype.clone();
                    }
                }
                
                if key.contains("layer") && key.contains("quant") {
                    quantized_layers += 1;
                }
                
                if key.contains("mixed") || key.contains("amp") || key.contains("auto") {
                    mixed_precision = true;
                }
            }
            
            // Enhanced tensor-level analysis
            match value {
                Value::Object(tensor_obj) => {
                    total_layers += 1;
                    
                    // Analyze data type distribution
                    if let Some(Value::String(dtype)) = tensor_obj.get("dtype") {
                        data_type = dtype.clone();
                        
                        // Enhanced precision classification
                        match dtype.as_str() {
                            "float32" | "fp32" => {
                                precision_dist.fp32_layers += 1;
                                bit_width = 32;
                            }
                            "float16" | "fp16" | "half" => {
                                precision_dist.fp16_layers += 1;
                                bit_width = 16;
                                mixed_precision = true;
                            }
                            "int8" | "uint8" => {
                                precision_dist.int8_layers += 1;
                                bit_width = 8;
                                quantized_layers += 1;
                            }
                            "int4" | "uint4" => {
                                precision_dist.int4_layers += 1;
                                bit_width = 4;
                                quantized_layers += 1;
                            }
                            "int16" | "uint16" => {
                                bit_width = 16;
                                quantized_layers += 1;
                            }
                            "int32" | "uint32" => bit_width = 32,
                            "int64" | "uint64" | "float64" => bit_width = 64,
                            _ => {
                                precision_dist.custom_precision_layers += 1;
                                // Custom quantization scheme detected
                            }
                        }
                    }
                    
                    // Dynamic range analysis for quantization quality
                    if let (Some(Value::Number(min_val)), Some(Value::Number(max_val))) = 
                        (tensor_obj.get("min"), tensor_obj.get("max")) {
                        if let (Some(min_f), Some(max_f)) = (min_val.as_f64(), max_val.as_f64()) {
                            dynamic_range_values.push((min_f, max_f));
                        }
                    }
                }
                Value::Array(arr) => {
                    // Handle array of tensors efficiently
                    for item in arr {
                        if let Value::Object(tensor_obj) = item {
                            total_layers += 1;
                            if let Some(Value::String(dtype)) = tensor_obj.get("dtype") {
                                match dtype.as_str() {
                                    "float16" | "fp16" | "half" => {
                                        precision_dist.fp16_layers += 1;
                                        mixed_precision = true;
                                    }
                                    "int8" | "uint8" => {
                                        precision_dist.int8_layers += 1;
                                        quantized_layers += 1;
                                    }
                                    "int4" | "uint4" => {
                                        precision_dist.int4_layers += 1;
                                        quantized_layers += 1;
                                    }
                                    "float32" | "fp32" => precision_dist.fp32_layers += 1,
                                    _ => precision_dist.custom_precision_layers += 1,
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }
    
    // Enhanced mixed precision detection
    let has_multiple_precisions = (precision_dist.fp32_layers > 0) as u8 +
                                 (precision_dist.fp16_layers > 0) as u8 +
                                 (precision_dist.int8_layers > 0) as u8 +
                                 (precision_dist.int4_layers > 0) as u8 +
                                 (precision_dist.custom_precision_layers > 0) as u8;
    
    if has_multiple_precisions >= 2 {
        mixed_precision = true;
    }
    
    // Calculate precision efficiency score (lawkit statistical analysis)
    if total_layers > 0 {
        let quantized_ratio = quantized_layers as f64 / total_layers as f64;
        let mixed_bonus = if mixed_precision { 0.2 } else { 0.0 };
        precision_dist.precision_efficiency_score = quantized_ratio * 0.8 + mixed_bonus;
    }
    
    // Calculate quantization coverage
    let quantization_coverage = if total_layers > 0 {
        quantized_layers as f64 / total_layers as f64
    } else {
        0.0
    };
    
    // Dynamic range analysis
    let dynamic_range_analysis = if !dynamic_range_values.is_empty() {
        let (global_min, global_max) = dynamic_range_values.iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(min_acc, max_acc), &(min_val, max_val)| {
                (min_acc.min(min_val), max_acc.max(max_val))
            });
        
        let dynamic_range = global_max - global_min;
        let range_utilization = if dynamic_range > 0.0 {
            // Estimate how well the quantization range is utilized
            let effective_range = dynamic_range_values.iter()
                .map(|(min_val, max_val)| max_val - min_val)
                .sum::<f64>() / dynamic_range_values.len() as f64;
            effective_range / dynamic_range
        } else {
            1.0
        };
        
        // Saturation risk assessment
        let saturation_risk = if bit_width <= 8 {
            (1.0 - range_utilization) * 0.5 // Higher risk for low-bit quantization
        } else {
            (1.0 - range_utilization) * 0.2
        };
        
        // Precision loss estimate
        let precision_loss_estimate = match bit_width {
            4 => 0.15,  // Significant loss expected
            8 => 0.05,  // Moderate loss
            16 => 0.01, // Minimal loss
            _ => 0.0,   // FP32 or higher
        };
        
        Some(DynamicRangeInfo {
            min_value: global_min,
            max_value: global_max,
            range_utilization,
            saturation_risk,
            precision_loss_estimate,
        })
    } else {
        None
    };
    
    Some(QuantizationInfo {
        bit_width,
        data_type,
        quantized_layers,
        mixed_precision,
        precision_distribution: precision_dist,
        quantization_coverage,
        dynamic_range_analysis,
    })
}

// Enhanced quantization methods extraction with advanced technique detection
fn extract_quantization_methods(obj: &serde_json::Map<String, Value>) -> Option<QuantizationMethods> {
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
        let is_quantization_related = key.contains("quant") || key.contains("precision") ||
                                     key.contains("optim") || key.contains("compress");
        
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

// Enhanced quantization impact analysis with lawkit incremental statistics
fn extract_quantization_impact(obj: &serde_json::Map<String, Value>) -> Option<QuantizationImpact> {
    let mut size_reduction = None;
    let mut accuracy_impact = None;
    let mut speed_improvement = None;
    let mut memory_efficiency = None;
    let mut inference_latency_reduction = None;
    let mut bandwidth_savings = None;
    let mut energy_efficiency_gain = None;
    let mut compression_ratio = None;
    let mut quality_degradation_risk = 0.0;
    
    // Enhanced performance metrics collection (lawkit comprehensive analysis)
    let mut precision_stats = Vec::new();
    let mut tensor_sizes = Vec::new();
    let mut bit_width_distribution = std::collections::HashMap::new();
    
    for (key, value) in obj {
        // Direct performance metrics
        if key.contains("size") && key.contains("reduction") {
            if let Value::Number(reduction) = value {
                size_reduction = reduction.as_f64();
            }
        } else if key.contains("accuracy") && (key.contains("drop") || key.contains("impact") || key.contains("loss")) {
            if let Value::Number(acc_impact) = value {
                accuracy_impact = acc_impact.as_f64();
            }
        } else if key.contains("speed") || key.contains("latency") {
            if let Value::Number(perf) = value {
                if key.contains("improvement") || key.contains("gain") {
                    speed_improvement = perf.as_f64();
                } else if key.contains("reduction") {
                    inference_latency_reduction = perf.as_f64();
                }
            }
        } else if key.contains("memory") {
            if let Value::Number(mem_metric) = value {
                memory_efficiency = mem_metric.as_f64();
            }
        } else if key.contains("energy") && key.contains("efficiency") {
            if let Value::Number(energy) = value {
                energy_efficiency_gain = energy.as_f64();
            }
        } else if key.contains("bandwidth") {
            if let Value::Number(bw) = value {
                bandwidth_savings = bw.as_f64();
            }
        }
        
        // Collect tensor information for analysis
        if key.contains("weight") || key.contains("bias") || key.contains("param") {
            match value {
                Value::Object(tensor_obj) => {
                    // Precision analysis
                    if let Some(Value::String(dtype)) = tensor_obj.get("dtype") {
                        precision_stats.push(dtype.clone());
                        
                        // Bit width distribution tracking
                        let bit_width = match dtype.as_str() {
                            "int4" | "uint4" => 4,
                            "int8" | "uint8" => 8,
                            "int16" | "uint16" | "float16" | "half" => 16,
                            "int32" | "uint32" | "float32" => 32,
                            "int64" | "uint64" | "float64" => 64,
                            _ => 32, // Default
                        };
                        *bit_width_distribution.entry(bit_width).or_insert(0) += 1;
                    }
                    
                    // Tensor size analysis
                    if let Some(Value::Array(shape)) = tensor_obj.get("shape") {
                        let size = shape.iter()
                            .filter_map(|v| v.as_u64())
                            .product::<u64>() as f64;
                        tensor_sizes.push(size);
                    }
                }
                Value::Array(arr) => {
                    // Handle array of tensors
                    for item in arr {
                        if let Value::Object(tensor_obj) = item {
                            if let Some(Value::String(dtype)) = tensor_obj.get("dtype") {
                                precision_stats.push(dtype.clone());
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }
    
    // Enhanced estimation using lawkit incremental statistics
    if !precision_stats.is_empty() {
        let total_tensors = precision_stats.len() as f64;
        
        // Calculate precision distribution
        let quantized_count = precision_stats.iter()
            .filter(|dtype| dtype.contains("int") && !dtype.contains("32") && !dtype.contains("64"))
            .count() as f64;
        
        let fp16_count = precision_stats.iter()
            .filter(|dtype| dtype.contains("16") || dtype.contains("half"))
            .count() as f64;
        
        let int8_count = precision_stats.iter()
            .filter(|dtype| dtype.contains("int8"))
            .count() as f64;
        
        let int4_count = precision_stats.iter()
            .filter(|dtype| dtype.contains("int4"))
            .count() as f64;
        
        // Enhanced size reduction calculation
        if size_reduction.is_none() {
            let mut total_reduction = 0.0;
            
            // Precision-based reduction estimates
            total_reduction += (fp16_count / total_tensors) * 0.5;   // FP16: 50% reduction
            total_reduction += (int8_count / total_tensors) * 0.75;   // INT8: 75% reduction
            total_reduction += (int4_count / total_tensors) * 0.875;  // INT4: 87.5% reduction
            
            size_reduction = Some(total_reduction);
        }
        
        // Enhanced compression ratio calculation
        let base_size = tensor_sizes.iter().sum::<f64>() * 32.0; // Assume FP32 baseline
        let compressed_size = bit_width_distribution.iter()
            .map(|(&bits, &count)| (count as f64) * (bits as f64))
            .sum::<f64>();
        
        if base_size > 0.0 && compressed_size > 0.0 {
            compression_ratio = Some(base_size / compressed_size);
        }
        
        // Quality degradation risk assessment
        quality_degradation_risk = (int4_count / total_tensors) * 0.3 +  // High risk for INT4
                                  (int8_count / total_tensors) * 0.1 +   // Moderate risk for INT8
                                  (fp16_count / total_tensors) * 0.02;   // Low risk for FP16
        
        // Enhanced speed improvement estimation
        if speed_improvement.is_none() {
            let mut perf_improvement = 1.0;
            
            // Hardware-aware performance modeling
            if int8_count > 0.0 {
                perf_improvement += (int8_count / total_tensors) * 2.0; // INT8 can be 2-3x faster
            }
            if fp16_count > 0.0 {
                perf_improvement += (fp16_count / total_tensors) * 1.5; // FP16 ~1.5-2x faster
            }
            if int4_count > 0.0 {
                perf_improvement += (int4_count / total_tensors) * 3.0; // INT4 can be 3-4x faster
            }
            
            speed_improvement = Some(perf_improvement);
        }
        
        // Memory efficiency estimation
        if memory_efficiency.is_none() {
            memory_efficiency = size_reduction; // Memory savings roughly follow size reduction
        }
        
        // Inference latency reduction estimation
        if inference_latency_reduction.is_none() {
            let latency_factor = size_reduction.unwrap_or(0.0) * 0.8; // Conservative estimate
            inference_latency_reduction = Some(latency_factor);
        }
        
        // Bandwidth savings estimation
        if bandwidth_savings.is_none() {
            bandwidth_savings = size_reduction; // Bandwidth savings follow model size reduction
        }
        
        // Energy efficiency estimation
        if energy_efficiency_gain.is_none() {
            let energy_factor = (quantized_count / total_tensors) * 0.4; // Quantization reduces energy
            energy_efficiency_gain = Some(energy_factor);
        }
    }
    
    Some(QuantizationImpact {
        size_reduction,
        accuracy_impact,
        speed_improvement,
        memory_efficiency,
        inference_latency_reduction,
        bandwidth_savings,
        energy_efficiency_gain,
        compression_ratio,
        quality_degradation_risk,
    })
}
