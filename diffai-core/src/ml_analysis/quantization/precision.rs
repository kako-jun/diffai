use serde_json::Value;

use super::types::{DynamicRangeInfo, PrecisionDistribution, QuantizationInfo};

// Enhanced quantization precision analysis with mixed precision focus
pub(crate) fn analyze_quantization_precision(
    old_obj: &serde_json::Map<String, Value>,
    new_obj: &serde_json::Map<String, Value>,
) -> Option<(String, String)> {
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
        let coverage_change =
            (new_quant.quantization_coverage - old_quant.quantization_coverage) * 100.0;
        precision_analysis.push(format!(
            "coverage: {:.1}% ({:+.1}%)",
            new_quant.quantization_coverage * 100.0,
            coverage_change
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
        precision_analysis.push(format!(
            "fp32_layers: {} ({:+})",
            new_dist.fp32_layers, change
        ));
    }

    if old_dist.fp16_layers != new_dist.fp16_layers {
        let change = new_dist.fp16_layers as i32 - old_dist.fp16_layers as i32;
        precision_analysis.push(format!(
            "fp16_layers: {} ({:+})",
            new_dist.fp16_layers, change
        ));
    }

    if old_dist.int8_layers != new_dist.int8_layers {
        let change = new_dist.int8_layers as i32 - old_dist.int8_layers as i32;
        precision_analysis.push(format!(
            "int8_layers: {} ({:+})",
            new_dist.int8_layers, change
        ));
    }

    if old_dist.int4_layers != new_dist.int4_layers {
        let change = new_dist.int4_layers as i32 - old_dist.int4_layers as i32;
        precision_analysis.push(format!(
            "int4_layers: {} ({:+})",
            new_dist.int4_layers, change
        ));
    }

    // Precision efficiency score comparison
    if (old_dist.precision_efficiency_score - new_dist.precision_efficiency_score).abs() > 0.01 {
        let score_change =
            new_dist.precision_efficiency_score - old_dist.precision_efficiency_score;
        precision_analysis.push(format!(
            "efficiency_score: {:.3} ({:+.3})",
            new_dist.precision_efficiency_score, score_change
        ));
    }

    // Dynamic range analysis comparison
    if let (Some(old_range), Some(new_range)) = (
        &old_quant.dynamic_range_analysis,
        &new_quant.dynamic_range_analysis,
    ) {
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

// Enhanced quantization information extraction with lawkit streaming analysis
pub(crate) fn extract_quantization_info(
    obj: &serde_json::Map<String, Value>,
) -> Option<QuantizationInfo> {
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
        let is_quantization_related = key.contains("quant")
            || key.contains("precision")
            || key.contains("bit")
            || key.contains("weight")
            || key.contains("bias")
            || key.contains("param");

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
                        (tensor_obj.get("min"), tensor_obj.get("max"))
                    {
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
    let has_multiple_precisions = (precision_dist.fp32_layers > 0) as u8
        + (precision_dist.fp16_layers > 0) as u8
        + (precision_dist.int8_layers > 0) as u8
        + (precision_dist.int4_layers > 0) as u8
        + (precision_dist.custom_precision_layers > 0) as u8;

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
        let (global_min, global_max) = dynamic_range_values.iter().fold(
            (f64::INFINITY, f64::NEG_INFINITY),
            |(min_acc, max_acc), &(min_val, max_val)| (min_acc.min(min_val), max_acc.max(max_val)),
        );

        let dynamic_range = global_max - global_min;
        let range_utilization = if dynamic_range > 0.0 {
            // Estimate how well the quantization range is utilized
            let effective_range = dynamic_range_values
                .iter()
                .map(|(min_val, max_val)| max_val - min_val)
                .sum::<f64>()
                / dynamic_range_values.len() as f64;
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
