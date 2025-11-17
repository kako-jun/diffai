use serde_json::Value;

use super::types::QuantizationImpact;

// Analyze quantization impact on model
pub(crate) fn analyze_quantization_impact(old_obj: &serde_json::Map<String, Value>, new_obj: &serde_json::Map<String, Value>) -> Option<(String, String)> {
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

// Enhanced quantization impact analysis with lawkit incremental statistics
pub(crate) fn extract_quantization_impact(obj: &serde_json::Map<String, Value>) -> Option<QuantizationImpact> {
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
