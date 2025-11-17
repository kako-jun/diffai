// Enhanced quantization analysis structures with lawkit memory patterns
#[derive(Debug, Clone)]
pub(crate) struct QuantizationInfo {
    pub(crate) bit_width: u8,
    pub(crate) data_type: String,
    pub(crate) quantized_layers: usize,
    pub(crate) mixed_precision: bool,
    // Enhanced mixed precision analysis
    pub(crate) precision_distribution: PrecisionDistribution,
    pub(crate) quantization_coverage: f64,
    pub(crate) dynamic_range_analysis: Option<DynamicRangeInfo>,
}

#[derive(Debug, Clone)]
pub(crate) struct PrecisionDistribution {
    pub(crate) fp32_layers: usize,
    pub(crate) fp16_layers: usize,
    pub(crate) int8_layers: usize,
    pub(crate) int4_layers: usize,
    pub(crate) custom_precision_layers: usize,
    pub(crate) precision_efficiency_score: f64,
}

#[derive(Debug, Clone)]
pub(crate) struct DynamicRangeInfo {
    pub(crate) min_value: f64,
    pub(crate) max_value: f64,
    pub(crate) range_utilization: f64,
    pub(crate) saturation_risk: f64,
    pub(crate) precision_loss_estimate: f64,
}

#[derive(Debug, Clone)]
pub(crate) struct QuantizationMethods {
    pub(crate) strategy: String,
    pub(crate) calibration_method: String,
    pub(crate) symmetric: bool,
    pub(crate) per_channel: bool,
    // Enhanced method analysis
    pub(crate) advanced_techniques: Vec<String>,
    pub(crate) optimization_level: String,
    pub(crate) hardware_compatibility: Vec<String>,
    pub(crate) calibration_dataset_size: Option<usize>,
}

#[derive(Debug, Clone)]
pub(crate) struct QuantizationImpact {
    pub(crate) size_reduction: Option<f64>,
    pub(crate) accuracy_impact: Option<f64>,
    pub(crate) speed_improvement: Option<f64>,
    pub(crate) memory_efficiency: Option<f64>,
    // Enhanced impact analysis using lawkit patterns
    pub(crate) inference_latency_reduction: Option<f64>,
    pub(crate) bandwidth_savings: Option<f64>,
    pub(crate) energy_efficiency_gain: Option<f64>,
    pub(crate) compression_ratio: Option<f64>,
    pub(crate) quality_degradation_risk: f64,
}
