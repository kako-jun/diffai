# ML Analysis Functions - Technical Reference

Complete technical documentation for the 11 specialized ML analysis functions automatically executed by diffai when comparing PyTorch (.pt/.pth) or Safetensors (.safetensors) files.

## Overview

diffai follows the **Convention over Configuration** principle: all 11 ML analysis functions run automatically when AI/ML files are detected, providing comprehensive insights without requiring manual configuration. Built using lawkit memory-efficient patterns and diffx-core optimization techniques.

**Automatic Trigger Conditions:**
- **PyTorch files (.pt/.pth)**: All 11 analyses execute
- **Safetensors files (.safetensors)**: All 11 analyses execute  
- **NumPy/MATLAB files**: Basic tensor statistics only
- **Other formats**: Standard structural comparison via diffx-core

## 1. Learning Rate Analysis

**Function**: `analyze_learning_rate_changes()`  
**Purpose**: Track learning rate changes and training dynamics

### Detection Logic
```rust
// Automatically searches for learning rate fields in model data
let lr_fields = ["learning_rate", "lr", "step_size", "base_lr", "current_lr"];
// Analyzes changes in optimizer learning rate parameters
```

### Output Format
```bash
learning_rate_analysis: old=0.001, new=0.0015, change=+50.0%, trend=increasing
```

### JSON Output
```json
{
  "learning_rate_analysis": {
    "old": 0.001,
    "new": 0.0015,
    "change": "+50.0%",
    "trend": "increasing",
    "significance": "moderate"
  }
}
```

### Technical Implementation
- **Algorithm**: Direct value comparison with percentage calculation
- **Memory Efficiency**: lawkit incremental processing patterns
- **Thresholds**: >5% change considered significant
- **Error Handling**: Graceful fallback when LR fields not found

## 2. Optimizer Comparison

**Function**: `analyze_optimizer_comparison()`  
**Purpose**: Compare optimizer states and momentum information

### Detection Logic
```rust
// Searches for optimizer state dictionaries
let optimizer_fields = ["optimizer", "optimizer_state_dict", "optim", "momentum", "adam"];
// Analyzes momentum, beta parameters, and state evolution
```

### Output Format
```bash
optimizer_comparison: type=Adam, momentum_change=+2.1%, state_evolution=stable
```

### JSON Output
```json
{
  "optimizer_comparison": {
    "type": "Adam",
    "momentum_change": "+2.1%",
    "state_evolution": "stable",
    "beta1": 0.9,
    "beta2": 0.999
  }
}
```

### Technical Implementation
- **State Tracking**: Compares momentum buffers and optimizer parameters
- **Memory Optimization**: Streaming comparison for large optimizer states
- **Supported Optimizers**: Adam, SGD, AdamW, RMSprop (auto-detected)

## 3. Loss Tracking

**Function**: `analyze_loss_tracking()`  
**Purpose**: Analyze loss function evolution and convergence patterns

### Detection Logic
```rust
// Automatically detects loss-related fields
let loss_fields = ["loss", "train_loss", "val_loss", "epoch_loss", "step_loss"];
// Analyzes loss trends and convergence indicators
```

### Output Format
```bash
loss_tracking: loss_trend=decreasing, improvement_rate=15.2%, convergence_score=0.89
```

### JSON Output
```json
{
  "loss_tracking": {
    "loss_trend": "decreasing",
    "improvement_rate": "15.2%",
    "convergence_score": 0.89,
    "volatility": "low"
  }
}
```

### Technical Implementation
- **Trend Analysis**: Statistical trend detection using lawkit patterns
- **Convergence Scoring**: Multi-factor convergence assessment (0.0-1.0)
- **Volatility Detection**: Loss stability measurement

## 4. Accuracy Tracking

**Function**: `analyze_accuracy_tracking()`  
**Purpose**: Monitor accuracy changes and performance metrics

### Detection Logic
```rust
// Searches for accuracy and performance metrics
let accuracy_fields = ["accuracy", "acc", "top1", "top5", "f1_score", "precision", "recall"];
// Tracks performance metric evolution
```

### Output Format
```bash
accuracy_tracking: accuracy_delta=+3.2%, performance_trend=improving
```

### JSON Output
```json
{
  "accuracy_tracking": {
    "accuracy_delta": "+3.2%",
    "performance_trend": "improving",
    "baseline_accuracy": 0.847,
    "current_accuracy": 0.874
  }
}
```

### Technical Implementation
- **Multi-Metric Support**: Accuracy, F1, precision, recall auto-detection
- **Delta Calculation**: Precise percentage change computation
- **Trend Classification**: Statistical trend analysis (improving/stable/declining)

## 5. Model Version Analysis

**Function**: `analyze_model_version_detection()`  
**Purpose**: Identify model versioning and checkpoint information

### Detection Logic
```rust
// Detects version and checkpoint fields
let version_fields = ["version", "model_version", "epoch", "step", "checkpoint", "iteration"];
// Analyzes version evolution and checkpoint progression
```

### Output Format
```bash
model_version_analysis: version_change=1.0->1.1, checkpoint_evolution=incremental
```

### JSON Output
```json
{
  "model_version_analysis": {
    "version_change": "1.0->1.1",
    "checkpoint_evolution": "incremental",
    "epoch_progression": "5->10",
    "version_type": "semantic"
  }
}
```

### Technical Implementation
- **Version Detection**: Semantic and numeric version parsing
- **Evolution Analysis**: Incremental vs. major version changes
- **Checkpoint Tracking**: Training progression analysis

## 6. Gradient Analysis

**Function**: `analyze_gradient_analysis()`  
**Purpose**: Analyze gradient flow, vanishing/exploding gradients, and stability

### Detection Logic
```rust
// Enhanced gradient analysis with lawkit memory efficiency
struct EnhancedGradientStats {
    total_norm: Option<f64>,
    max_gradient: Option<f64>,
    variance: Option<f64>,
}
// Uses Welford's algorithm for incremental statistics
```

### Output Format
```bash
gradient_analysis: flow_health=healthy, norm=0.021069, variance_change=+15.3%
```

### JSON Output
```json
{
  "gradient_analysis": {
    "flow_health": "healthy",
    "gradient_norm": 0.021069,
    "variance_change": "+15.3%",
    "vanishing_risk": "low",
    "exploding_risk": "none"
  }
}
```

### Technical Implementation
- **Algorithm**: Welford's incremental statistics for memory efficiency
- **Health Classification**: healthy/warning/critical based on norm thresholds
- **Vanishing Detection**: Gradient magnitude < 1e-7 threshold
- **Exploding Detection**: Gradient magnitude > 100 threshold
- **Memory Optimization**: Streaming computation for large models

## 7. Quantization Analysis

**Function**: `analyze_quantization_analysis()`  
**Purpose**: Detect mixed precision (FP32/FP16/INT8/INT4) and compression effects

### Detection Logic
```rust
// Enhanced quantization analysis with mixed precision detection
struct QuantizationInfo {
    bit_width: u8,
    data_type: String,
    mixed_precision: bool,
    precision_distribution: PrecisionDistribution,
}
// Analyzes data types and compression ratios
```

### Output Format
```bash
quantization_analysis: mixed_precision=FP16+FP32, compression=12.5%, precision_loss=1.2%
```

### JSON Output
```json
{
  "quantization_analysis": {
    "mixed_precision": "FP16+FP32",
    "compression_ratio": "12.5%",
    "precision_loss": "1.2%",
    "quantization_coverage": 0.67,
    "efficiency_gain": "1.8x"
  }
}
```

### Technical Implementation
- **Precision Detection**: FP32, FP16, INT8, INT4 auto-detection
- **Dynamic Range Analysis**: Value range and precision impact assessment
- **Compression Calculation**: Memory reduction percentage
- **Quality Assessment**: Precision loss estimation

## 8. Convergence Analysis

**Function**: `analyze_convergence_analysis()`  
**Purpose**: Learning curve analysis, plateau detection, and optimization trajectory

### Detection Logic
```rust
// Comprehensive convergence analysis with lawkit incremental patterns
fn analyze_learning_curves_comprehensive(old_obj: &Map<String, Value>, new_obj: &Map<String, Value>) -> Value {
    // Multi-dimensional convergence analysis
    // Plateau detection algorithm
    // Optimization trajectory analysis
}
```

### Output Format
```bash
convergence_analysis: status=converging, stability=0.92, plateau_detected=false
```

### JSON Output
```json
{
  "convergence_analysis": {
    "status": "converging",
    "stability_score": 0.92,
    "plateau_detected": false,
    "convergence_rate": "moderate",
    "trajectory_health": "stable"
  }
}
```

### Technical Implementation
- **Learning Curve Analysis**: Multi-point trend analysis
- **Plateau Detection**: Statistical flatness detection over training windows
- **Stability Scoring**: Variance-based stability measurement (0.0-1.0)
- **Trajectory Analysis**: Optimization path assessment
- **Memory Efficiency**: Incremental computation using lawkit patterns

## 9. Activation Analysis

**Function**: `analyze_activation_analysis()`  
**Purpose**: Analyze activation function usage and distribution

### Detection Logic
```rust
// Searches for activation function indicators
let activation_indicators = ["relu", "gelu", "tanh", "sigmoid", "swish", "activation"];
// Analyzes activation function distribution and health
```

### Output Format
```bash
activation_analysis: relu_usage=45%, gelu_usage=55%, distribution=healthy
```

### JSON Output
```json
{
  "activation_analysis": {
    "relu_usage": "45%",
    "gelu_usage": "55%",
    "activation_distribution": "healthy",
    "saturation_risk": "low",
    "dead_neurons": 0
  }
}
```

### Technical Implementation
- **Function Detection**: Pattern matching for activation function names
- **Distribution Analysis**: Percentage usage across model layers
- **Health Assessment**: Saturation and dead neuron detection
- **Modern Activations**: Support for GELU, Swish, Mish, etc.

## 10. Attention Analysis

**Function**: `analyze_attention_analysis()`  
**Purpose**: Analyze transformer and attention mechanisms

### Detection Logic
```rust
// Detects transformer and attention components
let attention_indicators = ["attention", "attn", "self_attention", "multi_head", "transformer"];
// Analyzes attention patterns and efficiency
```

### Output Format
```bash
attention_analysis: head_count=12, attention_patterns=stable, efficiency=0.87
```

### JSON Output
```json
{
  "attention_analysis": {
    "head_count": 12,
    "attention_patterns": "stable",
    "efficiency_score": 0.87,
    "transformer_detected": true,
    "attention_mechanism": "multi_head"
  }
}
```

### Technical Implementation
- **Component Detection**: Multi-head attention, self-attention identification
- **Pattern Analysis**: Attention weight stability assessment
- **Efficiency Scoring**: Computational efficiency measurement
- **Architecture Recognition**: Transformer, BERT, GPT pattern detection

## 11. Ensemble Analysis

**Function**: `analyze_ensemble_analysis()`  
**Purpose**: Detect and analyze ensemble model structures

### Detection Logic
```rust
// Detects ensemble and multi-model structures
let ensemble_indicators = ["ensemble", "models", "sub_models", "branches", "heads"];
// Analyzes ensemble composition and coordination
```

### Output Format
```bash
ensemble_analysis: ensemble_detected=false, model_type=feedforward
```

### JSON Output
```json
{
  "ensemble_analysis": {
    "ensemble_detected": false,
    "model_type": "feedforward",
    "component_count": 1,
    "ensemble_method": "none",
    "diversity_score": 0.0
  }
}
```

### Technical Implementation
- **Structure Detection**: Multi-model and ensemble pattern recognition
- **Composition Analysis**: Component model identification
- **Diversity Assessment**: Model variation measurement in ensembles
- **Method Classification**: Bagging, boosting, stacking detection

## Performance and Memory Considerations

### Memory Efficiency
All analyses use **lawkit incremental processing patterns**:
- **Welford's Algorithm**: For statistical computations
- **Streaming Processing**: For large tensor analysis
- **Incremental Updates**: Memory-efficient progressive computation

### Optimization Techniques
- **diffx-core Integration**: Leverages proven diff engine reliability
- **Early Termination**: Skip analysis when data patterns not detected
- **Batch Processing**: Efficient handling of large model parameters
- **Caching**: Intermediate result caching for repeated computations

### Error Handling
- **Graceful Degradation**: Continues execution when specific patterns not found
- **Fallback Mechanisms**: Default values when analysis cannot complete
- **Robust Parsing**: Handles various model file formats and structures
- **Memory Limits**: Automatic optimization for large model files

## Usage Examples

### Automatic Execution
```rust
use diffai_core::diff;
use serde_json::json;

let old_model = json!(/* PyTorch model data */);
let new_model = json!(/* Updated model data */);

// All 11 ML analyses run automatically - no configuration needed
let results = diff(&old_model, &new_model, None)?;
```

### Integration with MLOps
```bash
# CI/CD pipeline integration
diffai baseline.safetensors candidate.safetensors --output json > analysis.json
# All 11 analyses included in JSON output for automated processing
```

## See Also

- [API Reference](api-reference.md) - Core API documentation
- [ML Analysis Guide](ml-analysis.md) - High-level analysis overview
- [User Guide](../user-guide/ml-model-comparison.md) - Practical usage examples