# ML Analysis Features - Implementation Details

Complete reference for diffai's 11 automatically implemented ML analysis capabilities.

## Overview

diffai automatically provides comprehensive AI/ML analysis without requiring configuration. All analysis features are **standard capabilities** that activate based on file format, not optional flags.

**Design Philosophy**: ML analysis features are not "options" but standard functionality that diffai provides automatically for each file format based on technical feasibility.

## Core Analysis Features

### High Priority Features (Phase A3)

#### 1. Tensor Statistics Changed
**Function**: `TensorStatsChanged` - Comprehensive tensor statistical analysis

**Automatic Analysis**:
- **Mean/Standard Deviation**: Statistical moments of tensor values
- **Min/Max Values**: Value range analysis 
- **Shape Changes**: Dimensional modifications detection
- **Data Type**: Precision format analysis (fp32, fp16, int8, etc.)
- **Element Count**: Total parameter counting

**Example Output**:
```
TensorStatsChanged: fc1.weight
  Old: mean=-0.0002, std=0.0514, shape=[128, 256], dtype=float32
  New: mean=-0.0001, std=0.0716, shape=[128, 256], dtype=float32
```

**Implementation Notes**:
- Uses statistical computation on extracted tensor data
- Handles multiple data types (float32, float16, int8, etc.)
- Shape comparison with dimensional analysis

#### 2. Model Architecture Changed
**Function**: `ModelArchitectureChanged` - Model structure and architecture analysis

**Automatic Analysis**:
- **Layer Detection**: Automatic layer counting and type identification
- **Parameter Count**: Total parameter enumeration
- **Architecture Classification**: Layer type analysis (conv, linear, norm, attention, embedding)
- **Structural Changes**: Addition/removal of components

**Example Output**:
```
ModelArchitectureChanged: model
  Old: {layers: 12, parameters: 124439808, types: [conv, linear, norm]}
  New: {layers: 12, parameters: 124440064, types: [conv, linear, norm, attention]}
```

**Implementation Notes**:
- Heuristic analysis of parameter keys for layer type inference
- Parameter counting from tensor shapes
- Architecture comparison with detailed breakdown

#### 3. Weight Significant Change
**Function**: `WeightSignificantChange` - Significant parameter change detection

**Automatic Analysis**:
- **Change Threshold**: Default 0.01 threshold for significance detection
- **RMS Calculation**: Root-mean-square change computation
- **Layer-wise Analysis**: Per-layer significance assessment
- **Change Magnitude**: Quantitative change measurement

**Example Output**:
```
WeightSignificantChange: transformer.attention.query.weight
  Change Magnitude: 0.0234 (above threshold: 0.01)
```

**Implementation Notes**:
- Configurable threshold via `weight_threshold` option
- RMS-based change magnitude calculation
- Separate analysis for weights vs biases

#### 4. Memory Analysis
**Function**: `MemoryAnalysis` - Memory usage and optimization analysis

**Automatic Analysis**:
- **Memory Delta**: Exact memory change calculation
- **Breakdown by Component**: Tensor vs metadata memory attribution
- **Usage Estimation**: Total model memory footprint
- **Change Attribution**: Detailed memory change breakdown

**Example Output**:
```
MemoryAnalysis: memory_change
  Old: 487.2MB (tensors: 485.1MB, metadata: 2.1MB)
  New: memory_change: +12.5MB, breakdown: tensors: +12.3MB, metadata: +0.2MB
```

**Implementation Notes**:
- Estimates based on tensor sizes and JSON structure
- Separates tensor data from metadata overhead
- Memory-efficient calculation for large models

### Medium Priority Features (Phase A4)

#### 5. Learning Rate Changed
**Function**: `LearningRateChanged` - Learning rate tracking and analysis

**Automatic Analysis**:
- **Explicit Detection**: Direct learning rate parameter identification
- **Optimizer State**: Learning rate from optimizer metadata
- **Scheduler State**: Learning rate scheduler analysis
- **Implicit Inference**: Learning rate estimation from training metadata

**Example Output**:
```
LearningRateChanged: optimizer.learning_rate
  Old: 0.001, New: 0.0005 (scheduler: step_decay, epoch: 10)
```

**Implementation Notes**:
- Multiple detection strategies for comprehensive coverage
- Handles various optimizer formats (Adam, SGD, etc.)
- Scheduler state analysis when available
- **Improvement Potential**: Advanced pickle decoding for deeper optimizer analysis

#### 6. Convergence Analysis
**Function**: `ConvergenceAnalysis` - Training convergence pattern analysis

**Automatic Analysis**:
- **Loss Convergence**: Loss trend analysis (improving/stable/diverging)
- **Training Stability**: Gradient norm and parameter stability assessment
- **Epoch Progression**: Training progress tracking
- **Convergence Status**: Overall training convergence evaluation

**Example Output**:
```
ConvergenceAnalysis: convergence_patterns
  Old: evaluating
  New: loss: improving (trend: decreasing), stability: gradient_norm: stable, epoch: 10 → 11
```

**Implementation Notes**:
- Multi-factor analysis including loss, gradients, and epochs
- Trend analysis using simple linear regression
- Stability assessment across multiple metrics

#### 7. Gradient Analysis
**Function**: `GradientAnalysis` - Gradient pattern and flow analysis

**Automatic Analysis**:
- **Gradient Magnitudes**: Norm, maximum, and variance analysis
- **Gradient Distribution**: Sparsity and outlier detection
- **Gradient Flow**: Vanishing/exploding gradient detection and balance analysis
- **Flow Health Assessment**: Overall gradient quality evaluation

**Example Output**:
```
GradientAnalysis: gradient_magnitudes
  Old: norm: 0.018456, max: 0.145234, var: 0.000234
  New: total_norm: 0.021234 (+14.8%, increasing), max_gradient: 0.156789 (+8.0%)
```

**Implementation Notes**:
- Estimates gradients from weight changes and metadata
- Statistical analysis of gradient distributions
- Flow balance analysis across network layers
- **Improvement Potential**: Direct gradient tensor access for precise analysis

### Low Priority Features (Phase A5)

#### 8. Attention Analysis
**Function**: `AttentionAnalysis` - Transformer attention mechanism analysis

**Automatic Analysis**:
- **Attention Heads**: Head count, dimension, and pattern analysis
- **Weight Distribution**: Attention weight sparsity and entropy analysis
- **Multi-Head Configuration**: Layer count, self-attention ratio, dropout analysis
- **Position Encoding**: Encoding type detection (learned, sinusoidal, relative)

**Example Output**:
```
AttentionAnalysis: attention_heads
  Old: heads: 8, dim: 64, patterns: 4
  New: num_heads: 8 → 12, head_dim: 64 → 48, patterns: +query, +value
```

**Implementation Notes**:
- Heuristic detection of attention patterns from tensor shapes
- Entropy calculation for attention weight analysis
- Position encoding inference from model structure

#### 9. Ensemble Analysis
**Function**: `EnsembleAnalysis` - Multi-model ensemble analysis

**Automatic Analysis**:
- **Ensemble Composition**: Model count, types, and method detection
- **Voting Strategy**: Voting type, consensus threshold, weighting analysis
- **Model Weight Distribution**: Weight entropy, dominant model, variance analysis
- **Ensemble Method**: Detection of bagging, boosting, stacking, voting strategies

**Example Output**:
```
EnsembleAnalysis: ensemble_composition
  Old: models: 3, types: [svm, tree, neural], method: voting
  New: num_models: 3 → 5, model_types: +gradient_boosting, +logistic
```

**Implementation Notes**:
- Infers ensemble structure from model metadata
- Weight distribution analysis using entropy and variance
- Model type classification from parameter patterns

#### 10. Quantization Analysis
**Function**: `QuantizationAnalysis` - Model quantization and precision analysis

**Automatic Analysis**:
- **Quantization Precision**: Bit width, data type, layer coverage analysis
- **Quantization Methods**: Strategy, calibration, symmetric/asymmetric detection
- **Performance Impact**: Size reduction, accuracy impact, speed improvement estimation
- **Mixed Precision**: Detection and usage analysis

**Example Output**:
```
QuantizationAnalysis: quantization_precision
  Old: 32bit float32, layers: 0, mixed: false
  New: bit_width: 32 → 8, data_type: float32 → int8, quantized_layers: 8 (+8)
```

**Implementation Notes**:
- Infers quantization from tensor data types
- Estimates performance impact based on quantization ratio
- Mixed precision detection from heterogeneous data types

## Format Compatibility Matrix

| Feature | PyTorch (.pt/.pth) | Safetensors (.safetensors) | NumPy (.npy/.npz) | MATLAB (.mat) |
|---------|-------------------|---------------------------|-------------------|---------------|
| Tensor Statistics | ✅ Full | ✅ Full | ✅ Full | ✅ Full |
| Model Architecture | ✅ Full | ✅ Full | ✅ Basic | ✅ Basic |
| Weight Changes | ✅ Full | ✅ Full | ✅ Full | ✅ Full |
| Memory Analysis | ✅ Full | ✅ Full | ✅ Full | ✅ Full |
| Learning Rate | ✅ Full | ✅ Partial | ❌ N/A | ❌ N/A |
| Convergence | ✅ Full | ✅ Partial | ❌ N/A | ❌ N/A |
| Gradients | ✅ Full | ✅ Partial | ❌ N/A | ❌ N/A |
| Attention | ✅ Full | ✅ Partial | ❌ N/A | ❌ N/A |
| Ensemble | ✅ Full | ❌ Limited | ❌ N/A | ❌ N/A |
| Quantization | ✅ Full | ✅ Full | ✅ Limited | ✅ Limited |

## Technical Implementation Details

### Data Extraction Methods
- **PyTorch**: Heuristic binary analysis with pickle structure inference
- **Safetensors**: Direct tensor metadata and data extraction
- **NumPy/MATLAB**: Basic structure analysis with limited ML metadata

### Analysis Algorithms
- **Statistical Analysis**: Mean, std, min, max computation on tensor data
- **Trend Analysis**: Simple linear regression for convergence detection  
- **Entropy Calculation**: Information-theoretic measures for distributions
- **Heuristic Inference**: Pattern matching for architecture and method detection

### Performance Characteristics
- **Memory Efficient**: Streaming analysis for large models
- **Format Agnostic**: Unified analysis across all supported formats
- **Automatic Activation**: No configuration required, format-aware feature selection

## Integration Examples

### Basic Usage
```bash
# Automatic comprehensive analysis
diffai model_v1.pt model_v2.pt

# All 11 features activate automatically based on file content
# Output includes all applicable analyses for the detected format
```

### JSON Output for Automation
```bash
# Machine-readable output
diffai baseline.safetensors candidate.safetensors --output json
```

### CI/CD Integration
```yaml
# Automated model validation
- name: Model Analysis
  run: |
    diffai models/baseline.pt models/candidate.pt --output json > analysis.json
    # All 11 features analyzed automatically
```

## Constraints and Limitations

### Current Implementation Constraints
1. **PyTorch Files**: Heuristic analysis due to pickle format complexity
2. **Gradient Analysis**: Estimated from parameter changes, not direct gradient access
3. **Learning Rate Detection**: Limited by optimizer metadata availability
4. **Attention Analysis**: Inference-based, may not detect all attention patterns
5. **Ensemble Detection**: Relies on naming conventions and structure patterns

### Future Improvement Opportunities
1. **Advanced Pickle Decoding**: Full PyTorch pickle parsing for deeper analysis
2. **Direct Gradient Access**: Real gradient tensor analysis capability
3. **Enhanced Metadata Extraction**: Richer model metadata parsing
4. **Format-Specific Optimizations**: Specialized analysis per format
5. **Performance Metrics Integration**: Training metrics correlation analysis

## Design Philosophy

**Standard Feature Principle**: All ML analysis capabilities are standard features that activate automatically. diffai determines the appropriate level of analysis based on:

1. **File Format Capabilities**: What each format technically supports
2. **Data Availability**: What information is extractable from the model
3. **Analysis Feasibility**: What can be reliably computed

**No Configuration Required**: Users never need to specify which analyses to run. diffai automatically provides the maximum possible analysis for each file format.

## See Also

- [CLI Reference](cli-reference.md) - Command-line interface documentation
- [Format Support](formats.md) - Detailed format compatibility information  
- [Basic Usage](../user-guide/basic-usage.md) - Getting started guide
- [ML Workflows](../user-guide/ml-workflows.md) - Advanced usage patterns