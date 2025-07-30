# ML Analysis Functions

diffai automatically runs **11 specialized ML analysis functions** when comparing PyTorch (.pt/.pth) or Safetensors (.safetensors) files. No configuration required - follows Convention over Configuration principle.

## Automatic Execution

### When ML Analysis Runs
- **PyTorch files (.pt/.pth)**: All 11 analyses execute automatically
- **Safetensors files (.safetensors)**: All 11 analyses execute automatically
- **NumPy/MATLAB files**: Basic tensor statistics only
- **Other formats**: Standard structural comparison via diffx-core

### Zero Configuration
```bash
# All 11 ML analysis functions run automatically
diffai baseline.safetensors finetuned.safetensors

# No flags needed - diffai detects AI/ML files and runs comprehensive analysis
```

## The 11 ML Analysis Functions

### 1. Learning Rate Analysis
**Purpose**: Track learning rate changes and training dynamics

**Output Example**:
```bash
learning_rate_analysis: old=0.001, new=0.0015, change=+50.0%, trend=increasing
```

**What it detects**:
- Learning rate parameter changes
- Training schedule adjustments
- Adaptive learning rate modifications
- Trend analysis (increasing/decreasing/stable)

### 2. Optimizer Comparison
**Purpose**: Compare optimizer states and momentum information

**Output Example**:
```bash
optimizer_comparison: type=Adam, momentum_change=+2.1%, state_evolution=stable
```

**What it detects**:
- Optimizer type (Adam, SGD, AdamW, RMSprop)
- Momentum buffer changes
- Beta parameter evolution
- Optimizer state consistency

### 3. Loss Tracking
**Purpose**: Analyze loss function evolution and convergence patterns

**Output Example**:
```bash
loss_tracking: loss_trend=decreasing, improvement_rate=15.2%, convergence_score=0.89
```

**What it detects**:
- Loss trend direction
- Rate of improvement
- Convergence indicators
- Training stability

### 4. Accuracy Tracking
**Purpose**: Monitor accuracy changes and performance metrics

**Output Example**:
```bash
accuracy_tracking: accuracy_delta=+3.2%, performance_trend=improving
```

**What it detects**:
- Accuracy/F1/precision/recall changes
- Performance trend analysis
- Metric improvement rates
- Multi-metric support

### 5. Model Version Analysis
**Purpose**: Identify model versioning and checkpoint information

**Output Example**:
```bash
model_version_analysis: version_change=1.0->1.1, checkpoint_evolution=incremental
```

**What it detects**:
- Version number changes
- Checkpoint progression
- Epoch/iteration tracking
- Semantic vs numeric versioning

### 6. Gradient Analysis
**Purpose**: Analyze gradient flow, vanishing/exploding gradients, and stability

**Output Example**:
```bash
gradient_analysis: flow_health=healthy, norm=0.021069, variance_change=+15.3%
```

**What it detects**:
- Gradient flow health (healthy/warning/critical)
- Vanishing gradient detection (< 1e-7)
- Exploding gradient detection (> 100)
- Gradient variance and stability
- Uses lawkit memory-efficient incremental statistics

### 7. Quantization Analysis
**Purpose**: Detect mixed precision (FP32/FP16/INT8/INT4) and compression effects

**Output Example**:
```bash
quantization_analysis: mixed_precision=FP16+FP32, compression=12.5%, precision_loss=1.2%
```

**What it detects**:
- Mixed precision usage (FP32, FP16, INT8, INT4)
- Compression ratios
- Precision loss estimation
- Quantization coverage across model
- Memory efficiency gains

### 8. Convergence Analysis
**Purpose**: Learning curve analysis, plateau detection, and optimization trajectory

**Output Example**:
```bash
convergence_analysis: status=converging, stability=0.92, plateau_detected=false
```

**What it detects**:
- Convergence status (converging/converged/diverging)
- Learning curve patterns
- Plateau detection in training
- Stability scoring (0.0-1.0)
- Optimization trajectory health

### 9. Activation Analysis
**Purpose**: Analyze activation function usage and distribution

**Output Example**:
```bash
activation_analysis: relu_usage=45%, gelu_usage=55%, distribution=healthy
```

**What it detects**:
- Activation function types (ReLU, GELU, Tanh, Sigmoid, Swish)
- Usage distribution across layers
- Saturation risk assessment
- Dead neuron detection
- Modern activation support

### 10. Attention Analysis
**Purpose**: Analyze transformer and attention mechanisms

**Output Example**:
```bash
attention_analysis: head_count=12, attention_patterns=stable, efficiency=0.87
```

**What it detects**:
- Multi-head attention structures
- Attention pattern stability
- Transformer component identification
- Attention efficiency scoring
- BERT/GPT/T5 architecture recognition

### 11. Ensemble Analysis
**Purpose**: Detect and analyze ensemble model structures

**Output Example**:
```bash
ensemble_analysis: ensemble_detected=false, model_type=feedforward
```

**What it detects**:
- Ensemble model detection
- Component model counting
- Ensemble methods (bagging, boosting, stacking)
- Model diversity scoring
- Single vs multi-model classification

## Output Formats

### CLI Output (Default)
Human-readable with color coding and intuitive symbols.

### JSON Output (MLOps Integration)
```bash
diffai model1.safetensors model2.safetensors --output json
```

```json
{
  "learning_rate_analysis": {
    "old": 0.001,
    "new": 0.0015,
    "change": "+50.0%",
    "trend": "increasing"
  },
  "gradient_analysis": {
    "flow_health": "healthy",
    "gradient_norm": 0.021069,
    "variance_change": "+15.3%"
  }
  // ... all 11 analyses included
}
```

### YAML Output (Reports)
```bash
diffai model1.safetensors model2.safetensors --output yaml
```

## Technical Implementation

### Memory Efficiency
- **lawkit patterns**: Incremental statistics using Welford's algorithm
- **Streaming processing**: For large model analysis
- **diffx-core foundation**: Proven diff engine reliability

### Error Handling
- **Graceful degradation**: Continues when specific patterns not found
- **Robust parsing**: Handles various model file structures
- **Fallback mechanisms**: Default values when analysis cannot complete

### Performance Optimization
- **Early termination**: Skips analysis when data patterns not detected
- **Batch processing**: Efficient handling of large model parameters
- **Memory limits**: Automatic optimization for large files

## Use Cases

### Research & Development
Monitor training progress, detect convergence issues, analyze architectural changes.

### MLOps & CI/CD
Automated model validation, regression detection, performance monitoring.

### Model Optimization
Quantization analysis, memory usage tracking, compression assessment.

### Experiment Tracking  
Compare model variants, track hyperparameter effects, validate improvements.

## See Also

- **[Quick Start](quick-start.md)** - Get started in 5 minutes
- **[API Reference](reference/api-reference.md)** - Use in your code
- **[Examples](examples/)** - Real usage examples and outputs
- **[Technical Details](reference/ml-analysis-detailed.md)** - Implementation specifics