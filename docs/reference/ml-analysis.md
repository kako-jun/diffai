# ML Analysis Functions

Comprehensive guide to diffai's 28 advanced machine learning analysis functions.

## Overview

diffai provides 28 specialized analysis functions designed specifically for machine learning model comparison and analysis. These functions cover everything from research and development to MLOps and deployment.

## Learning & Convergence Analysis (4 functions)

### 1. `--learning-progress` Learning Progress Tracking
Track and analyze learning progress between model checkpoints.

**Usage**:
```bash
diffai checkpoint_epoch_10.safetensors checkpoint_epoch_20.safetensors --learning-progress
```

**Output**:
```
+ learning_progress: trend=improving, magnitude=0.0543, speed=0.80
```

**Analysis Fields**:
- **trend**: `improving`, `degrading`, `stable`
- **magnitude**: Change magnitude (0.0-1.0)
- **speed**: Convergence speed (0.0-1.0)

### 2. `--convergence-analysis` Convergence Analysis
Evaluate model stability and convergence status.

**Usage**:
```bash
diffai model_before.safetensors model_after.safetensors --convergence-analysis
```

**Output**:
```
+ convergence_analysis: status=stable, stability=0.0234, action="Continue training"
```

### 3. `--anomaly-detection` Anomaly Detection
Detect abnormal patterns during training.

**Usage**:
```bash
diffai normal_model.safetensors anomalous_model.safetensors --anomaly-detection
```

**Output**:
```
[CRITICAL] anomaly_detection: type=gradient_explosion, severity=critical, affected=2 layers
```

### 4. `--gradient-analysis` Gradient Analysis
Analyze gradient characteristics and flow.

**Usage**:
```bash
diffai model1.safetensors model2.safetensors --gradient-analysis
```

## Architecture & Performance Analysis (4 functions)

### 5. `--architecture-comparison` Architecture Comparison
Compare model structures and designs.

**Usage**:
```bash
diffai resnet.safetensors transformer.safetensors --architecture-comparison
```

**Output**:
```
architecture_comparison: type1=cnn, type2=transformer, depth=50→24, differences=15
```

### 6. `--param-efficiency-analysis` Parameter Efficiency Analysis
Analyze parameter efficiency between models.

**Usage**:
```bash
diffai baseline.safetensors optimized.safetensors --param-efficiency-analysis
```

### 7. `--memory-analysis` Memory Analysis
Analyze memory usage and optimization opportunities.

**Usage**:
```bash
diffai small_model.safetensors large_model.safetensors --memory-analysis
```

### 8. `--inference-speed-estimate` Inference Speed Estimation
Estimate inference speed and performance characteristics.

**Usage**:
```bash
diffai model1.safetensors model2.safetensors --inference-speed-estimate
```

## MLOps & Deployment Support (7 functions)

### 9. `--deployment-readiness` Deployment Readiness Assessment
Evaluate deployment readiness and compatibility.

**Usage**:
```bash
diffai production.safetensors candidate.safetensors --deployment-readiness
```

**Output**:
```
[GRADUAL] deployment_readiness: readiness=0.75, strategy=gradual, risk=medium
```

### 10. `--regression-test` Regression Testing
Perform automated regression testing.

**Usage**:
```bash
diffai baseline.safetensors new_version.safetensors --regression-test
```

### 11. `--risk-assessment` Risk Assessment
Evaluate deployment risks and stability.

**Usage**:
```bash
diffai current.safetensors candidate.safetensors --risk-assessment
```

### 12. `--hyperparameter-impact` Hyperparameter Impact Analysis
Analyze the impact of hyperparameter changes.

**Usage**:
```bash
diffai model_lr_001.safetensors model_lr_0001.safetensors --hyperparameter-impact
```

### 13. `--learning-rate-analysis` Learning Rate Analysis
Analyze learning rate effects and optimization.

**Usage**:
```bash
diffai model1.safetensors model2.safetensors --learning-rate-analysis
```

### 14. `--alert-on-degradation` Performance Degradation Alerts
Alert on performance degradation beyond thresholds.

**Usage**:
```bash
diffai baseline.safetensors new_model.safetensors --alert-on-degradation
```

### 15. `--performance-impact-estimate` Performance Impact Estimation
Estimate performance impact of changes.

**Usage**:
```bash
diffai model1.safetensors model2.safetensors --performance-impact-estimate
```

## Experiment & Documentation Support (4 functions)

### 16. `--generate-report` Generate Comprehensive Reports
Generate comprehensive analysis reports automatically.

**Usage**:
```bash
diffai model1.safetensors model2.safetensors --generate-report
```

### 17. `--markdown-output` Markdown Output
Generate reports in markdown format.

**Usage**:
```bash
diffai model1.safetensors model2.safetensors --markdown-output
```

### 18. `--include-charts` Include Charts and Visualizations
Include visualization charts in output.

**Usage**:
```bash
diffai model1.safetensors model2.safetensors --include-charts
```

### 19. `--review-friendly` Review-Friendly Output
Generate output optimized for human review.

**Usage**:
```bash
diffai model1.safetensors model2.safetensors --review-friendly
```

## Advanced Analysis Functions (6 functions)

### 20. `--embedding-analysis` Embedding Analysis
Analyze embedding layer changes and semantic drift.

**Usage**:
```bash
diffai model1.safetensors model2.safetensors --embedding-analysis
```

### 21. `--similarity-matrix` Similarity Matrix
Generate similarity matrix for model comparison.

**Usage**:
```bash
diffai model1.safetensors model2.safetensors --similarity-matrix
```

### 22. `--clustering-change` Clustering Change Analysis
Analyze clustering changes in model representations.

**Usage**:
```bash
diffai model1.safetensors model2.safetensors --clustering-change
```

### 23. `--attention-analysis` Attention Analysis
Analyze attention mechanism patterns (Transformer models).

**Usage**:
```bash
diffai transformer1.safetensors transformer2.safetensors --attention-analysis
```

### 24. `--head-importance` Attention Head Importance
Analyze attention head importance and specialization.

**Usage**:
```bash
diffai model1.safetensors model2.safetensors --head-importance
```

### 25. `--attention-pattern-diff` Attention Pattern Differences
Compare attention patterns between models.

**Usage**:
```bash
diffai model1.safetensors model2.safetensors --attention-pattern-diff
```

## Additional Analysis Functions (3 functions)

### 26. `--quantization-analysis` Quantization Analysis
Analyze quantization effects and efficiency.

**Usage**:
```bash
diffai fp32.safetensors quantized.safetensors --quantization-analysis
```

**Output**:
```
quantization_analysis: compression=75.0%, speedup=2.5x, precision_loss=2.0%, suitability=good
```

### 27. `--sort-by-change-magnitude` Sort by Change Magnitude
Sort differences by magnitude for prioritization.

**Usage**:
```bash
diffai model1.safetensors model2.safetensors --sort-by-change-magnitude
```

### 28. `--change-summary` Change Summary
Generate detailed summaries of changes.

**Usage**:
```bash
diffai model1.safetensors model2.safetensors --change-summary
```

## Function Combinations

### For Training Monitoring
```bash
diffai checkpoint_old.safetensors checkpoint_new.safetensors \
  --learning-progress \
  --convergence-analysis \
  --anomaly-detection
```

### For Production Deployment
```bash
diffai current_prod.safetensors candidate.safetensors \
  --deployment-readiness \
  --risk-assessment \
  --regression-test
```

### For Research Analysis
```bash
diffai baseline.safetensors experiment.safetensors \
  --architecture-comparison \
  --embedding-analysis \
  --generate-report
```

### For Quantization Validation
```bash
diffai fp32.safetensors quantized.safetensors \
  --quantization-analysis \
  --memory-analysis \
  --performance-impact-estimate
```

## Related Documentation

- [CLI Reference](cli-reference.md) - Complete command-line options
- [Supported Formats](formats.md) - Supported file formats
- [Output Formats](output-formats.md) - Output format specifications

## Language Support

- **English**: Current documentation
- **日本語**: [Japanese version](ml-analysis_ja.md)