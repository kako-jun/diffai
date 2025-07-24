# ML/AI Workflows

Integration guide for machine learning and AI development with diffai.

## ML Development Use Cases

### 1. Model Development & Improvement

```bash
# Compare new architecture with baseline (comprehensive analysis automatic)
diffai baseline/resnet18.pth experiment/resnet34.pth

# Before/after fine-tuning comparison (comprehensive analysis automatic)
diffai pretrained/model.pth finetuned/model.pth
```

### 2. Experiment Management

```bash
# Compare experiment results (automatic directory detection)
diffai experiment_001/ experiment_002/ --include "*.json"

# Check hyperparameter differences
diffai config/baseline.yaml config/experiment.yaml
```

### 3. Model Optimization

```bash
# Compare before/after quantization
diffai original/model.pth quantized/model.pth --show-structure

# Check pruning effects
diffai full/model.pth pruned/model.pth --diff-only
```

## Directory-Based ML Workflows

### Automatic Directory Processing

diffai automatically handles directory comparisons for ML workflows without requiring special flags:

```bash
# Compare entire experiment directories
diffai baseline_experiment/ new_experiment/

# Automatic detection of model files, configs, and results
diffai run_001/ run_002/

# Filter specific ML file types
diffai checkpoint_dir_A/ checkpoint_dir_B/ --include "*.pth" --include "*.safetensors"
```

**ML-Specific Benefits:**
- **Model file detection**: Automatically finds .pth, .safetensors, .pt files
- **Config comparison**: Compares YAML/JSON configuration files
- **Results analysis**: Processes metrics, logs, and output files
- **Batch processing**: Handles multiple model checkpoints efficiently

## Typical Workflow

### Experiment Cycle

```bash
# 1. Run baseline experiment
python train.py --config baseline.yaml --output baseline/

# 2. Run new experiment
python train.py --config experiment.yaml --output experiment/

# 3. Compare results (automatic directory detection)
diffai baseline/ experiment/ --include "*.json" --include "*.pth"

# 4. Detailed analysis (comprehensive analysis automatic)
diffai baseline/model.pth experiment/model.pth
```

### Model Comparison Report

```bash
# Generate comprehensive comparison report (30+ analysis features automatic)
diffai baseline/model.pth experiment/model.pth --output json > comparison.json

# Visualize report
python scripts/visualize_comparison.py comparison.json
```

## Practical Examples

### PyTorch Model Evolution Tracking

```bash
# Track model changes across epochs
for checkpoint in checkpoints/epoch_*.pth; do
  echo "=== Epoch $(basename $checkpoint .pth | cut -d_ -f2) ==="
  diffai checkpoints/epoch_1.pth $checkpoint --show-structure --diff-only
done
```

**Example Output:**
```
=== Epoch 5 ===
Model Changes:
  No structural changes
  Parameters: 11,689,512 -> 11,689,512 (0% change)
  Weight updates: 94.3% of parameters modified

=== Epoch 10 ===
Model Changes:
  No structural changes  
  Parameters: 11,689,512 -> 11,689,512 (0% change)
  Weight updates: 87.1% of parameters modified
```

### Safetensors Format Comparison

```bash
# Compare Hugging Face format models
diffai model_v1/model.safetensors model_v2/model.safetensors --tensor-details

# Focus on specific layers
diffai model_v1/model.safetensors model_v2/model.safetensors --filter "attention.*"
```

### Dataset Change Tracking

```bash
# Track dataset changes
diffai dataset_v1.csv dataset_v2.csv --format json

# Compare train/validation data distributions
diffai train.csv val.csv --show-distribution
```

## Continuous Integration

### GitHub Actions Integration

```yaml
name: ML Model Comparison

on:
  pull_request:
    paths:
      - 'models/**'
      - 'experiments/**'

jobs:
  compare:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Install diffai
      run: cargo install diffai
    
    - name: Compare models
      run: |
        # Compare PR branch model with main branch model
        git checkout main
        cp models/current.pth /tmp/main_model.pth
        git checkout -
        
        diffai /tmp/main_model.pth models/current.pth --format json > model_diff.json
        
    - name: Comment PR
      uses: actions/github-script@v6
      with:
        script: |
          const fs = require('fs');
          const diff = JSON.parse(fs.readFileSync('model_diff.json', 'utf8'));
          
          const comment = `## Model Comparison Report
          
          **Parameter Count:** ${diff.comparison.param_diff}
          **Structure Changes:** ${diff.comparison.structure_changes}
          **Significant Changes:** ${diff.comparison.significant_changes}
          
          <details>
          <summary>Detailed Comparison</summary>
          
          \`\`\`json
          ${JSON.stringify(diff, null, 2)}
          \`\`\`
          </details>`;
          
          github.rest.issues.createComment({
            issue_number: context.issue.number,
            owner: context.repo.owner,
            repo: context.repo.repo,
            body: comment
          });
```

### MLflow Integration

```python
# mlflow_integration.py
import mlflow
import subprocess
import json

def compare_models_with_mlflow(run_id1, run_id2):
    """Compare models between MLflow runs"""
    
    # Download models from MLflow
    model1_path = mlflow.artifacts.download_artifacts(
        run_id=run_id1, artifact_path="model/model.pth"
    )
    model2_path = mlflow.artifacts.download_artifacts(
        run_id=run_id2, artifact_path="model/model.pth"
    )
    
    # Compare with diffai
    result = subprocess.run([
        "diffai", model1_path, model2_path, "--format", "json"
    ], capture_output=True, text=True)
    
    comparison = json.loads(result.stdout)
    
    # Log results to MLflow
    with mlflow.start_run():
        mlflow.log_dict(comparison, "model_comparison.json")
        mlflow.log_metric("param_count_diff", comparison["param_diff"])
        
    return comparison
```

## Performance Analysis

### Model Size Tracking

```bash
# Track model size changes
diffai baseline/model.pth optimized/model.pth --show-size-reduction

# Compare multiple model sizes
for model in models/*.pth; do
  size=$(stat -f%z "$model")
  name=$(basename "$model")
  echo "$name: $size bytes"
done | sort -k2 -n
```

### Quantization Impact Assessment

```bash
# Compare before/after quantization
diffai full_precision/model.pth quantized/model.pth --quantization-analysis

# Check accuracy impact
diffai full_precision/results.json quantized/results.json --metric-comparison
```

## Best Practices

### 1. Comparison Automation

```bash
# comparison_script.sh
#!/bin/bash

BASELINE="baseline/model.pth"
EXPERIMENT="experiment/model.pth"
REPORT="comparison_report.json"

# Basic comparison
diffai $BASELINE $EXPERIMENT --format json > $REPORT

# Check for structural changes
if diffai $BASELINE $EXPERIMENT --diff-only --quiet; then
    echo "No structural changes detected"
else
    echo "WARNING: Structural changes found - review required"
    diffai $BASELINE $EXPERIMENT --show-structure
fi

# Check parameter count changes
python scripts/check_param_changes.py $REPORT
```

### 2. Team Collaboration

```bash
# Generate team comparison report
diffai model1.pth model2.pth --output json > team_comparison.json

# Notify team via Slack
curl -X POST -H 'Content-type: application/json' \
  --data '{"text":"Model comparison completed: see attached report"}' \
  $SLACK_WEBHOOK_URL
```

## Integration with ML Tools

### Weights & Biases

```python
import wandb
import subprocess
import json

def log_model_comparison(run1, run2):
    # Compare two models from wandb
    comparison = subprocess.run([
        "diffai", f"wandb_models/{run1}.pth", f"wandb_models/{run2}.pth", 
        "--format", "json"
    ], capture_output=True, text=True)
    
    wandb.log({"model_comparison": json.loads(comparison.stdout)})
```

### TensorBoard

```python
from torch.utils.tensorboard import SummaryWriter
import subprocess
import json

def log_comparison_to_tensorboard(model1, model2, step):
    writer = SummaryWriter()
    
    result = subprocess.run([
        "diffai", model1, model2, "--format", "json"
    ], capture_output=True, text=True)
    
    comparison = json.loads(result.stdout)
    
    writer.add_text("model_comparison", 
                   json.dumps(comparison, indent=2), step)
    writer.close()
```

## Next Steps

- [Configuration](configuration.md) - Advanced configuration
- [API Reference](../api/cli.md) - Complete command reference
- [Examples](../examples/) - Practical usage examples

