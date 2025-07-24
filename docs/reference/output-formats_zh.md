# Output Formats

Output formats supported by diffai and their specifications.

## Overview

diffai supports four different output formats to suit various use cases:

1. **CLI** - Human-readable format (default)
2. **JSON** - Machine processing and automation
3. **YAML** - Configuration files and human-readable structured data
4. **Unified** - Git integration and traditional diff format

## CLI Output Format

### Overview
Human-readable colored output format used by default.

### Features
- **Colored display**: Color-coded by change type
- **Symbol display**: Intuitive symbols (`+`, `-`, `~`, `□`)
- **Hierarchical display**: Nested structure representation
- **ML-specific symbols**: Specialized display for ML analysis results

### Usage
```bash
diffai model1.safetensors model2.safetensors --output cli
# or
diffai model1.safetensors model2.safetensors  # default
```

### Example Output
```
~ fc1.bias: mean=0.0018->0.0017, std=0.0518->0.0647
~ fc1.weight: mean=-0.0002->-0.0001, std=0.0514->0.0716
+ new_layer.weight: shape=[64, 64], dtype=f32, params=4096
- old_layer.bias: shape=[256], dtype=f32, params=256
[CRITICAL] anomaly_detection: type=gradient_explosion, severity=critical, affected=2 layers
```

### Symbol Meanings

#### Basic Symbols
| Symbol | Meaning | Color | Description |
|--------|---------|-------|-------------|
| `~` | Modified | Blue | Value changed but structure same |
| `+` | Added | Green | New element added |
| `-` | Removed | Red | Element removed |
| `□` | Shape Changed | Yellow | Tensor shape changed |

#### ML Analysis Symbols
| Symbol | Meaning | Color | Description |
|--------|---------|-------|-------------|
| `◦` | Analysis Result | Cyan | ML analysis function result |
| `[CRITICAL]` | Critical Alert | Red | Critical issue detected |
| `[WARNING]` | Warning | Yellow | Attention needed |
| `[ALERT]` | Alert | Red | Threshold exceeded |
| `[HIGH]` | High Priority | Red | High priority issue |
| `[HOLD]` | Hold Deployment | Red | Deployment hold recommended |
| `[GRADUAL]` | Gradual Deployment | Yellow | Gradual deployment recommended |

## JSON Output Format

### Overview
Structured data format suitable for machine processing and automation.

### Features
- **Structured data**: Easy to parse programmatically
- **Type information**: Complete type information preserved
- **API integration**: Easy integration with RESTful APIs
- **Automation**: Optimal for CI/CD and scripting

### Usage
```bash
diffai model1.safetensors model2.safetensors --output json
```

### Example Output
```json
[
  {
    "TensorStatsChanged": [
      "fc1.bias",
      {"mean": 0.0018, "std": 0.0518, "shape": [64], "dtype": "f32", "total_params": 64},
      {"mean": 0.0017, "std": 0.0647, "shape": [64], "dtype": "f32", "total_params": 64}
    ]
  },
  {
    "TensorAdded": [
      "new_layer.weight",
      {"mean": 0.0123, "std": 0.0456, "shape": [64, 64], "dtype": "f32", "total_params": 4096}
    ]
  },
  {
    "AnomalyDetection": [
      "anomaly_detection",
      {
        "anomaly_type": "gradient_explosion",
        "severity": "critical",
        "affected_layers": ["layer1", "layer2"],
        "recommended_action": "Reduce learning rate"
      }
    ]
  }
]
```

### Data Structure Types

#### Basic Change Types
- **Added**: `["key", value]` - Added element
- **Removed**: `["key", value]` - Removed element
- **Modified**: `["key", old_value, new_value]` - Modified element
- **TypeChanged**: `["key", old_type, new_type]` - Type changed

#### ML-Specific Types
- **TensorStatsChanged**: Tensor statistics changed
- **TensorShapeChanged**: Tensor shape changed
- **TensorAdded**: Tensor added
- **TensorRemoved**: Tensor removed
- **LearningProgress**: Learning progress analysis
- **ConvergenceAnalysis**: Convergence analysis
- **AnomalyDetection**: Anomaly detection
- **MemoryAnalysis**: Memory analysis
- **DeploymentReadiness**: Deployment readiness
- **RiskAssessment**: Risk assessment

## YAML Output Format

### Overview
Human-readable structured data format suitable for configuration files and documentation.

### Features
- **Readability**: Human-friendly format
- **Comments**: Comment support
- **Hierarchical structure**: Clear hierarchy representation
- **Configuration files**: Can be used as configuration files

### Usage
```bash
diffai model1.safetensors model2.safetensors --output yaml
```

### Example Output
```yaml
- TensorStatsChanged:
  - fc1.bias
  - mean: 0.0018
    std: 0.0518
    shape: [64]
    dtype: f32
    total_params: 64
  - mean: 0.0017
    std: 0.0647
    shape: [64]
    dtype: f32
    total_params: 64

- AnomalyDetection:
  - anomaly_detection
  - anomaly_type: gradient_explosion
    severity: critical
    affected_layers:
      - layer1
      - layer2
    recommended_action: Reduce learning rate
```

## Unified Output Format

### Overview
Git integration and compatibility with traditional diff tools.

### Features
- **Git integration**: Compatible with git diff
- **Merge tools**: Works with 3-way merge tools
- **Traditional format**: Compatible with existing diff tools
- **Patch application**: Can be applied with git apply

### Usage
```bash
```

### Example Output
```diff
--- config1.json
+++ config2.json
@@ -1,5 +1,6 @@
 {
   "model": {
-    "layers": 12,
+    "layers": 24,
     "hidden_size": 768
   },
+  "optimizer": "adam"
 }
```

## Output Format Selection Guidelines

### By Use Case

| Use Case | Recommended Format | Reason |
|----------|-------------------|---------|
| Human review | CLI | Colored, intuitive symbols |
| Automation/scripting | JSON | Machine processable |
| Configuration files | YAML | Readability, comment support |
| Documentation | YAML | Human-friendly |
| Git integration | Unified | Existing tool compatibility |
| API integration | JSON | Standard data exchange format |
| Report generation | JSON | Structured data processing |

### By Environment

| Environment | Recommended Format | Reason |
|-------------|-------------------|---------|
| Interactive command line | CLI | Immediate understanding |
| CI/CD pipeline | JSON | Automated checks |
| Development environment | CLI | Quick debugging |
| Production environment | JSON | Logging and monitoring |
| Research/experiments | YAML | Result documentation |

## Advanced Usage Examples

### Pipeline Processing
```bash
# Output JSON and process with jq
diffai model1.safetensors model2.safetensors --output json | \
  jq '.[] | select(.TensorStatsChanged)'

# Output YAML and save to file
diffai config1.yaml config2.yaml --output yaml > changes.yaml
```

### Conditional Logic
```bash
# Check if changes exist
if diffai model1.safetensors model2.safetensors --output json | jq -e 'length > 0'; then
  echo "Changes detected"
fi
```

### Multiple Format Generation
```bash
# Generate both human and machine readable formats
diffai model1.safetensors model2.safetensors > human_readable.txt
diffai model1.safetensors model2.safetensors --output json > machine_readable.json
```

## Configuration and Customization

### Configuration File
```toml
[output]
default = "cli"
json_pretty = true
yaml_flow = false
cli_colors = true

[colors]
added = "green"
removed = "red"
modified = "blue"
warning = "yellow"
error = "red"
```

### Environment Variables
```bash
export DIFFAI_OUTPUT_FORMAT="json"
export DIFFAI_CLI_COLORS="true"
export DIFFAI_JSON_PRETTY="true"
export DIFFAI_YAML_FLOW="false"
```

## Related Documentation

- [CLI参考](cli-reference_zh.md) - Complete command-line options
- [Supported Formats](formats_zh.md) - Input file formats
- [ML Analysis Functions](ml-analysis_zh.md) - Machine learning analysis functions

