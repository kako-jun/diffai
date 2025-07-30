# File Formats & Output Options

diffai specializes in AI/ML and scientific computing file formats, with automatic analysis for PyTorch and Safetensors files.

## Supported Input Formats

### AI/ML Model Formats (Auto ML Analysis)

#### PyTorch Models (.pt, .pth)
- **Automatic ML Analysis**: All 11 functions run automatically
- **Support**: Model state dictionaries, checkpoints, complete models
- **Integration**: Uses Candle for PyTorch format parsing
- **Memory**: Efficient handling of large models (GB+ files)

```bash
# Automatic comprehensive analysis
diffai baseline_model.pt finetuned_model.pt
# Outputs: All 11 ML analysis functions + tensor statistics
```

#### Safetensors (.safetensors)
- **Automatic ML Analysis**: All 11 functions run automatically  
- **Support**: HuggingFace standard format
- **Performance**: Fast loading and memory-efficient parsing
- **Safety**: Secure tensor storage format

```bash
# Automatic comprehensive analysis
diffai model_v1.safetensors model_v2.safetensors
# Outputs: All 11 ML analysis functions + tensor statistics
```

### Scientific Data Formats (Basic Analysis)

#### NumPy Arrays (.npy, .npz)
- **Analysis**: Tensor statistics only (no ML-specific analysis)
- **Support**: Single arrays (.npy) and archives (.npz)
- **Data Types**: All NumPy dtypes including complex numbers
- **Statistics**: Shape, mean, std, min, max, memory usage

```bash
# Basic tensor analysis
diffai experiment_data_v1.npy experiment_data_v2.npy
# Outputs: Shape, statistics, dtype changes
```

#### MATLAB Files (.mat)
- **Analysis**: Tensor statistics only (no ML-specific analysis)
- **Support**: MATLAB matrix files with variable detection
- **Data Types**: Double, single, complex, logical arrays
- **Variables**: Multi-variable file support

```bash
# Basic tensor analysis
diffai simulation_v1.mat simulation_v2.mat
# Outputs: Variable changes, matrix statistics
```

### Format Detection
diffai automatically detects file formats based on:
1. **File extension** (.pt, .safetensors, .npy, .mat)
2. **Magic bytes** and file headers
3. **Content structure** analysis

## Output Formats

### CLI Output (Default)
Human-readable format with color coding and intuitive symbols.

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

#### Example CLI Output
```bash
diffai model1.safetensors model2.safetensors

learning_rate_analysis: old=0.001, new=0.0015, change=+50.0%, trend=increasing
gradient_analysis: flow_health=healthy, norm=0.021069, variance_change=+15.3%
~ fc1.bias: mean=0.0018->0.0017, std=0.0518->0.0647
~ fc1.weight: mean=-0.0002->-0.0001, std=0.0514->0.0716
+ new_layer.weight: shape=[64, 64], dtype=f32, params=4096
- old_layer.bias: shape=[256], dtype=f32, params=256
```

### JSON Output
Structured format for automation and MLOps integration.

```bash
diffai model1.safetensors model2.safetensors --output json
```

**Example JSON Structure**:
```json
{
  "learning_rate_analysis": {
    "old": 0.001,
    "new": 0.0015,
    "change": "+50.0%",
    "trend": "increasing"
  },
  "tensor_changes": [
    {
      "path": "fc1.weight",
      "change_type": "modified",
      "old_stats": {"mean": -0.0002, "std": 0.0514},
      "new_stats": {"mean": -0.0001, "std": 0.0716}
    }
  ]
}
```

### YAML Output
Human-readable structured format for reports and documentation.

```bash
diffai model1.safetensors model2.safetensors --output yaml
```

**Example YAML Structure**:
```yaml
learning_rate_analysis:
  old: 0.001
  new: 0.0015
  change: "+50.0%"
  trend: "increasing"

gradient_analysis:
  flow_health: "healthy"
  gradient_norm: 0.021069
  variance_change: "+15.3%"
```

### Verbose Mode
Comprehensive diagnostic information for debugging and analysis.

```bash
diffai model1.safetensors model2.safetensors --verbose
```

**Verbose Output Includes**:
- Configuration diagnostics
- File analysis details
- Performance metrics
- Processing context
- Memory usage information

## Format-Specific Features

### PyTorch Integration
- **State Dictionary**: Automatic parsing of model state
- **Optimizer State**: Analysis of optimizer parameters
- **Checkpoints**: Full checkpoint comparison support
- **Custom Objects**: Handling of custom PyTorch objects

### Safetensors Advantages
- **Security**: No arbitrary code execution risk
- **Performance**: Fast memory-mapped loading
- **Cross-Platform**: Consistent across languages
- **Metadata**: Rich tensor metadata support

### NumPy Capabilities
- **Arrays**: Single array files (.npy)
- **Archives**: Multi-array files (.npz)
- **Complex Numbers**: Full complex data support
- **Memory Views**: Efficient large array handling

### MATLAB Support
- **Variables**: Multi-variable MATLAB files
- **Data Types**: All MATLAB numeric types
- **Cell Arrays**: Basic cell array support
- **Sparse Matrices**: Sparse matrix detection

## Performance Considerations

### Memory Usage
- **Streaming**: Large files processed in chunks
- **Memory Mapping**: Efficient memory usage for Safetensors
- **Garbage Collection**: Automatic memory cleanup
- **Batch Processing**: Configurable batch sizes

### File Size Limits
- **No Hard Limits**: Handles files of any size
- **Automatic Optimization**: Memory optimization for large files
- **Progress Reporting**: Progress indication for large files
- **Chunked Processing**: Breaks large comparisons into chunks

### Speed Optimization
- **Early Termination**: Stops when patterns not detected
- **Parallel Processing**: Multi-threaded where beneficial
- **Caching**: Intermediate result caching
- **Efficient Algorithms**: lawkit incremental statistics

## Error Handling

### Unsupported Formats
diffai focuses on AI/ML formats. For general-purpose structured data:
```bash
# For JSON, XML, CSV, etc. - use diffx instead
diffx data1.json data2.json
```

### Corrupted Files
- **Graceful Degradation**: Partial analysis when possible
- **Error Recovery**: Continues with available data
- **Clear Messages**: Descriptive error messages
- **Validation**: File format validation before processing

### Large File Handling
- **Memory Monitoring**: Automatic memory usage tracking
- **Optimization**: Automatic performance optimization
- **Progress**: Progress reporting for long operations
- **Interruption**: Graceful handling of user interruption

## Integration Examples

### Python Integration
```python
import diffai
import json

# Compare PyTorch models
result = diffai.diff_files("model1.pt", "model2.pt", output_format="json")
analysis = json.loads(result)

# Access ML analysis results
lr_analysis = analysis["learning_rate_analysis"]
gradient_info = analysis["gradient_analysis"]
```

### JavaScript Integration
```javascript
const diffai = require('diffai-js');

// Compare Safetensors models
const result = diffai.diffFiles("model1.safetensors", "model2.safetensors", {
    outputFormat: "json"
});

const analysis = JSON.parse(result);
console.log(analysis.quantization_analysis);
```

### Command Line Automation
```bash
#!/bin/bash
# CI/CD model validation script

RESULT=$(diffai baseline.safetensors candidate.safetensors --output json)
ACCURACY_CHANGE=$(echo "$RESULT" | jq -r '.accuracy_tracking.accuracy_delta')

if [[ "$ACCURACY_CHANGE" == +* ]]; then
    echo "Model improved: $ACCURACY_CHANGE"
    exit 0
else
    echo "Model degraded: $ACCURACY_CHANGE"
    exit 1
fi
```

## See Also

- **[Quick Start](quick-start.md)** - Get started quickly
- **[ML Analysis](ml-analysis.md)** - Understand the 11 analysis functions
- **[API Reference](reference/api-reference.md)** - Programming interface
- **[Examples](examples/)** - Real usage examples