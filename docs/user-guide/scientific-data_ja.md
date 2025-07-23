# Scientific Data Analysis Guide

This guide covers diffai's capabilities for analyzing scientific data formats, including NumPy arrays and MATLAB matrices.

## Overview

diffai extends beyond ML models to support scientific data formats commonly used in research and computational science. This enables meaningful comparison of numerical arrays, experimental results, and simulation data.

## Supported Scientific Data Formats

### NumPy Arrays
- **`.npy` files**: Single NumPy arrays with complete statistical analysis
- **`.npz` files**: Compressed NumPy archives containing multiple arrays

### MATLAB Matrices
- **`.mat` files**: MATLAB matrix files with complex number support

## What diffai Analyzes

### Automatic Comprehensive Analysis
diffai automatically provides complete statistical analysis for all scientific data formats. No flags are required - all analysis features are included by default.

### Array Statistics
For each array in the data, diffai automatically calculates and compares:

- **Mean**: Average value of all elements
- **Standard Deviation**: Measure of data variance
- **Minimum**: Smallest value in the array
- **Maximum**: Largest value in the array
- **Shape**: Array dimensions
- **Data Type**: Element precision (float64, int32, etc.)
- **Element Count**: Total number of elements

### MATLAB-Specific Features
- **Complex Number Support**: Real and imaginary parts analyzed separately
- **Variable Names**: MATLAB variable names preserved
- **Multi-dimensional Arrays**: Full support for N-dimensional matrices
- **Mixed Data Types**: Different data types within single .mat file

## Basic Data Comparison

### NumPy Array Comparison

```bash
# Compare single NumPy arrays
diffai data_v1.npy data_v2.npy

# Compare compressed NumPy archives
diffai dataset_v1.npz dataset_v2.npz

# With specific output format
diffai experiment_baseline.npy experiment_result.npy --output json
```

### MATLAB File Comparison

```bash
# Compare MATLAB files
diffai simulation_v1.mat simulation_v2.mat

# Focus on specific variables
diffai results_v1.mat results_v2.mat --path "experiment_data"

# Export to YAML for documentation
diffai analysis_v1.mat analysis_v2.mat --output yaml
```

## Example Outputs

### NumPy Array Changes

```bash
$ diffai experiment_data_v1.npy experiment_data_v2.npy
  ~ data: shape=[1000, 256], mean=0.1234->0.1456, std=0.9876->0.9654, dtype=float64
```

### MATLAB File Changes

```bash
$ diffai simulation_v1.mat simulation_v2.mat
  ~ results: var=results, shape=[500, 100], mean=2.3456->2.4567, std=1.2345->1.3456, dtype=double
  + new_variable: var=new_variable, shape=[100], dtype=single, elements=100, size=0.39KB
```

### Compressed Archive Comparison

```bash
$ diffai dataset_v1.npz dataset_v2.npz
  ~ train_data: shape=[60000, 784], mean=0.1307->0.1309, std=0.3081->0.3082, dtype=float32
  ~ test_data: shape=[10000, 784], mean=0.1325->0.1327, std=0.3105->0.3106, dtype=float32
  + validation_data: shape=[5000, 784], mean=0.1315, std=0.3095, dtype=float32
```

## Advanced Options

### Epsilon Tolerance for Numerical Data

```bash
# Ignore small numerical differences
diffai experiment_v1.npy experiment_v2.npy --epsilon 1e-6

# Useful for comparing simulation results
diffai simulation_v1.mat simulation_v2.mat --epsilon 1e-8
```

### Filtering Results

```bash
# Focus on specific variables in MATLAB files
diffai results_v1.mat results_v2.mat --path "experimental_data"

# Ignore metadata variables
diffai data_v1.mat data_v2.mat --ignore-keys-regex "^(metadata|timestamp)"
```

## Common Use Cases

### 1. Experimental Data Validation

Compare experimental results across different conditions:

```bash
diffai baseline_experiment.npy treated_experiment.npy

# Expected output: Statistical significance of changes
# ~ data: shape=[1000, 50], mean=0.4567->0.5123, std=0.1234->0.1456, dtype=float64
```

**Analysis**:
- Mean shift indicates treatment effect
- Standard deviation changes show variance impact
- Shape consistency confirms data integrity

### 2. Simulation Result Comparison

Compare simulation outputs across parameter sets:

```bash
diffai simulation_param_1.mat simulation_param_2.mat

# Expected output: Parameter sensitivity analysis
# ~ velocity_field: var=velocity_field, shape=[100, 100, 50], mean=1.234->1.567
# ~ pressure_field: var=pressure_field, shape=[100, 100, 50], mean=101.3->102.1
```

**Analysis**:
- Velocity field changes indicate flow differences
- Pressure variations show system response
- Consistent shapes confirm mesh stability

### 3. Data Processing Pipeline Validation

Compare data at different processing stages:

```bash
diffai raw_data.npz processed_data.npz

# Expected output: Processing impact assessment
# ~ features: shape=[10000, 512], mean=0.0->0.5, std=1.0->0.25, dtype=float32
# ~ labels: shape=[10000], mean=4.5->4.5, std=2.87->2.87, dtype=int64
```

**Analysis**:
- Feature normalization successful (mean ~0.5, std ~0.25)
- Labels unchanged (processing preserved classification)
- Consistent shapes ensure no data loss

### 4. Time Series Analysis

Compare time series data across different time periods:

```bash
diffai timeseries_q1.npy timeseries_q2.npy

# Expected output: Temporal pattern changes
# ~ data: shape=[2160, 24], mean=23.45->25.67, std=5.67->6.23, dtype=float32
```

**Analysis**:
- Mean increase indicates seasonal trend
- Standard deviation increase shows higher variability
- Shape consistency confirms data structure

## Performance Optimization

### Large Array Handling

For very large arrays (>1GB):

```bash
# Use higher epsilon for faster comparison
diffai large_array_v1.npy large_array_v2.npy --epsilon 1e-3

# Focus on specific sections
diffai large_sim_v1.mat large_sim_v2.mat --path "summary_stats"
```

### Memory Considerations

```bash
# Set memory limits for large files
DIFFAI_MAX_MEMORY=2048 diffai huge_dataset_v1.npz huge_dataset_v2.npz
```

## Integration Examples

### Python Data Science Workflow

```python
import numpy as np
import subprocess
import json

def compare_arrays(array1_path, array2_path, epsilon=1e-6):
    """Compare two NumPy arrays using diffai"""
    result = subprocess.run([
        'diffai', array1_path, array2_path, 
        '--output', 'json', '--epsilon', str(epsilon)
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        return json.loads(result.stdout)
    else:
        raise RuntimeError(f"Comparison failed: {result.stderr}")

# Example usage
changes = compare_arrays('experiment_v1.npy', 'experiment_v2.npy')
for change in changes:
    if 'NumpyArrayChanged' in change:
        array_name, old_stats, new_stats = change['NumpyArrayChanged']
        print(f"Array {array_name}: mean changed from {old_stats['mean']:.4f} to {new_stats['mean']:.4f}")
```

### MATLAB Integration

```matlab
function compare_mat_files(file1, file2)
    % Compare MATLAB files using diffai
    command = sprintf('diffai %s %s --output json', file1, file2);
    [status, result] = system(command);
    
    if status == 0
        changes = jsondecode(result);
        for i = 1:length(changes)
            if isfield(changes(i), 'MatlabArrayChanged')
                change = changes(i).MatlabArrayChanged;
                fprintf('Variable %s: mean %.4f -> %.4f\n', ...
                    change{1}, change{2}.mean, change{3}.mean);
            end
        end
    else
        error('Comparison failed: %s', result);
    end
end
```

### R Statistical Analysis

```r
library(jsonlite)

compare_data <- function(file1, file2, epsilon = 1e-6) {
  # Compare data files using diffai
  command <- sprintf("diffai %s %s --output json --epsilon %.2e", 
                     file1, file2, epsilon)
  result <- system(command, intern = TRUE)
  
  if (length(result) > 0) {
    changes <- fromJSON(paste(result, collapse = ""))
    return(changes)
  } else {
    return(NULL)
  }
}

# Example usage
changes <- compare_data("analysis_v1.mat", "analysis_v2.mat")
if (!is.null(changes)) {
  for (change in changes) {
    if ("MatlabArrayChanged" %in% names(change)) {
      cat(sprintf("Variable %s changed\n", change$MatlabArrayChanged[[1]]))
    }
  }
}
```

## Best Practices

### 1. Choosing Epsilon Values

| Data Type | Recommended Epsilon | Reason |
|-----------|-------------------|---------|
| Experimental measurements | 1e-6 to 1e-8 | Account for measurement precision |
| Simulation results | 1e-8 to 1e-10 | Numerical computation accuracy |
| Image data | 1e-3 to 1e-6 | Pixel value precision |
| Time series | 1e-4 to 1e-6 | Temporal resolution |

### 2. Output Format Selection

- **CLI**: Human review and quick comparison
- **JSON**: Automated analysis and scripting
- **YAML**: Documentation and reporting

### 3. Performance Tips

- Use appropriate epsilon values to avoid noise
- Consider data size when choosing comparison strategy
- Use path filtering for large multi-variable files
- Monitor memory usage for very large datasets

## Data Format Specifications

### NumPy Array Support

diffai supports all NumPy data types:
- **Integer types**: int8, int16, int32, int64, uint8, uint16, uint32, uint64
- **Float types**: float16, float32, float64
- **Complex types**: complex64, complex128
- **Boolean type**: bool

### MATLAB Matrix Support

diffai supports MATLAB data types:
- **Numeric types**: double, single, int8, int16, int32, int64, uint8, uint16, uint32, uint64
- **Complex types**: Complex double and single precision
- **Logical type**: logical (boolean)
- **Character arrays**: Basic support for metadata

## Troubleshooting

### Common Issues

#### 1. File Format Errors

```bash
# Verify file format
file data.npy

# Check file integrity
python -c "import numpy as np; print(np.load('data.npy').shape)"

# For MATLAB files
python -c "import scipy.io; print(scipy.io.loadmat('data.mat').keys())"
```

#### 2. Memory Issues

```bash
# Check file size
ls -lh large_data.npy

# Use streaming mode (when available)
diffai --stream large_data_v1.npy large_data_v2.npy
```

#### 3. Precision Issues

```bash
# Check data precision
python -c "import numpy as np; print(np.load('data.npy').dtype)"

# Adjust epsilon accordingly
diffai data_v1.npy data_v2.npy --epsilon 1e-8
```

## Next Steps

- [Basic Usage](basic-usage_ja.md) - Learn fundamental operations
- [ML Model Comparison](ml-model-comparison_ja.md) - PyTorch and Safetensors analysis
- [CLIリファレンス](../reference/cli-reference_ja.md) - Complete command reference