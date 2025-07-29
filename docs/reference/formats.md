# Supported AI/ML File Formats

AI/ML and scientific computing file formats supported by diffai and their specifications.

## Overview

diffai is specialized for AI/ML and scientific computing file formats only. For general-purpose structured data formats (JSON, YAML, CSV, XML, etc.), please use our sibling project [diffx](https://github.com/kako-jun/diffx).

## Machine Learning Model Formats

### PyTorch Models
- **Extensions**: `.pt`, `.pth`
- **Format**: Pickle format with Candle library integration
- **Support**: Tensor statistics, shape changes, parameter additions/deletions
- **Data Types**: F32, F64, F16, BF16, I64, U32, U8

**Example**:
```bash
diffai model1.pt model2.pt
```

### Safetensors Models
- **Extension**: `.safetensors`
- **Format**: HuggingFace Safetensors format (recommended)
- **Support**: Fast loading, memory efficient, secure
- **Data Types**: F32, F64, F16, BF16, I64, U32, U8

**Example**:
```bash
diffai model1.safetensors model2.safetensors
```

## Scientific Data Formats

### NumPy Arrays
- **Extensions**: `.npy`, `.npz`
- **Format**: NumPy array format
- **Support**: Statistical analysis, shape comparison, data type validation
- **Data Types**: All NumPy data types

**Example**:
```bash
diffai data1.npy data2.npy
diffai archive1.npz archive2.npz
```

### MATLAB Matrices
- **Extension**: `.mat`
- **Format**: MATLAB matrix files
- **Support**: Complex numbers, variable names, multi-dimensional arrays
- **Data Types**: double, single, int8-64, uint8-64, logical

**Example**:
```bash
diffai simulation1.mat simulation2.mat
```

## Supported Extensions

diffai exclusively supports the following file extensions:

- **ML Model Files**: `.pt`, `.pth`, `.safetensors`
- **Scientific Data Files**: `.npy`, `.npz`, `.mat`

For general-purpose formats like `.json`, `.yaml`, `.csv`, `.xml`, `.ini`, `.toml`, please use [diffx](https://github.com/kako-jun/diffx) instead.

## Format Auto-Detection

diffai automatically detects file formats based on extensions - no configuration needed.

## Implementation Status

### Phase 1-2 (Completed)
- ✅ PyTorch (.pt, .pth)
- ✅ Safetensors (.safetensors)
- ✅ NumPy (.npy, .npz)
- ✅ MATLAB (.mat)

### Phase 3 (Planned)
- ⏳ TensorFlow (.pb, .h5, SavedModel)
- ⏳ ONNX (.onnx)
- ⏳ HDF5 (.h5, .hdf5)

## Format Comparison

### ML Model Formats

| Format | Speed | Security | Memory Efficiency | Recommendation |
|--------|-------|----------|------------------|----------------|
| Safetensors | High | High | High | Recommended |
| PyTorch | Medium | Low | Medium | Available |

### Scientific Data Formats

| Format | Precision | Compression | Complex Numbers | Recommended Use |
|--------|-----------|-------------|-----------------|-----------------|
| NumPy | High | Medium | Supported | Python scientific computing |
| MATLAB | High | Low | Supported | MATLAB environment |

## Related Documentation

- [CLI Reference](cli-reference.md) - Complete command-line options
- [ML Analysis Functions](ml-analysis.md) - Machine learning specialized functions
- [Output Formats](output-formats.md) - Output format specifications

