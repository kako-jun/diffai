# 支持格式

diffai 支持的文件格式及其规范。

## 概述

diffai 支持为不同用例优化的广泛文件格式，从机器学习模型到科学数据和结构化配置文件。

## 机器学习模型格式

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

## Structured Data Formats

### JSON
- **Extension**: `.json`
- **Format**: JavaScript Object Notation
- **Support**: Nested objects, arrays, basic data types
- **Use Cases**: Configuration files, API responses, experiment results

### YAML
- **Extensions**: `.yaml`, `.yml`
- **Format**: YAML Ain't Markup Language
- **Support**: Hierarchical structure, comments, multiple documents
- **Use Cases**: Configuration files, CI/CD, Kubernetes

### TOML
- **Extension**: `.toml`
- **Format**: Tom's Obvious, Minimal Language
- **Support**: Type-safe, configuration-oriented structure
- **Use Cases**: Rust configuration, project settings

### XML
- **Extension**: `.xml`
- **Format**: eXtensible Markup Language
- **Support**: Attributes, namespaces, hierarchical structure
- **Use Cases**: Configuration files, data exchange

### INI
- **Extension**: `.ini`
- **Format**: Initialization file
- **Support**: Sections, key-value pairs
- **Use Cases**: Configuration files, legacy applications

### CSV
- **Extension**: `.csv`
- **Format**: Comma-Separated Values
- **Support**: Tabular data, header rows
- **Use Cases**: Data analysis, spreadsheets

## Format Auto-Detection

diffai detects formats in the following priority order:

1. **--format** option (explicit specification)
2. **File extension** based auto-detection
3. **Configuration file** default format
4. **Error** (if unable to determine)

## Implementation Status

### Phase 1-2 (Completed)
- ✅ PyTorch (.pt, .pth)
- ✅ Safetensors (.safetensors)
- ✅ NumPy (.npy, .npz)
- ✅ MATLAB (.mat)
- ✅ JSON (.json)
- ✅ YAML (.yaml, .yml)
- ✅ TOML (.toml)
- ✅ XML (.xml)
- ✅ INI (.ini)
- ✅ CSV (.csv)

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

- [CLI参考](cli-reference_zh.md) - Complete command-line options
- [ML Analysis Functions](ml-analysis_zh.md) - Machine learning specialized functions
- [Output Formats](output-formats_zh.md) - Output format specifications

