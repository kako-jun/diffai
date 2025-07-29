# 支持的AI/ML文件格式

diffai 支持的AI/ML和科学计算文件格式及其规范。

## 概述

diffai专门为AI/ML和科学计算文件格式而特化。对于通用结构化数据格式（JSON、YAML、CSV、XML等），请使用专为这些格式设计的姊妹项目[diffx](https://github.com/kako-jun/diffx)。

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

## 支持的扩展名

diffai专门支持以下文件扩展名：

- **ML模型文件**: `.pt`, `.pth`, `.safetensors`
- **科学数据文件**: `.npy`, `.npz`, `.mat`

对于`.json`, `.yaml`, `.csv`, `.xml`, `.ini`, `.toml`等通用格式，请使用[diffx](https://github.com/kako-jun/diffx)。

## Format Auto-Detection

diffai detects formats in the following priority order:

1. **--format** option (explicit specification)
2. **File extension** based auto-detection
3. **Configuration file** default format
4. **Error** (if unable to determine)

## Implementation Status

### Phase 1-2 (已完成)
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

- [CLI参考](cli-reference_zh.md) - Complete command-line options
- [ML Analysis Functions](ml-analysis_zh.md) - Machine learning specialized functions
- [Output Formats](output-formats_zh.md) - Output format specifications

