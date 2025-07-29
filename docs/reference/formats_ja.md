# サポートされるAI/MLファイル形式

diffaiがサポートするAI/MLと科学計算ファイル形式とその仕様。

## 概要

diffaiはAI/MLと科学計算ファイル形式専用に特化されています。汎用構造化データ形式（JSON、YAML、CSV、XML等）については、姉妹プロジェクト[diffx](https://github.com/kako-jun/diffx)をご利用ください。

## 機械学習モデル形式

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

## サポートされる拡張子

diffaiは以下のファイル拡張子のみを排他的にサポートします：

- **MLモデルファイル**: `.pt`, `.pth`, `.safetensors`
- **科学データファイル**: `.npy`, `.npz`, `.mat`

`.json`, `.yaml`, `.csv`, `.xml`, `.ini`, `.toml`等の汎用形式については、代わりに[diffx](https://github.com/kako-jun/diffx)をご利用ください。

## Format Auto-Detection

diffai detects formats in the following priority order:

1. **--format** option (explicit specification)
2. **File extension** based auto-detection
3. **Configuration file** default format
4. **Error** (if unable to determine)

## Implementation Status

### Phase 1-2 (完了)
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

- [CLIリファレンス](cli-reference_ja.md) - Complete command-line options
- [ML Analysis Functions](ml-analysis_ja.md) - Machine learning specialized functions
- [Output Formats](output-formats_ja.md) - Output format specifications

