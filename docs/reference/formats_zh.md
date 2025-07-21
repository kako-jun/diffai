# 支持的格式

diffai支持的文件格式及其规范。

## 概述

diffai支持多种针对不同用例优化的文件格式，从机器学习模型到科学数据和结构化配置文件。

## 机器学习模型格式

### PyTorch模型
- **扩展名**: `.pt`、`.pth`
- **格式**: Pickle格式，集成Candle库
- **支持**: 张量统计、形状变化、参数添加/删除
- **数据类型**: F32、F64、F16、BF16、I64、U32、U8

**示例**:
```bash
diffai model1.pt model2.pt
```

### Safetensors模型
- **扩展名**: `.safetensors`
- **格式**: HuggingFace Safetensors格式（推荐）
- **支持**: 快速加载、内存高效、安全
- **数据类型**: F32、F64、F16、BF16、I64、U32、U8

**示例**:
```bash
diffai model1.safetensors model2.safetensors
```

## 科学数据格式

### NumPy数组
- **扩展名**: `.npy`、`.npz`
- **格式**: NumPy数组格式
- **支持**: 统计分析、形状比较、数据类型验证
- **数据类型**: 所有NumPy数据类型

**示例**:
```bash
diffai data1.npy data2.npy
diffai archive1.npz archive2.npz
```

### MATLAB矩阵
- **扩展名**: `.mat`
- **格式**: MATLAB矩阵文件
- **支持**: 复数、变量名、多维数组
- **数据类型**: double、single、int8-64、uint8-64、logical

**示例**:
```bash
diffai simulation1.mat simulation2.mat
```

## 结构化数据格式

### JSON
- **扩展名**: `.json`
- **格式**: JavaScript对象表示法
- **支持**: 嵌套对象、数组、基本数据类型
- **用例**: 配置文件、API响应、实验结果

### YAML
- **扩展名**: `.yaml`、`.yml`
- **格式**: YAML不是标记语言
- **支持**: 层次结构、注释、多文档
- **用例**: 配置文件、CI/CD、Kubernetes

### TOML
- **扩展名**: `.toml`
- **格式**: Tom的显而易见的最小语言
- **支持**: 类型安全、面向配置的结构
- **用例**: Rust配置、项目设置

### XML
- **扩展名**: `.xml`
- **格式**: 可扩展标记语言
- **支持**: 属性、命名空间、层次结构
- **用例**: 配置文件、数据交换

### INI
- **扩展名**: `.ini`
- **格式**: 初始化文件
- **支持**: 节、键值对
- **用例**: 配置文件、传统应用程序

### CSV
- **扩展名**: `.csv`
- **格式**: 逗号分隔值
- **支持**: 表格数据、标题行
- **用例**: 数据分析、电子表格

## 格式自动检测

diffai按以下优先级顺序检测格式：

1. **--format** 选项（显式指定）
2. **文件扩展名** 基于的自动检测
3. **配置文件** 默认格式
4. **错误**（如果无法确定）

## 实现状态

### 第1-2阶段（已完成）
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

### 第3阶段（计划中）
- ⏳ TensorFlow (.pb, .h5, SavedModel)
- ⏳ ONNX (.onnx)
- ⏳ HDF5 (.h5, .hdf5)

## 格式比较

### ML模型格式

| 格式 | 速度 | 安全性 | 内存效率 | 推荐 |
|------|------|--------|----------|------|
| Safetensors | 高 | 高 | 高 | 推荐 |
| PyTorch | 中等 | 低 | 中等 | 可用 |

### 科学数据格式

| 格式 | 精度 | 压缩 | 复数支持 | 推荐用途 |
|------|------|------|----------|----------|
| NumPy | 高 | 中等 | 支持 | Python科学计算 |
| MATLAB | 高 | 低 | 支持 | MATLAB环境 |

## 相关文档

- [CLI参考](cli-reference_zh.md) - 完整的命令行选项
- [ML分析功能](ml-analysis_zh.md) - 机器学习专用功能
- [输出格式](output-formats_zh.md) - 输出格式规范

