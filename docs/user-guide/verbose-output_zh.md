# 详细输出指南

diffai中的`--verbose`标志提供综合诊断信息，帮助用户理解处理详情、调试问题和分析性能。本指南解释如何有效使用详细模式。

## 概述

详细模式显示以下详细信息：
- 配置设置和启用的功能
- 文件分析和格式检测
- 处理时间和性能指标
- ML特定的分析状态
- 目录比较统计

## 基本用法

### 启用详细模式

```bash
# 基本详细输出
diffai file1.json file2.json --verbose

# 简写形式
diffai file1.json file2.json -v
```

### 输出示例

```
=== diffai verbose mode enabled ===
Configuration:
  Input format: None
  Output format: Cli
  Recursive mode: false

File analysis:
  Input 1: file1.json
  Input 2: file2.json
  Detected format: Json
  File 1 size: 156 bytes
  File 2 size: 162 bytes

Processing results:
  Total processing time: 234.567µs
  Differences found: 3
  Format-specific analysis: Json
```

## 配置诊断

详细模式显示所有活动的配置选项：

### 基本配置
- **输入格式**: 明确设置或自动检测的格式
- **输出格式**: CLI、JSON、YAML或统一diff
- **递归模式**: 是否启用目录比较

### 高级选项
```bash
diffai file1.json file2.json --verbose \
  --epsilon 0.001 \
  --ignore-keys-regex "^id$" \
  --path "config.users"
```

输出包括：
```
Configuration:
  Input format: None
  Output format: Cli
  Recursive mode: false
  Epsilon tolerance: 0.001
  Ignore keys regex: ^id$
  Path filter: config.users
```

### ML分析功能
当ML分析选项启用时，详细模式显示哪些功能处于活动状态：

```bash
diffai model1.safetensors model2.safetensors --verbose \
  \
  --architecture-comparison \
  --memory-analysis \
  --anomaly-detection
```

输出：
```
Configuration:
  ML analysis features: statistics, architecture_comparison, memory_analysis, anomaly_detection
```

## 文件分析信息

### 文件元数据
详细模式提供详细的文件信息：

- **文件路径**: 输入文件的完整路径
- **文件大小**: 每个文件的确切字节数
- **格式检测**: diffai如何识别文件格式

示例：
```
File analysis:
  Input 1: /path/to/model1.safetensors
  Input 2: /path/to/model2.safetensors
  Detected format: Safetensors
  File 1 size: 1048576 bytes
  File 2 size: 1048576 bytes
```

### 格式检测过程
详细模式解释如何确定文件格式：

1. **明确格式**: 当指定`--format`时
2. **自动检测**: 基于文件扩展名
3. **回退逻辑**: 当格式无法推断时

## 性能指标

### 处理时间
详细模式以微秒精度测量和报告处理时间：

```
Processing results:
  Total processing time: 1.234567ms
  Differences found: 15
```

### ML/科学数据分析
对于ML和科学数据文件，详细模式指示完成状态：

```
Processing results:
  Format-specific analysis: Safetensors
  ML/Scientific data analysis completed
```

## 目录比较

在使用`--recursive`进行目录比较时，详细模式提供额外的统计信息：

```bash
diffai dir1/ dir2/ --verbose --recursive
```

示例输出：
```
Configuration:
  Recursive mode: true

Directory scan results:
  Files in /path/to/dir1: 12
  Files in /path/to/dir2: 14
  Total files to compare: 16

Directory comparison summary:
  Files compared: 10
  Files only in one directory: 6
  Total files processed: 16
```

## 使用场景

### 调试处理问题

当diffai表现异常时，详细模式帮助识别：

1. **格式检测问题**: 检查是否检测到正确的格式
2. **配置冲突**: 验证所有选项是否正确应用
3. **性能瓶颈**: 识别缓慢的处理步骤
4. **文件访问问题**: 确认文件可读且大小符合预期

调试会话示例：
```bash
# 检查格式检测是否正确
diffai problematic_file1.dat problematic_file2.dat --verbose

# 验证ML分析功能（ML模型自动执行）
diffai model1.pt model2.pt --verbose

# 分析目录比较行为
diffai dir1/ dir2/ --verbose --recursive
```

### 性能分析

使用详细模式了解处理性能：

```bash
# 测量不同格式的处理时间
diffai large_model1.safetensors large_model2.safetensors --verbose

# 比较不同选项的性能
diffai data1.json data2.json --verbose --epsilon 0.0001
```

### 配置验证

验证复杂配置是否正确应用：

```bash
# 检查多个过滤器和选项
diffai config1.yaml config2.yaml --verbose \
  --ignore-keys-regex "^(id|timestamp)$" \
  --path "application.settings" \
  --epsilon 0.01 \
  --output json
```

## 技巧和最佳实践

### 1. 调试时始终使用详细模式
遇到意外行为时，始终添加`--verbose`来理解diffai在做什么。

### 2. 性能监控
使用详细模式监控大文件的处理时间并相应优化。

### 3. 配置验证
在运行批处理操作之前，在单个文件上使用详细模式验证配置。

### 4. 学习文件格式
使用详细模式了解diffai如何检测和处理不同的文件格式。

### 5. ML分析优化
使用多个ML分析功能时，详细模式帮助识别哪些功能处于活动状态及其对处理时间的影响。

## 输出重定向

### 分离详细输出和结果
详细信息发送到stderr，允许您将其与结果分离：

```bash
# 将结果保存到文件，在屏幕上显示详细输出
diffai file1.json file2.json --verbose --output json > results.json

# 保存结果和详细输出
diffai file1.json file2.json --verbose > results.txt 2> verbose.log

# 仅显示详细信息
diffai file1.json file2.json --verbose 2>&1 >/dev/null
```

## 与其他工具集成

### CI/CD流水线
在CI/CD中使用详细模式进行调试：

```bash
# 在GitHub Actions或类似工具中
- name: 使用详细输出比较模型
  run: diffai baseline.safetensors new_model.safetensors --verbose
```

### 脚本和自动化
解析详细输出进行自动化：

```bash
#!/bin/bash
output=$(diffai file1.json file2.json --verbose 2>&1)
if echo "$output" | grep -q "Differences found: 0"; then
    echo "文件相同"
else
    echo "文件不同"
fi
```

## 相关命令

- [`--help`](basic-usage.md#help): 显示所有可用选项
- [`--output`](output-formats.md): 控制输出格式
- [`--recursive`](directory-comparison.md): 启用目录比较