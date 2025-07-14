# 科学数据分析指南

本指南介绍了diffai在分析科学数据格式方面的能力，包括NumPy数组和MATLAB矩阵。

## 概述

diffai超越了ML模型，支持研究和计算科学中常用的科学数据格式。这使得能够对数值数组、实验结果和仿真数据进行有意义的比较。

## 支持的科学数据格式

### NumPy数组
- **`.npy`文件**：单个NumPy数组，具有完整的统计分析
- **`.npz`文件**：包含多个数组的压缩NumPy存档

### MATLAB矩阵
- **`.mat`文件**：支持复数的MATLAB矩阵文件

## diffai分析内容

### 数组统计
对于数据中的每个数组，diffai计算并比较：

- **均值**：所有元素的平均值
- **标准差**：数据方差的度量
- **最小值**：数组中的最小值
- **最大值**：数组中的最大值
- **形状**：数组维度
- **数据类型**：元素精度（float64、int32等）
- **元素计数**：元素总数

### MATLAB特定功能
- **复数支持**：实部和虚部分别分析
- **变量名**：保留MATLAB变量名
- **多维数组**：完全支持N维矩阵
- **混合数据类型**：单个.mat文件中的不同数据类型

## 基本数据比较

### NumPy数组比较

```bash
# 比较单个NumPy数组
diffai data_v1.npy data_v2.npy

# 比较压缩的NumPy存档
diffai dataset_v1.npz dataset_v2.npz

# 使用特定输出格式
diffai experiment_baseline.npy experiment_result.npy --output json
```

### MATLAB文件比较

```bash
# 比较MATLAB文件
diffai simulation_v1.mat simulation_v2.mat

# 专注于特定变量
diffai results_v1.mat results_v2.mat --path "experiment_data"

# 导出到YAML用于文档
diffai analysis_v1.mat analysis_v2.mat --output yaml
```

## 示例输出

### NumPy数组变化

```bash
$ diffai experiment_data_v1.npy experiment_data_v2.npy
  ~ data: shape=[1000, 256], mean=0.1234->0.1456, std=0.9876->0.9654, dtype=float64
```

### MATLAB文件变化

```bash
$ diffai simulation_v1.mat simulation_v2.mat
  ~ results: var=results, shape=[500, 100], mean=2.3456->2.4567, std=1.2345->1.3456, dtype=double
  + new_variable: var=new_variable, shape=[100], dtype=single, elements=100, size=0.39KB
```

### 压缩存档比较

```bash
$ diffai dataset_v1.npz dataset_v2.npz
  ~ train_data: shape=[60000, 784], mean=0.1307->0.1309, std=0.3081->0.3082, dtype=float32
  ~ test_data: shape=[10000, 784], mean=0.1325->0.1327, std=0.3105->0.3106, dtype=float32
  + validation_data: shape=[5000, 784], mean=0.1315, std=0.3095, dtype=float32
```

## 高级选项

### 数值数据的Epsilon容差

```bash
# 忽略小的数值差异
diffai experiment_v1.npy experiment_v2.npy --epsilon 1e-6

# 用于比较仿真结果
diffai simulation_v1.mat simulation_v2.mat --epsilon 1e-8
```

### 过滤结果

```bash
# 专注于MATLAB文件中的特定变量
diffai results_v1.mat results_v2.mat --path "experimental_data"

# 忽略元数据变量
diffai data_v1.mat data_v2.mat --ignore-keys-regex "^(metadata|timestamp)"
```

## 常见用例

### 1. 实验数据验证

比较不同条件下的实验结果：

```bash
diffai baseline_experiment.npy treated_experiment.npy

# 预期输出：变化的统计显著性
# ~ data: shape=[1000, 50], mean=0.4567->0.5123, std=0.1234->0.1456, dtype=float64
```

**分析**：
- 均值偏移表明治疗效果
- 标准差变化显示方差影响
- 形状一致性确认数据完整性

### 2. 仿真结果比较

比较不同参数集的仿真输出：

```bash
diffai simulation_param_1.mat simulation_param_2.mat

# 预期输出：参数敏感性分析
# ~ velocity_field: var=velocity_field, shape=[100, 100, 50], mean=1.234->1.567
# ~ pressure_field: var=pressure_field, shape=[100, 100, 50], mean=101.3->102.1
```

**分析**：
- 速度场变化表明流动差异
- 压力变化显示系统响应
- 一致的形状确认网格稳定性

### 3. 数据处理流水线验证

比较不同处理阶段的数据：

```bash
diffai raw_data.npz processed_data.npz

# 预期输出：处理影响评估
# ~ features: shape=[10000, 512], mean=0.0->0.5, std=1.0->0.25, dtype=float32
# ~ labels: shape=[10000], mean=4.5->4.5, std=2.87->2.87, dtype=int64
```

**分析**：
- 特征归一化成功（均值~0.5，标准差~0.25）
- 标签未变（处理保留了分类）
- 一致的形状确保无数据丢失

### 4. 时间序列分析

比较不同时间段的时间序列数据：

```bash
diffai timeseries_q1.npy timeseries_q2.npy

# 预期输出：时间模式变化
# ~ data: shape=[2160, 24], mean=23.45->25.67, std=5.67->6.23, dtype=float32
```

**分析**：
- 均值增加表明季节趋势
- 标准差增加显示更高的变异性
- 形状一致性确认数据结构

## 性能优化

### 大数组处理

对于非常大的数组（>1GB）：

```bash
# 使用更高的epsilon进行更快比较
diffai large_array_v1.npy large_array_v2.npy --epsilon 1e-3

# 专注于特定部分
diffai large_sim_v1.mat large_sim_v2.mat --path "summary_stats"
```

### 内存考虑

```bash
# 为大文件设置内存限制
DIFFAI_MAX_MEMORY=2048 diffai huge_dataset_v1.npz huge_dataset_v2.npz
```

## 集成示例

### Python数据科学工作流

```python
import numpy as np
import subprocess
import json

def compare_arrays(array1_path, array2_path, epsilon=1e-6):
    """使用diffai比较两个NumPy数组"""
    result = subprocess.run([
        'diffai', array1_path, array2_path, 
        '--output', 'json', '--epsilon', str(epsilon)
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        return json.loads(result.stdout)
    else:
        raise RuntimeError(f"比较失败：{result.stderr}")

# 使用示例
changes = compare_arrays('experiment_v1.npy', 'experiment_v2.npy')
for change in changes:
    if 'NumpyArrayChanged' in change:
        array_name, old_stats, new_stats = change['NumpyArrayChanged']
        print(f"数组 {array_name}：均值从 {old_stats['mean']:.4f} 变为 {new_stats['mean']:.4f}")
```

### MATLAB集成

```matlab
function compare_mat_files(file1, file2)
    % 使用diffai比较MATLAB文件
    command = sprintf('diffai %s %s --output json', file1, file2);
    [status, result] = system(command);
    
    if status == 0
        changes = jsondecode(result);
        for i = 1:length(changes)
            if isfield(changes(i), 'MatlabArrayChanged')
                change = changes(i).MatlabArrayChanged;
                fprintf('变量 %s: 均值 %.4f -> %.4f\n', ...
                    change{1}, change{2}.mean, change{3}.mean);
            end
        end
    else
        error('比较失败: %s', result);
    end
end
```

### R统计分析

```r
library(jsonlite)

compare_data <- function(file1, file2, epsilon = 1e-6) {
  # 使用diffai比较数据文件
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

# 使用示例
changes <- compare_data("analysis_v1.mat", "analysis_v2.mat")
if (!is.null(changes)) {
  for (change in changes) {
    if ("MatlabArrayChanged" %in% names(change)) {
      cat(sprintf("变量 %s 已更改\n", change$MatlabArrayChanged[[1]]))
    }
  }
}
```

## 最佳实践

### 1. 选择Epsilon值

| 数据类型 | 推荐Epsilon | 原因 |
|----------|------------|------|
| 实验测量 | 1e-6到1e-8 | 考虑测量精度 |
| 仿真结果 | 1e-8到1e-10 | 数值计算精度 |
| 图像数据 | 1e-3到1e-6 | 像素值精度 |
| 时间序列 | 1e-4到1e-6 | 时间分辨率 |

### 2. 输出格式选择

- **CLI**：人工审查和快速比较
- **JSON**：自动化分析和脚本编写
- **YAML**：文档和报告

### 3. 性能提示

- 使用适当的epsilon值避免噪声
- 选择比较策略时考虑数据大小
- 对大型多变量文件使用路径过滤
- 监控超大数据集的内存使用

## 数据格式规范

### NumPy数组支持

diffai支持所有NumPy数据类型：
- **整数类型**：int8、int16、int32、int64、uint8、uint16、uint32、uint64
- **浮点类型**：float16、float32、float64
- **复数类型**：complex64、complex128
- **布尔类型**：bool

### MATLAB矩阵支持

diffai支持MATLAB数据类型：
- **数值类型**：double、single、int8、int16、int32、int64、uint8、uint16、uint32、uint64
- **复数类型**：复数双精度和单精度
- **逻辑类型**：logical（布尔）
- **字符数组**：元数据的基本支持

## 故障排除

### 常见问题

#### 1. 文件格式错误

```bash
# 验证文件格式
file data.npy

# 检查文件完整性
python -c "import numpy as np; print(np.load('data.npy').shape)"

# 对于MATLAB文件
python -c "import scipy.io; print(scipy.io.loadmat('data.mat').keys())"
```

#### 2. 内存问题

```bash
# 检查文件大小
ls -lh large_data.npy

# 使用流模式（可用时）
diffai --stream large_data_v1.npy large_data_v2.npy
```

#### 3. 精度问题

```bash
# 检查数据精度
python -c "import numpy as np; print(np.load('data.npy').dtype)"

# 相应调整epsilon
diffai data_v1.npy data_v2.npy --epsilon 1e-8
```

## 下一步

- [基本用法](basic-usage.md) - 学习基本操作
- [ML模型比较](ml-model-comparison.md) - PyTorch和Safetensors分析
- [CLI参考](../reference/cli-reference.md) - 完整命令参考