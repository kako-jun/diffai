# 开始使用 diffai

本综合指南将帮助您快速有效地开始使用 `diffai`。

## 什么是 diffai？

`diffai` 是一个AI驱动的差异工具，超越了传统比较。它结合语义理解和机器学习能力，为结构化数据文件之间的变化提供智能洞察。

### 主要优势

- **AI增强分析**: 使用机器学习检测变化中的模式和异常
- **语义理解**: 理解数据变化的含义和上下文
- **多种格式**: 支持JSON、YAML、TOML、XML、CSV等
- **智能洞察**: 提供变化模式的ML驱动分析
- **高级统计**: 数据分布和趋势的统计分析

## 先决条件

开始之前，请确保已安装 `diffai`。详细说明请参见[安装指南](installation.md)。

快速安装:
```bash
cargo install diffai
```

## 基本用法

### 简单文件比较

最基本的用法是比较两个文件：

```bash
# 使用AI分析比较JSON文件
diffai config_v1.json config_v2.json

# 使用ML洞察比较YAML文件
diffai docker-compose.yml docker-compose.new.yml

# 比较TOML文件
diffai Cargo.toml Cargo.toml.backup

# 比较XML文件
diffai settings.xml settings.new.xml

# 使用统计分析比较CSV文件
diffai data.csv data_updated.csv
```

### 输出格式

控制结果显示方式：

```bash
# 用于API集成的JSON输出
diffai --format json file1.json file2.json

# 用于人类可读性的YAML输出
diffai --format yaml config1.yml config2.yml

# 带有ML分析详情的详细输出
diffai --verbose data1.csv data2.csv
```

### 递归目录比较

比较整个目录结构：

```bash
# 比较目录中的所有文件
diffai --recursive dir1/ dir2/

# 将结果保存到文件
diffai --recursive --output results.json dir1/ dir2/
```

## 高级功能

### 机器学习分析

启用AI驱动的分析以获得更深入的洞察：

```bash
# 启用ML异常检测
diffai --epsilon 0.01 dataset1.json dataset2.json

# 使用智能ID匹配进行数组比较
diffai --array-id-key id users1.json users2.json

# 使用正则表达式忽略特定模式
diffai --ignore-keys-regex "timestamp|temp_" log1.json log2.json
```

## 常见用例

### 配置管理

```bash
# 比较应用程序配置
diffai app-config-dev.json app-config-prod.json

# 跟踪基础设施变化
diffai --recursive infrastructure/dev/ infrastructure/prod/
```

### 数据分析

```bash
# 使用ML洞察比较数据集
diffai --verbose --epsilon 0.05 dataset_before.csv dataset_after.csv

# 跟踪用户行为变化
diffai --array-id-key user_id users_jan.json users_feb.json
```

## 获取帮助

有关特定功能的详细信息：

```bash
# 显示所有可用选项
diffai --help

# 获取版本信息
diffai --version
```

## 下一步

- 了解高级AI分析的[ML工作流程](ml-workflows.md)
- 探索[科学数据](scientific-data.md)分析功能
- 查看详细报告的[详细输出](verbose-output.md)
- 查看更多示例的[基本用法](basic-usage.md)

## 故障排除

### 常见问题

**大文件**: 对于非常大的文件，考虑使用 `--path` 专注于特定部分。

**内存使用**: 对于大型数据集，使用 `--format json` 以提高内存效率。

**性能**: 只有在需要详细ML分析时才启用详细模式。

