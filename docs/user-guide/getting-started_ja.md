# Getting Started with diffai

This comprehensive guide will help you get up and running with `diffai` quickly and effectively.

## What is diffai?

`diffai` is an AI-powered diff tool that goes beyond traditional comparison. It combines semantic understanding with machine learning capabilities to provide intelligent insights into changes between structured data files.

### Key Benefits

- **AI-Enhanced Analysis**: Uses machine learning to detect patterns and anomalies in changes
- **Semantic Understanding**: Understands the meaning and context of data changes
- **AI/ML特化**: PyTorch (.pt/.pth)、Safetensors (.safetensors)、NumPy (.npy/.npz)、MATLAB (.mat) 形式をサポート
- **Intelligent Insights**: Provides ML-driven analysis of change patterns
- **Advanced Statistics**: Statistical analysis of data distributions and trends

## Prerequisites

Before starting, make sure you have `diffai` installed. See the [インストールガイド](installation_ja.md) for detailed instructions.

Quick install:
```bash
cargo install diffai
```

## Basic Usage

### Simple File Comparison

The most basic usage is comparing two files:

```bash
# PyTorchモデルファイルをAI分析で比較
diffai model_v1.pt model_v2.pt

# SafetensorsモデルをML洞察で比較
diffai model1.safetensors model2.safetensors

# NumPy配列を比較
diffai data1.npy data2.npy

# MATLAB行列を比較
diffai simulation1.mat simulation2.mat

# 汎用構造化データ形式にはdiffxを使用:
diffx config_v1.json config_v2.json
diffx docker-compose.yml docker-compose.new.yml
```

### Output Formats

Control how results are displayed:

```bash
# API統合用JSON出力
diffai --output json model1.pt model2.pt

# 人間が読みやすいYAML出力
diffai --output yaml model1.safetensors model2.safetensors

# ML分析詳細付き詳細出力
diffai --verbose data1.npy data2.npy
```

### ディレクトリ比較

自動検出でディレクトリ構造全体を比較：

```bash
# ML分析でディレクトリ内のすべてのファイルを比較
diffai dir1/ dir2/

# 結果をファイルに保存
diffai --output results.json dir1/ dir2/
```

**注意**: ディレクトリ比較（`/` で終わるパス）の場合、diffaiはディレクトリ内のすべてのサポートされたファイルタイプを自動的に検出・分析し、包括的なML主導の洞察を提供します。

## Advanced Features

### Machine Learning Analysis

Enable AI-powered analysis for deeper insights:

```bash
# テンソル比較用ML异常検出を有効化
diffai --epsilon 0.01 model1.safetensors model2.safetensors

# 統計分析付きNumPyアーカイブ比較
diffai --verbose data1.npz data2.npz

# 特定のテンソルパスに焦点を当てる
diffai --path "layer1" model1.pt model2.pt

# 汎用データ分析にはdiffxを使用:
diffx --array-id-key id users1.json users2.json
diffx --ignore-keys-regex "timestamp|temp_" log1.json log2.json
```

### Path-specific Analysis

Focus on specific data paths:

```bash
# Analyze specific configuration paths
diffai --path "database.config" app1.json app2.json

# Multiple path analysis
diffai --path "users[].preferences" user_data1.json user_data2.json
```

### Statistical Insights

Get statistical analysis of your data changes:

```bash
# Verbose mode shows statistical summaries
diffai --verbose financial_q1.csv financial_q2.csv

# Focus on numerical changes with epsilon tolerance
diffai --epsilon 0.001 metrics1.json metrics2.json
```

## Common Use Cases

### Configuration Management

```bash
# Compare application configurations
diffai app-config-dev.json app-config-prod.json

# 自動検出でインフラ構造変更を追跡
diffai infrastructure/dev/ infrastructure/prod/
```

### Data Analysis

```bash
# Compare datasets with ML insights
diffai --verbose --epsilon 0.05 dataset_before.csv dataset_after.csv

# Track user behavior changes
diffai --array-id-key user_id users_jan.json users_feb.json
```

### API Response Comparison

```bash
# Compare API responses
diffai --ignore-keys-regex "timestamp|request_id" api_v1.json api_v2.json

# Focus on data payload only
diffai --path "data" response1.json response2.json
```

## Getting Help

For more detailed information about specific features:

```bash
# Show all available options
diffai --help

# Get version information
diffai --version
```

## Next Steps

- Learn about [ML Workflows](ml-workflows_ja.md) for advanced AI analysis
- Explore [Scientific Data](scientific-data_ja.md) analysis capabilities  
- Check out [Verbose Output](verbose-output_ja.md) for detailed reporting
- Review [Basic Usage](basic-usage_ja.md) for more examples

## Troubleshooting

### Common Issues

**Large Files**: For very large files, consider using `--path` to focus on specific sections.

**Memory Usage**: Use `--format json` for better memory efficiency with large datasets.

**Performance**: Enable verbose mode only when you need detailed ML analysis.

### Performance Tips

- Use `--ignore-keys-regex` to skip irrelevant fields
- Specify `--path` for targeted analysis
- Consider `--epsilon` for numerical tolerance in large datasets