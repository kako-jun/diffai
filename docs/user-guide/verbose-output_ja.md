# Verbose Output Guide

The `--verbose` flag in diffai provides comprehensive diagnostic information to help users understand processing details, debug issues, and analyze performance. This guide explains how to use verbose mode effectively.

## Overview

Verbose mode displays detailed information about:
- Configuration settings and enabled features
- File analysis and format detection
- Processing time and performance metrics
- ML-specific analysis status
- Directory comparison statistics

## Basic Usage

### Enable Verbose Mode

```bash
# AI/MLファイルでの基本的な詳細出力
diffai model1.safetensors model2.safetensors --verbose

# 短縮形
diffai model1.pt model2.pt -v

# 汎用フォーマットにはdiffxを使用:
# diffx file1.json file2.json --verbose
```

### Example Output

```
=== diffai verbose mode enabled ===
Configuration:
  Input format: None
  Output format: Cli
  Recursive mode: false

File analysis:
  Input 1: model1.safetensors
  Input 2: model2.safetensors
  Detected format: Safetensors
  File 1 size: 156 bytes
  File 2 size: 162 bytes

Processing results:
  Total processing time: 234.567µs
  Differences found: 3
  Format-specific analysis: Safetensors
```

## Configuration Diagnostics

Verbose mode displays all active configuration options:

### Basic Configuration
- **Input format**: Explicitly set or auto-detected format
- **Output format**: diffai, JSON, or YAML
- **Recursive mode**: Whether directory comparison is enabled

### Advanced Options
```bash
diffai file1.json file2.json --verbose \
  --epsilon 0.001 \
  --ignore-keys-regex "^id$" \
  --path "config.users"
```

Output includes:
```
Configuration:
  Input format: None
  Output format: Cli
  Recursive mode: false
  Epsilon tolerance: 0.001
  Ignore keys regex: ^id$
  Path filter: config.users
```

### ML Analysis Features
When ML analysis options are enabled, verbose mode shows which features are active:

```bash
diffai model1.safetensors model2.safetensors --verbose \
  \
  --architecture-comparison \
  --memory-analysis \
  --anomaly-detection
```

Output:
```
Configuration:
  ML analysis features: statistics, architecture_comparison, memory_analysis, anomaly_detection
```

## File Analysis Information

### File Metadata
Verbose mode provides detailed file information:

- **File paths**: Full paths to input files
- **File sizes**: Exact byte counts for each file
- **Format detection**: How diffai identified the file format

Example:
```
File analysis:
  Input 1: /path/to/model1.safetensors
  Input 2: /path/to/model2.safetensors
  Detected format: Safetensors
  File 1 size: 1048576 bytes
  File 2 size: 1048576 bytes
```

### Format Detection Process
Verbose mode explains how file formats were determined:

1. **Explicit format**: When `--format` is specified
2. **Auto-detection**: Based on file extensions
3. **Fallback logic**: When format cannot be inferred

## Performance Metrics

### Processing Time
Verbose mode measures and reports processing time with microsecond precision:

```
Processing results:
  Total processing time: 1.234567ms
  Differences found: 15
```

### ML/Scientific Data Analysis
For ML and scientific data files, verbose mode indicates completion status:

```
Processing results:
  Format-specific analysis: Safetensors
  ML/Scientific data analysis completed
```

## Directory Comparison

ディレクトリ比較時（自動検出）、詳細モードは追加統計情報を提供します：

```bash
diffai dir1/ dir2/ --verbose
```

Example output:
```
Configuration:
  Directory mode: true (automatic)

Directory scan results:
  Files in /path/to/dir1: 12
  Files in /path/to/dir2: 14
  Total files to compare: 16

Directory comparison summary:
  Files compared: 10
  Files only in one directory: 6
  Total files processed: 16
```

## Use Cases

### Debugging Processing Issues

When diffai behaves unexpectedly, verbose mode helps identify:

1. **Format detection problems**: Check if the correct format was detected
2. **Configuration conflicts**: Verify all options are applied correctly
3. **Performance bottlenecks**: Identify slow processing steps
4. **File access issues**: Confirm files are readable and sizes are expected

Example debugging session:
```bash
# Check if format detection is correct
diffai problematic_file1.dat problematic_file2.dat --verbose

# Verify ML analysis features (automatic for ML models)
diffai model1.pt model2.pt --verbose

# ディレクトリ比較動作を分析（自動検出）
diffai dir1/ dir2/ --verbose
```

### Performance Analysis

Use verbose mode to understand processing performance:

```bash
# Measure processing time for different formats
diffai large_model1.safetensors large_model2.safetensors --verbose

# Compare performance with different options
diffai data1.json data2.json --verbose --epsilon 0.0001
```

### Configuration Validation

Verify that complex configurations are applied correctly:

```bash
# Check multiple filters and options
diffai config1.yaml config2.yaml --verbose \
  --ignore-keys-regex "^(id|timestamp)$" \
  --path "application.settings" \
  --epsilon 0.01 \
  --output json
```

## Tips and Best Practices

### 1. Always Use Verbose for Debugging
When encountering unexpected behavior, always add `--verbose` to understand what diffai is doing.

### 2. Performance Monitoring
Use verbose mode to monitor processing time for large files and optimize accordingly.

### 3. Configuration Verification
Before running batch operations, use verbose mode on a single file to verify configuration.

### 4. Learning File Formats
Use verbose mode to understand how diffai detects and processes different file formats.

### 5. ML Analysis Optimization
When using multiple ML analysis features, verbose mode helps identify which features are active and their impact on processing time.

## Output Redirection

### Separate Verbose from Results
Verbose information is sent to stderr, allowing you to separate it from results:

```bash
# Save results to file, show verbose on screen
diffai file1.json file2.json --verbose --output json > results.json

# Save both results and verbose output
diffai file1.json file2.json --verbose > results.txt 2> verbose.log

# Show only verbose information
diffai file1.json file2.json --verbose 2>&1 >/dev/null
```

## Integration with Other Tools

### CI/CD Pipelines
Use verbose mode in CI/CD for debugging:

```bash
# In GitHub Actions or similar
- name: Compare models with verbose output
  run: diffai baseline.safetensors new_model.safetensors --verbose
```

### Scripts and Automation
Parse verbose output for automation:

```bash
#!/bin/bash
output=$(diffai file1.json file2.json --verbose 2>&1)
if echo "$output" | grep -q "Differences found: 0"; then
    echo "Files are identical"
else
    echo "Files differ"
fi
```

## Related Commands

- [`--help`](basic-usage.md#help): Show all available options
- [`--output`](../reference/output-formats_ja.md): Control output format
- [ディレクトリ比較](directory-comparison_ja.md): 自動ディレクトリ検出