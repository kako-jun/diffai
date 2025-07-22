# Getting Started with diffai

This comprehensive guide will help you get up and running with `diffai` quickly and effectively.

## What is diffai?

`diffai` is an AI-powered diff tool that goes beyond traditional comparison. It combines semantic understanding with machine learning capabilities to provide intelligent insights into changes between structured data files.

### Key Benefits

- **AI-Enhanced Analysis**: Uses machine learning to detect patterns and anomalies in changes
- **Semantic Understanding**: Understands the meaning and context of data changes
- **Multiple Formats**: Supports JSON, YAML, TOML, XML, CSV, and more
- **Intelligent Insights**: Provides ML-driven analysis of change patterns
- **Advanced Statistics**: Statistical analysis of data distributions and trends

## Prerequisites

Before starting, make sure you have `diffai` installed. See the [Installation Guide](installation.md) for detailed instructions.

Quick install:
```bash
cargo install diffai
```

## Basic Usage

### Simple File Comparison

The most basic usage is comparing two files:

```bash
# Compare JSON files with AI analysis
diffai config_v1.json config_v2.json

# Compare YAML files with ML insights
diffai docker-compose.yml docker-compose.new.yml

# Compare TOML files
diffai Cargo.toml Cargo.toml.backup

# Compare XML files
diffai settings.xml settings.new.xml

# Compare CSV files with statistical analysis
diffai data.csv data_updated.csv
```

### Output Formats

Control how results are displayed:

```bash
# JSON output for API integration
diffai --format json file1.json file2.json

# YAML output for human readability
diffai --format yaml config1.yml config2.yml

# Verbose output with ML analysis details
diffai --verbose data1.csv data2.csv
```

### Recursive Directory Comparison

Compare entire directory structures:

```bash
# Compare all files in directories
diffai --recursive dir1/ dir2/

# Save results to file
diffai --recursive --output results.json dir1/ dir2/
```

## Advanced Features

### Machine Learning Analysis

Enable AI-powered analysis for deeper insights:

```bash
# Enable ML anomaly detection
diffai --epsilon 0.01 dataset1.json dataset2.json

# Array comparison with intelligent ID matching
diffai --array-id-key id users1.json users2.json

# Ignore specific patterns with regex
diffai --ignore-keys-regex "timestamp|temp_" log1.json log2.json
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

# Track infrastructure changes
diffai --recursive infrastructure/dev/ infrastructure/prod/
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

- Learn about [ML Workflows](ml-workflows.md) for advanced AI analysis
- Explore [Scientific Data](scientific-data.md) analysis capabilities  
- Check out [Verbose Output](verbose-output.md) for detailed reporting
- Review [Basic Usage](basic-usage.md) for more examples

## Troubleshooting

### Common Issues

**Large Files**: For very large files, consider using `--path` to focus on specific sections.

**Memory Usage**: Use `--format json` for better memory efficiency with large datasets.

**Performance**: Enable verbose mode only when you need detailed ML analysis.

### Performance Tips

- Use `--ignore-keys-regex` to skip irrelevant fields
- Specify `--path` for targeted analysis
- Consider `--epsilon` for numerical tolerance in large datasets