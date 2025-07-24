# Examples

This page provides comprehensive examples of using diffai in various scenarios, demonstrating its AI-powered analysis capabilities.

## Basic Usage Examples

### JSON Configuration Files

Compare application configuration files with AI insights:

```bash
# Basic JSON comparison
diffai config-dev.json config-prod.json

# Output with ML analysis
diffai --verbose config-dev.json config-prod.json

# Focus on specific configuration paths
diffai --path "database.settings" config-dev.json config-prod.json
```

**Example files:**
```json
// config-dev.json
{
  "app_name": "myapp",
  "version": "1.0.0",
  "database": {
    "host": "localhost",
    "port": 5432,
    "name": "dev_db"
  },
  "debug": true
}

// config-prod.json  
{
  "app_name": "myapp",
  "version": "1.0.1",
  "database": {
    "host": "prod-server.com",
    "port": 5432,
    "name": "prod_db"
  },
  "debug": false
}
```

### YAML Docker Compose Files

```bash
# Compare Docker Compose configurations
diffai docker-compose.yml docker-compose.prod.yml

# Ignore timestamp-related changes
diffai --ignore-keys-regex "created_at|updated_at" docker-compose.yml docker-compose.new.yml
```

### CSV Data Analysis

Compare datasets with statistical analysis:

```bash
# Compare sales data with ML insights
diffai --verbose --epsilon 0.05 sales-q1.csv sales-q2.csv

# Focus on specific columns
diffai --path "revenue,profit" financial-data-old.csv financial-data-new.csv
```

## Advanced AI Analysis Examples

### Anomaly Detection

```bash
# Detect anomalies in user behavior data
diffai --epsilon 0.01 --verbose user-metrics-baseline.json user-metrics-current.json

# Array comparison with intelligent ID matching
diffai --array-id-key "user_id" users-jan.json users-feb.json
```

### Machine Learning Model Comparison

```bash
# Compare ML model configurations
diffai --verbose --path "hyperparameters" model-v1.json model-v2.json

# Compare training results with statistical tolerance
diffai --epsilon 0.001 training-results-baseline.json training-results-new.json
```

### Scientific Data Analysis

```bash
# Compare experimental datasets
diffai --verbose --epsilon 0.01 experiment-control.csv experiment-test.csv

# Compare research directories (automatic directory detection)
diffai --output analysis-report.json research-v1/ research-v2/
```

## Format-Specific Examples

### XML Configuration Files

```bash
# Compare Spring Boot configurations
diffai application-dev.xml application-prod.xml

# Maven POM file comparison
diffai pom.xml pom.xml.backup
```

### TOML Cargo Files

```bash
# Compare Rust project dependencies
diffai Cargo.toml Cargo.toml.new

# Focus on dependency changes only
diffai --path "dependencies" Cargo.toml Cargo.toml.updated
```

### INI Configuration Files

```bash
# Compare application settings
diffai app.ini app-new.ini

# Database configuration comparison
diffai database.ini database-backup.ini
```

## Integration Examples

### CI/CD Pipeline Integration

```bash
#\!/bin/bash
# Automated configuration drift detection
diffai --format json --output config-drift.json \
  production-config.json staging-config.json

# Exit with error if significant changes detected
if [ $(jq '.changes | length' config-drift.json) -gt 10 ]; then
  echo "Significant configuration drift detected\!"
  exit 1
fi
```

### API Response Monitoring

```bash
# Compare API responses for regression testing
diffai --ignore-keys-regex "timestamp|request_id" \
  api-response-baseline.json api-response-current.json

# Focus on data payload changes only
diffai --path "data" api-v1-response.json api-v2-response.json
```

### Database Schema Evolution

```bash
# Compare database schema exports
diffai --verbose schema-v1.json schema-v2.json

# Track migration changes (automatic directory detection)
diffai migrations-old/ migrations-new/
```

## Output Format Examples

### JSON Output for Automation

```bash
# Generate machine-readable reports
diffai --format json --output report.json dataset1.csv dataset2.csv

# Pipe to jq for processing
diffai --format json file1.json file2.json | jq '.summary.change_count'
```

### YAML Output for Human Review

```bash
# Human-friendly reports
diffai --format yaml --verbose config1.yml config2.yml > changes-report.yml
```

### Verbose Analysis Reports

```bash
# Comprehensive ML analysis
diffai --verbose --epsilon 0.01 \
  --output detailed-analysis.json \
  large-dataset-before.csv large-dataset-after.csv
```

## Real-World Scenarios

### Monitoring System Configuration

```bash
# Daily configuration drift check (automatic directory detection)
diffai --format json \
  /etc/production-config/ /etc/staging-config/ > daily-drift-report.json
```

### Data Quality Assurance

```bash
# Compare data quality metrics
diffai --verbose --epsilon 0.05 \
  quality-metrics-baseline.json quality-metrics-current.json
```

### Performance Regression Testing

```bash
# Compare performance benchmarks
diffai --epsilon 0.1 --path "performance_metrics" \
  benchmark-baseline.json benchmark-current.json
```

## Tips and Best Practices

### Choosing the Right Options

- Use `--epsilon` for numerical tolerance in scientific data
- Use `--ignore-keys-regex` for dynamic fields like timestamps
- Use `--path` to focus analysis on specific data sections
- Use `--array-id-key` for intelligent array element matching

### Performance Optimization

- Start with basic comparison, add `--verbose` only when needed
- Use `--format json` for large datasets to reduce memory usage
- Consider `--path` filtering for very large files

### Integration Patterns

- Combine with `jq` for JSON output processing
- Use exit codes in scripts for automated decision making
- Generate reports in CI/CD for change tracking

