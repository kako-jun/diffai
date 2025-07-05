# CLI Reference

Complete command-line reference for diffai.

## 📋 Synopsis

```
diffai [OPTIONS] <INPUT1> <INPUT2>
```

## 📝 Description

diffai is a specialized diff tool for AI/ML workflows that understands model structures, tensor statistics, and experiment data. It compares structured data files and ML models, focusing on semantic changes rather than formatting differences.

## 🔧 Arguments

### Required Arguments

#### `<INPUT1>` 
First input file or directory to compare.

- **Type**: File path or directory path
- **Formats**: JSON, YAML, TOML, XML, INI, CSV, PyTorch (.pt/.pth), Safetensors (.safetensors)
- **Special**: Use `-` for stdin

**Examples**:
```bash
diffai model1.safetensors model2.safetensors
diffai config.json config_new.json
diffai - config.json < input.json
```

#### `<INPUT2>`
Second input file or directory to compare.

- **Type**: File path or directory path  
- **Formats**: Same as INPUT1
- **Special**: Use `-` for stdin (only one input can be stdin)

## ⚙️ Options

### Format Options

#### `-f, --format <FORMAT>`
Specify input file format explicitly.

- **Type**: Enum
- **Values**: `json`, `yaml`, `toml`, `ini`, `xml`, `csv`, `safetensors`, `pytorch`
- **Default**: Auto-detected from file extension

**Examples**:
```bash
# Explicit format specification
diffai --format json file1 file2

# Useful when file extension is ambiguous
diffai --format safetensors model.bin model_new.bin
```

**Auto-detection Rules**:
| Extension | Detected Format |
|-----------|----------------|
| `.json` | JSON |
| `.yaml`, `.yml` | YAML |
| `.toml` | TOML |
| `.ini` | INI |
| `.xml` | XML |
| `.csv` | CSV |
| `.safetensors` | Safetensors |
| `.pt`, `.pth` | PyTorch |

### Output Options

#### `-o, --output <FORMAT>`
Specify output format.

- **Type**: Enum  
- **Values**: `cli`, `json`, `yaml`, `unified`
- **Default**: `cli`

**Examples**:
```bash
# Human-readable CLI output (default)
diffai model1.safetensors model2.safetensors

# Machine-readable JSON
diffai model1.safetensors model2.safetensors --output json

# YAML format for configuration
diffai config1.yaml config2.yaml --output yaml

# Git-compatible unified diff
diffai data1.json data2.json --output unified
```

**Output Format Details**:

| Format | Use Case | Features |
|--------|----------|----------|
| `cli` | Human review | Colored output, ML symbols (📊⬚), hierarchical display |
| `json` | Automation | Machine-readable, structured data, API integration |
| `yaml` | Configuration | Human-readable structured format |
| `unified` | Git integration | Traditional diff format, merge tool compatibility |

### Comparison Options

#### `--epsilon <VALUE>`
Set tolerance for floating-point comparisons.

- **Type**: Float
- **Default**: Exact comparison (no tolerance)
- **Range**: Any positive floating-point number

**Examples**:
```bash
# Ignore tiny differences (numerical noise)
diffai model1.safetensors model2.safetensors --epsilon 1e-6

# Quantization analysis (larger tolerance)
diffai fp32_model.safetensors int8_model.safetensors --epsilon 0.01

# Training progress (medium tolerance)
diffai checkpoint1.pt checkpoint2.pt --epsilon 1e-4
```

**Use Cases by Epsilon Value**:
| Epsilon | Use Case | Description |
|---------|----------|-------------|
| None | Exact comparison | Detect all changes, including numerical noise |
| `1e-8` | High precision | Scientific computing, minimal tolerance |
| `1e-6` | Standard ML | Normal model training, ignore floating-point errors |
| `1e-4` | Training progress | Focus on significant learning changes |
| `0.01` | Quantization | Account for precision loss in quantized models |
| `0.1` | Architecture focus | Ignore small weight changes, focus on structure |

### Filtering Options

#### `--path <PATH>`
Filter differences by specific path.

- **Type**: String
- **Format**: Dot notation for nested objects, bracket notation for arrays
- **Default**: Show all paths

**Examples**:
```bash
# Focus on classifier layer only
diffai model1.safetensors model2.safetensors --path "tensor.classifier"

# Specific configuration section
diffai config1.json config2.json --path "database.connection"

# Array element with ID
diffai users1.json users2.json --path "users[id=123]"
```

**Path Syntax**:
```
# Object properties
config.database.host

# Array indices  
users[0].name

# Array elements by ID (when using --array-id-key)
users[id=123].email

# Nested structures
model.layers[0].weights.data

# ML model tensors
tensor.bert.encoder.layer.11.attention.self.query.weight
```

#### `--ignore-keys-regex <REGEX>`
Ignore keys matching regular expression.

- **Type**: Regular expression string
- **Default**: None (compare all keys)

**Examples**:
```bash
# Ignore timestamp fields
diffai config1.json config2.json --ignore-keys-regex "^timestamp$"

# Ignore multiple metadata fields
diffai model1.safetensors model2.safetensors --ignore-keys-regex "^(_metadata|timestamp|run_id)$"

# Ignore version info
diffai package1.json package2.json --ignore-keys-regex "^version"

# Ignore temporary or generated fields
diffai data1.json data2.json --ignore-keys-regex "^(tmp_|generated_|_temp)"
```

**Common Regex Patterns**:
| Pattern | Matches | Use Case |
|---------|---------|----------|
| `^timestamp$` | Exact "timestamp" key | Ignore timestamps |
| `^_.*` | Keys starting with underscore | Ignore private fields |
| `.*_temp$` | Keys ending with "_temp" | Ignore temporary data |
| `^(id\|uuid)$` | "id" or "uuid" keys | Ignore identifiers |
| `version` | Any key containing "version" | Ignore version fields |

#### `--array-id-key <KEY>`
Key to use for identifying array elements.

- **Type**: String
- **Default**: Index-based comparison
- **Purpose**: Improved array element tracking

**Examples**:
```bash
# Track users by ID field
diffai users1.json users2.json --array-id-key "id"

# Track tasks by UUID
diffai tasks1.json tasks2.json --array-id-key "uuid" 

# Track model layers by name
diffai config1.json config2.json --array-id-key "layer_name"
```

**Without array-id-key (index-based)**:
```json
// Changes shown as index modifications
[0].name: "Alice" -> "Bob"
[1]: {"name": "Charlie", "age": 30} (added)
```

**With array-id-key="id" (ID-based)**:
```json
// Changes shown with semantic meaning
[id=1].name: "Alice" -> "Bob"  
[id=3]: {"id": 3, "name": "Charlie", "age": 30} (added)
```

### Directory Options

#### `-r, --recursive`
Compare directories recursively.

- **Type**: Boolean flag
- **Default**: `false` (single file comparison)

**Examples**:
```bash
# Compare all files in directories
diffai config_dir1/ config_dir2/ --recursive

# Combined with format specification
diffai experiments_v1/ experiments_v2/ --recursive --format json

# With filtering
diffai models_old/ models_new/ --recursive --ignore-keys-regex "^timestamp$"
```

**Directory Comparison Behavior**:
- Compares files with same relative paths
- Shows files unique to each directory
- Auto-detects format for each file pair
- Applies all filtering options to each comparison

### Help and Version

#### `-h, --help`
Display help information.

```bash
diffai --help
diffai -h
```

#### `-V, --version`
Display version information.

```bash
diffai --version
diffai -V
```

## 🔗 Combined Options

### ML Model Analysis

```bash
# Comprehensive model comparison
diffai model1.safetensors model2.safetensors \
  --epsilon 1e-6 \
  --output json \
  --path "tensor.classifier" \
  > analysis.json

# Training progress analysis
diffai checkpoint_epoch_10.pt checkpoint_epoch_50.pt \
  --epsilon 1e-4 \
  --ignore-keys-regex "^(optimizer_state|scheduler_state)" \
  --output yaml
```

### Configuration Management

```bash
# Environment config comparison
diffai config_dev.json config_prod.json \
  --ignore-keys-regex "^(timestamp|environment|debug)" \
  --output yaml \
  --path "database"

# Recursive directory analysis
diffai config_v1/ config_v2/ \
  --recursive \
  --format json \
  --ignore-keys-regex "^_.*" \
  --output json > config_diff.json
```

### MLOps Integration

```bash
# CI/CD model validation
diffai baseline_model.safetensors candidate_model.safetensors \
  --epsilon 1e-5 \
  --output json \
  --ignore-keys-regex "^(training_metadata|timestamp)" | \
  jq '.[] | select(.TensorShapeChanged)' > architecture_changes.json

# Experiment tracking
diffai experiment_run_123.json experiment_run_124.json \
  --array-id-key "metric_name" \
  --ignore-keys-regex "^(start_time|end_time|run_id)" \
  --path "results.metrics"
```

## 📊 Exit Codes

| Code | Meaning | Description |
|------|---------|-------------|
| `0` | Success | Comparison completed successfully |
| `1` | General error | Invalid arguments, file not found, etc. |
| `2` | Parse error | Unable to parse input files |
| `3` | I/O error | File read/write permissions issue |

## 🔍 Examples

### Basic Usage

```bash
# Simple file comparison
diffai config.json config_new.json

# Model comparison with tolerance
diffai model.safetensors model_v2.safetensors --epsilon 1e-6

# Directory comparison
diffai experiments_old/ experiments_new/ --recursive
```

### Advanced Filtering

```bash
# Focus on specific model layers
diffai transformer_v1.safetensors transformer_v2.safetensors \
  --path "tensor.encoder.layer"

# Ignore metadata and timestamps
diffai config1.json config2.json \
  --ignore-keys-regex "^(_meta|timestamp|created_at)$"

# Track array elements by ID
diffai users_before.json users_after.json \
  --array-id-key "user_id"
```

### Output Formats

```bash
# JSON for automation
diffai data1.json data2.json --output json | jq '.[] | .Added'

# YAML for documentation  
diffai config.yaml config_new.yaml --output yaml > changes.yaml

# Unified diff for git
diffai old.json new.json --output unified | git apply
```

### ML-Specific Workflows

```bash
# Fine-tuning analysis
diffai pretrained.safetensors finetuned.safetensors \
  --epsilon 1e-6 \
  --output json | \
  jq '[.[] | select(.TensorStatsChanged)] | length'

# Quantization impact
diffai fp32_model.safetensors int8_model.safetensors \
  --epsilon 0.01 \
  --path "tensor" \
  --output yaml

# Training checkpoint comparison
diffai checkpoint_1.pt checkpoint_10.pt \
  --epsilon 1e-4 \
  --ignore-keys-regex "^optimizer" \
  --output json > training_progress.json
```

## 🚨 Notes and Limitations

### File Size Limits
- **Recommended maximum**: 10GB per file
- **Memory usage**: ~2x file size during processing
- **Large files**: Consider using `--path` filtering

### Format Support
- **Binary formats**: PyTorch and Safetensors only
- **Text formats**: All encoding types supported  
- **Compressed files**: Not directly supported (decompress first)

### Performance Considerations
- **Large models**: Use appropriate epsilon values
- **Deep structures**: May require significant memory
- **Directory comparison**: Processes files sequentially

### Regex Notes
- Uses Rust regex syntax (similar to Perl)
- Case-sensitive by default
- Full Unicode support
- Performance optimized for common patterns

---

**See Also**: 
- [API Reference](api-reference.md) - Rust library documentation
- [ML Model Comparison](../user-guide/ml-model-comparison.md) - Detailed ML usage guide
- [Examples](../user-guide/examples.md) - Real-world usage scenarios