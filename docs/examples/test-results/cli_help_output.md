# diffai Test: cli/help_output

**Description:** Complete help output showing all available options

**Command:** `diffai --help`

**Generated:** Mon Jul 14 02:00:34 PM JST 2025
**Version:** v0.3.4

## Command Output

```
AI/ML specialized diff CLI - PyTorch, Safetensors model comparison, tensor analysis

Usage: diffai [OPTIONS] <INPUT1> <INPUT2>

Arguments:
  <INPUT1>  The first input (file path or directory path, use '-' for stdin)
  <INPUT2>  The second input (file path or directory path, use '-' for stdin)

Options:
  -f, --format <FORMAT>
          Input file format [possible values: json, yaml, toml, ini, xml, csv, safetensors, pytorch, numpy, npz, matlab]
  -o, --output <OUTPUT>
          Output format [possible values: diffai, json, yaml]
      --path <PATH>
          Filter differences by a specific path (e.g., "config.users\[0\].name")
      --ignore-keys-regex <IGNORE_KEYS_REGEX>
          Ignore keys matching a regular expression (e.g., "^id$")
      --epsilon <EPSILON>
          Tolerance for float comparisons (e.g., "0.001")
      --array-id-key <ARRAY_ID_KEY>
          Key to use for identifying array elements (e.g., "id")
      --show-layer-impact
          Show layer-by-layer impact analysis for ML models
      --quantization-analysis
          Enable quantization analysis for ML models
      --sort-by-change-magnitude
          Sort differences by change magnitude (ML models only)
      --stats
          Show detailed statistics for ML models
  -v, --verbose
          Show verbose processing information
      --learning-progress
          Analyze learning progress between training checkpoints
      --convergence-analysis
          Perform convergence analysis for training stability
      --anomaly-detection
          Detect training anomalies (gradient explosion, vanishing gradients)
      --gradient-analysis
          Analyze gradient characteristics and stability
      --memory-analysis
          Analyze memory usage and efficiency between models
      --inference-speed-estimate
          Estimate inference speed and performance characteristics
      --regression-test
          Perform automated regression testing
      --alert-on-degradation
          Alert on performance degradation beyond thresholds
      --review-friendly
          Generate review-friendly output for human reviewers
      --change-summary
          Generate detailed change summary
      --risk-assessment
          Assess deployment risk and readiness
      --architecture-comparison
          Compare model architectures and structural differences
      --param-efficiency-analysis
          Analyze parameter efficiency between models
      --hyperparameter-impact
          Analyze hyperparameter impact on model changes
      --learning-rate-analysis
          Analyze learning rate effects and patterns
      --deployment-readiness
          Assess deployment readiness and safety
      --performance-impact-estimate
          Estimate performance impact of model changes
      --generate-report
          Generate comprehensive analysis report
      --markdown-output
          Output results in markdown format
      --include-charts
          Include charts and visualizations in output
      --embedding-analysis
          Analyze embedding layer changes and semantic drift
      --similarity-matrix
          Generate similarity matrix for model comparison
      --clustering-change
          Analyze clustering changes in model representations
      --attention-analysis
          Analyze attention mechanism patterns (Transformer models)
      --head-importance
          Analyze attention head importance and specialization
      --attention-pattern-diff
          Compare attention patterns between models
      --hyperparameter-comparison
          Compare hyperparameters from JSON/YAML configs (Phase 2)
      --learning-curve-analysis
          Analyze learning curves from training logs (Phase 2)
      --statistical-significance
          Perform statistical significance testing for metric changes (Phase 2)
  -h, --help
          Print help
  -V, --version
          Print version
```

**Exit Code:** 0

✅ **Status:** SUCCESS

---
