# diffai Test: ml/combined_features

**Description:** Comprehensive ML analysis (automatic)

**Command:** `diffai tests/fixtures/ml_models/simple_base.safetensors tests/fixtures/ml_models/simple_modified.safetensors`

**Generated:** Mon Jul 14 02:00:33 PM JST 2025
**Version:** v0.3.4

## Command Output

```
architecture_comparison: type1=feedforward, type2=feedforward, depth=6→6, differences=0, deployment_readiness=ready, "safe_to_upgrade"
  ~ fc1.bias: mean=0.0018->0.0017, std=0.0518->0.0647
  ~ fc1.weight: mean=-0.0002->-0.0001, std=0.0514->0.0716
  ~ fc2.bias: mean=-0.0076->-0.0257, std=0.0661->0.0973
  ~ fc2.weight: mean=-0.0008->-0.0018, std=0.0719->0.0883
  ~ fc3.bias: mean=-0.0074->-0.0130, std=0.1031->0.1093
  ~ fc3.weight: mean=-0.0035->-0.0010, std=0.0990->0.1113
memory_analysis: delta=+0.0MB, gpu_est=0.1MB, efficiency=1.000000, review_friendly="optimal_no_action_needed", "optimal_no_action_needed"
```

**Exit Code:** 0

✅ **Status:** SUCCESS

---
