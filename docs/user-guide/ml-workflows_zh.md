# ML/AI工作流

机器学习和AI开发中diffai集成指南。

## ML开发用例

### 1. 模型开发与改进

```bash
# 比较新架构与基线
diffai baseline/resnet18.pth experiment/resnet34.pth --show-structure

# 微调前后比较
diffai pretrained/model.pth finetuned/model.pth --tensor-details
```

### 2. 实验管理

```bash
# 比较实验结果
diffai experiment_001/ experiment_002/ --recursive --include "*.json"

# 检查超参数差异
diffai config/baseline.yaml config/experiment.yaml
```

### 3. 模型优化

```bash
# 比较量化前后
diffai original/model.pth quantized/model.pth --show-structure

# 检查剪枝效果
diffai full/model.pth pruned/model.pth --diff-only
```

## 典型工作流

### 实验循环

```bash
# 1. 运行基线实验
python train.py --config baseline.yaml --output baseline/

# 2. 运行新实验
python train.py --config experiment.yaml --output experiment/

# 3. 比较结果
diffai baseline/ experiment/ --recursive --include "*.json" --include "*.pth"

# 4. 详细分析
diffai baseline/model.pth experiment/model.pth --show-structure --tensor-details
```

### 模型比较报告

```bash
# 生成全面比较报告
diffai baseline/model.pth experiment/model.pth --format json > comparison.json

# 可视化报告
python scripts/visualize_comparison.py comparison.json
```

## 实际示例

### PyTorch模型演进跟踪

```bash
# 跨轮次跟踪模型变化
for checkpoint in checkpoints/epoch_*.pth; do
  echo "=== 轮次 $(basename $checkpoint .pth | cut -d_ -f2) ==="
  diffai checkpoints/epoch_1.pth $checkpoint --show-structure --diff-only
done
```

**示例输出：**
```
=== 轮次 5 ===
模型变化：
  无结构变化
  参数：11,689,512 -> 11,689,512（0%变化）
  权重更新：94.3%的参数已修改

=== 轮次 10 ===
模型变化：
  无结构变化  
  参数：11,689,512 -> 11,689,512（0%变化）
  权重更新：87.1%的参数已修改
```

### Safetensors格式比较

```bash
# 比较Hugging Face格式模型
diffai model_v1/model.safetensors model_v2/model.safetensors --tensor-details

# 专注于特定层
diffai model_v1/model.safetensors model_v2/model.safetensors --filter "attention.*"
```

### 数据集变化跟踪

```bash
# 跟踪数据集变化
diffai dataset_v1.csv dataset_v2.csv --stats --format json

# 比较训练/验证数据分布
diffai train.csv val.csv --stats --show-distribution
```

## 持续集成

### GitHub Actions集成

```yaml
name: ML模型比较

on:
  pull_request:
    paths:
      - 'models/**'
      - 'experiments/**'

jobs:
  compare:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: 安装diffai
      run: cargo install diffai
    
    - name: 比较模型
      run: |
        # 比较PR分支模型与main分支模型
        git checkout main
        cp models/current.pth /tmp/main_model.pth
        git checkout -
        
        diffai /tmp/main_model.pth models/current.pth --format json > model_diff.json
        
    - name: 评论PR
      uses: actions/github-script@v6
      with:
        script: |
          const fs = require('fs');
          const diff = JSON.parse(fs.readFileSync('model_diff.json', 'utf8'));
          
          const comment = `## 模型比较报告
          
          **参数数量：** ${diff.comparison.param_diff}
          **结构变化：** ${diff.comparison.structure_changes}
          **重大变化：** ${diff.comparison.significant_changes}
          
          <details>
          <summary>详细比较</summary>
          
          \`\`\`json
          ${JSON.stringify(diff, null, 2)}
          \`\`\`
          </details>`;
          
          github.rest.issues.createComment({
            issue_number: context.issue.number,
            owner: context.repo.owner,
            repo: context.repo.repo,
            body: comment
          });
```

### MLflow集成

```python
# mlflow_integration.py
import mlflow
import subprocess
import json

def compare_models_with_mlflow(run_id1, run_id2):
    """比较MLflow运行之间的模型"""
    
    # 从MLflow下载模型
    model1_path = mlflow.artifacts.download_artifacts(
        run_id=run_id1, artifact_path="model/model.pth"
    )
    model2_path = mlflow.artifacts.download_artifacts(
        run_id=run_id2, artifact_path="model/model.pth"
    )
    
    # 使用diffai比较
    result = subprocess.run([
        "diffai", model1_path, model2_path, "--format", "json"
    ], capture_output=True, text=True)
    
    comparison = json.loads(result.stdout)
    
    # 将结果记录到MLflow
    with mlflow.start_run():
        mlflow.log_dict(comparison, "model_comparison.json")
        mlflow.log_metric("param_count_diff", comparison["param_diff"])
        
    return comparison
```

## 性能分析

### 模型大小跟踪

```bash
# 跟踪模型大小变化
diffai baseline/model.pth optimized/model.pth --show-size-reduction

# 比较多个模型大小
for model in models/*.pth; do
  size=$(stat -f%z "$model")
  name=$(basename "$model")
  echo "$name: $size 字节"
done | sort -k2 -n
```

### 量化影响评估

```bash
# 比较量化前后
diffai full_precision/model.pth quantized/model.pth --quantization-analysis

# 检查精度影响
diffai full_precision/results.json quantized/results.json --metric-comparison
```

## 最佳实践

### 1. 比较自动化

```bash
# comparison_script.sh
#!/bin/bash

BASELINE="baseline/model.pth"
EXPERIMENT="experiment/model.pth"
REPORT="comparison_report.json"

# 基本比较
diffai $BASELINE $EXPERIMENT --format json > $REPORT

# 检查结构变化
if diffai $BASELINE $EXPERIMENT --diff-only --quiet; then
    echo "未检测到结构变化"
else
    echo "警告：发现结构变化 - 需要审查"
    diffai $BASELINE $EXPERIMENT --show-structure
fi

# 检查参数数量变化
python scripts/check_param_changes.py $REPORT
```

### 2. 配置管理

```toml
# .diffai.toml
[project]
name = "my-ml-project"

[pytorch]
show_structure = true
tensor_details = true
filter_small_changes = true
threshold = 0.01

[output]
format = "enhanced"
color = true
max_tensor_display = 10
```

### 3. 团队协作

```bash
# 生成团队比较报告
diffai model1.pth model2.pth --template team_report.jinja2 > team_comparison.html

# 通过Slack通知团队
curl -X POST -H 'Content-type: application/json' \
  --data '{"text":"模型比较已完成：查看附件报告"}' \
  $SLACK_WEBHOOK_URL
```

## 与ML工具集成

### Weights & Biases

```python
import wandb
import subprocess
import json

def log_model_comparison(run1, run2):
    # 比较wandb中的两个模型
    comparison = subprocess.run([
        "diffai", f"wandb_models/{run1}.pth", f"wandb_models/{run2}.pth", 
        "--format", "json"
    ], capture_output=True, text=True)
    
    wandb.log({"model_comparison": json.loads(comparison.stdout)})
```

### TensorBoard

```python
from torch.utils.tensorboard import SummaryWriter
import subprocess
import json

def log_comparison_to_tensorboard(model1, model2, step):
    writer = SummaryWriter()
    
    result = subprocess.run([
        "diffai", model1, model2, "--format", "json"
    ], capture_output=True, text=True)
    
    comparison = json.loads(result.stdout)
    
    writer.add_text("model_comparison", 
                   json.dumps(comparison, indent=2), step)
    writer.close()
```

## 下一步

- [配置](configuration.md) - 高级配置
- [API参考](../api/cli.md) - 完整命令参考
- [示例](../examples/) - 实际使用示例