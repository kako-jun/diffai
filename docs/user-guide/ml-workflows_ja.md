# ML/AI ワークフロー

機械学習・AI開発における diffai の活用方法について説明します。

## ML開発での活用シーン

### 1. モデル開発・改良

```bash
# 新しいアーキテクチャとベースラインの比較
diffai baseline/resnet18.pth experiment/resnet34.pth --show-structure

# ファインチューニング前後の比較
diffai pretrained/model.pth finetuned/model.pth --tensor-details
```

### 2. 実験管理

```bash
# 実験結果の比較
diffai experiment_001/ experiment_002/ --recursive --include "*.json"

# ハイパーパラメータの違いを確認
diffai config/baseline.yaml config/experiment.yaml
```

### 3. モデル最適化

```bash
# 量子化前後の比較
diffai original/model.pth quantized/model.pth --show-structure

# プルーニング後の変化を確認
diffai full/model.pth pruned/model.pth --diff-only
```

## 典型的なワークフロー

### 実験サイクル

```bash
# 1. ベースライン実験の実行
python train.py --config baseline.yaml --output baseline/

# 2. 新しい実験の実行  
python train.py --config experiment.yaml --output experiment/

# 3. 結果の比較
diffai baseline/ experiment/ --recursive --include "*.json" --include "*.pth"

# 4. 詳細な分析
diffai baseline/model.pth experiment/model.pth --show-structure --tensor-details
```

### モデル比較レポート

```bash
# 包括的な比較レポートを生成
diffai baseline/model.pth experiment/model.pth --format json > comparison.json

# レポートの可視化
python scripts/visualize_comparison.py comparison.json
```

## 具体的な活用例

### PyTorchモデルの進化追跡

```bash
# モデルの変更履歴を追跡
for checkpoint in checkpoints/epoch_*.pth; do
  echo "=== Epoch $(basename $checkpoint .pth | cut -d_ -f2) ==="
  diffai checkpoints/epoch_1.pth $checkpoint --show-structure --diff-only
done
```

**出力例:**
```
=== Epoch 5 ===
Model Changes:
  No structural changes
  Parameters: 11,689,512 -> 11,689,512 (0% change)
  Weight updates: 94.3% of parameters modified

=== Epoch 10 ===
Model Changes:
  No structural changes  
  Parameters: 11,689,512 -> 11,689,512 (0% change)
  Weight updates: 87.1% of parameters modified
```

### Safetensors形式での比較

```bash
# Hugging Face形式のモデル比較
diffai model_v1/model.safetensors model_v2/model.safetensors --tensor-details

# 特定のレイヤーのみに注目
diffai model_v1/model.safetensors model_v2/model.safetensors --filter "attention.*"
```

### データセット変更の追跡

```bash
# データセットの変更を追跡
diffai dataset_v1.csv dataset_v2.csv --stats --format json

# 訓練・検証データの分布比較
diffai train.csv val.csv --stats --show-distribution
```

## 継続的インテグレーション

### GitHub Actions での活用

```yaml
name: ML Model Comparison

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
    
    - name: Install diffai
      run: cargo install diffai
    
    - name: Compare models
      run: |
        # PRブランチのモデルとmainブランチのモデルを比較
        git checkout main
        cp models/current.pth /tmp/main_model.pth
        git checkout -
        
        diffai /tmp/main_model.pth models/current.pth --format json > model_diff.json
        
    - name: Comment PR
      uses: actions/github-script@v6
      with:
        script: |
          const fs = require('fs');
          const diff = JSON.parse(fs.readFileSync('model_diff.json', 'utf8'));
          
          const comment = `## Model Comparison Report
          
          **Parameter Count:** ${diff.comparison.param_diff}
          **Structure Changes:** ${diff.comparison.structure_changes}
          **Significant Changes:** ${diff.comparison.significant_changes}
          
          <details>
          <summary>Detailed Comparison</summary>
          
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

### MLflow との統合

```python
# mlflow_integration.py
import mlflow
import subprocess
import json

def compare_models_with_mlflow(run_id1, run_id2):
    """MLflow run間でモデルを比較"""
    
    # MLflowからモデルをダウンロード
    model1_path = mlflow.artifacts.download_artifacts(
        run_id=run_id1, artifact_path="model/model.pth"
    )
    model2_path = mlflow.artifacts.download_artifacts(
        run_id=run_id2, artifact_path="model/model.pth"
    )
    
    # diffaiで比較
    result = subprocess.run([
        "diffai", model1_path, model2_path, "--format", "json"
    ], capture_output=True, text=True)
    
    comparison = json.loads(result.stdout)
    
    # MLflowに結果を記録
    with mlflow.start_run():
        mlflow.log_dict(comparison, "model_comparison.json")
        mlflow.log_metric("param_count_diff", comparison["param_diff"])
        
    return comparison
```

## パフォーマンス分析

### モデルサイズの追跡

```bash
# モデルサイズの変化を追跡
diffai baseline/model.pth optimized/model.pth --show-size-reduction

# 複数のモデルサイズを比較
for model in models/*.pth; do
  size=$(stat -f%z "$model")
  name=$(basename "$model")
  echo "$name: $size bytes"
done | sort -k2 -n
```

### 量子化効果の確認

```bash
# 量子化前後の詳細比較
diffai full_precision/model.pth quantized/model.pth --quantization-analysis

# 精度への影響を確認
diffai full_precision/results.json quantized/results.json --metric-comparison
```

## ベストプラクティス

### 1. 比較の自動化

```bash
# comparison_script.sh
#!/bin/bash

BASELINE="baseline/model.pth"
EXPERIMENT="experiment/model.pth"
REPORT="comparison_report.json"

# 基本比較
diffai $BASELINE $EXPERIMENT --format json > $REPORT

# 構造的な変更があるかチェック
if diffai $BASELINE $EXPERIMENT --diff-only --quiet; then
    echo "No structural changes detected"
else
    echo "WARNING: Structural changes found - review required"
    diffai $BASELINE $EXPERIMENT --show-structure
fi

# パラメータ数の変化をチェック
python scripts/check_param_changes.py $REPORT
```

### 2. チーム共有

```bash
# チーム用の比較レポート
diffai model1.pth model2.pth --output json > team_comparison.json

# Slackに結果を通知
curl -X POST -H 'Content-type: application/json' \
  --data '{"text":"Model comparison completed: see attached report"}' \
  $SLACK_WEBHOOK_URL
```

## 関連ツールとの連携

### Weights & Biases

```python
import wandb
import subprocess
import json

def log_model_comparison(run1, run2):
    # wandbから2つのモデルを比較
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

## 次のステップ

- [設定](configuration_ja.md) - 詳細な設定方法
- [API リファレンス](../api/cli_ja.md) - 全コマンドの詳細
- [実践例](../examples/) - 具体的な使用例