# 基本的な使い方

diffai の基本的な操作方法について説明します。

## 🚀 クイックスタート

### 基本的なファイル比較

```bash
# 2つのファイルを比較
diffai file1.txt file2.txt

# 詳細な出力
diffai file1.txt file2.txt --verbose

# 出力形式の指定
diffai file1.txt file2.txt --format json
```

### ディレクトリ比較

```bash
# ディレクトリ全体を比較
diffai dir1/ dir2/ --recursive

# 特定の拡張子のみ
diffai dir1/ dir2/ --include "*.py" --include "*.json"
```

## 🤖 AI/ML特化機能

### PyTorchモデルの比較

```bash
# PyTorchモデルファイルの比較
diffai model1.pth model2.pth

# 詳細な構造情報を表示
diffai model1.pth model2.pth --show-structure

# 差分のみを表示
diffai model1.pth model2.pth --diff-only
```

**出力例:**
```
=== PyTorch Model Comparison ===

📊 Model Structure:
  ├─ model1.pth: ResNet-18 (11.7M params)
  └─ model2.pth: ResNet-34 (21.8M params)

🔍 Layer Differences:
  + model2.pth: layer4.1.conv2 (512x512x3x3)
  + model2.pth: layer4.1.bn2 (512 features)
  - model1.pth: Only has 2 blocks in layer4

📈 Parameter Count:
  model1.pth: 11,689,512 parameters
  model2.pth: 21,797,672 parameters
  Difference: +10,108,160 parameters (+86.4%)
```

### Safetensorsファイルの比較

```bash
# Safetensorsファイルの比較
diffai model1.safetensors model2.safetensors

# テンソルの詳細情報
diffai model1.safetensors model2.safetensors --tensor-details
```

### データセットの比較

```bash
# CSVデータセットの比較
diffai train.csv test.csv --format csv

# JSONデータセットの比較
diffai dataset1.json dataset2.json --format json

# 統計情報の表示
diffai train.csv test.csv --stats
```

## 📋 コマンドオプション

### 基本オプション

| オプション | 説明 | 例 |
|-----------|------|-----|
| `--format` | 出力形式を指定 | `--format json` |
| `--verbose` | 詳細出力 | `--verbose` |
| `--quiet` | 最小出力 | `--quiet` |
| `--color` | カラー出力の制御 | `--color always` |

### ファイル処理オプション

| オプション | 説明 | 例 |
|-----------|------|-----|
| `--recursive` | ディレクトリを再帰的に処理 | `--recursive` |
| `--include` | 含めるファイルパターン | `--include "*.py"` |
| `--exclude` | 除外するファイルパターン | `--exclude "*.pyc"` |
| `--follow-symlinks` | シンボリックリンクを追跡 | `--follow-symlinks` |

### AI/ML専用オプション

| オプション | 説明 | 例 |
|-----------|------|-----|
| `--show-structure` | モデル構造を表示 | `--show-structure` |
| `--tensor-details` | テンソル詳細情報 | `--tensor-details` |
| `--diff-only` | 差分のみを表示 | `--diff-only` |
| `--stats` | 統計情報を表示 | `--stats` |

## 🎨 出力形式

### デフォルト形式

標準的な diff 形式で出力：

```
--- model1.pth
+++ model2.pth
@@ -1,3 +1,4 @@
 layer1.conv1: Conv2d(3, 64, kernel_size=(7, 7))
 layer1.bn1: BatchNorm2d(64, eps=1e-05)
+layer1.relu: ReLU(inplace=True)
 layer1.maxpool: MaxPool2d(kernel_size=3, stride=2)
```

### JSON形式

```bash
diffai model1.pth model2.pth --format json
```

```json
{
  "comparison": {
    "file1": "model1.pth",
    "file2": "model2.pth",
    "type": "pytorch",
    "differences": [
      {
        "type": "added",
        "layer": "layer1.relu",
        "details": "ReLU(inplace=True)"
      }
    ]
  }
}
```

### カスタム形式

```bash
# カスタムテンプレートを使用
diffai model1.pth model2.pth --template custom.jinja2
```

## 🔧 設定ファイル

### グローバル設定

`~/.config/diffai/config.toml`:

```toml
[defaults]
format = "default"
color = "auto"
verbose = false

[pytorch]
show_structure = true
tensor_details = false

[output]
pager = "less"
max_lines = 1000
```

### プロジェクト設定

`.diffai.toml`:

```toml
[project]
name = "my-ml-project"

[include]
patterns = ["*.py", "*.pth", "*.safetensors"]

[exclude]
patterns = ["*.pyc", "__pycache__/*"]

[pytorch]
show_structure = true
```

## 🎯 実用的な例

### 実験の比較

```bash
# 2つの実験結果を比較
diffai experiment_v1/ experiment_v2/ --recursive --include "*.json"

# モデルチェックポイントの比較
diffai checkpoints/epoch_10.pth checkpoints/epoch_20.pth --show-structure
```

### CI/CDでの使用

```yaml
- name: Compare models
  run: |
    diffai baseline/model.pth new/model.pth --format json > model_diff.json
    
- name: Check significant changes
  run: |
    if diffai baseline/model.pth new/model.pth --diff-only --quiet; then
      echo "No significant model changes"
    else
      echo "Model has changed - review required"
      exit 1
    fi
```

## 📚 次のステップ

- [ML/AI ワークフロー](ml-workflows_ja.md) - ML開発での活用法
- [設定](configuration_ja.md) - 詳細な設定方法
- [API リファレンス](../api/cli_ja.md) - 全コマンドの詳細