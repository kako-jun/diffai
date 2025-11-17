# クイックスタート - diffai

5分でdiffaiを始めましょう。diffaiはAI/MLモデル専用の差分ツールで、PyTorchやSafetensorsファイルを比較する際に11種類の包括的な分析機能を自動的に提供します。

## インストール

```bash
# crates.ioからインストール（推奨）
cargo install diffai

# またはソースから
git clone https://github.com/kako-jun/diffai.git
cd diffai && cargo build --release
```

## 基本的な使い方

### MLモデルの比較（自動分析）

```bash
# PyTorchモデル - 11のML分析が自動的に実行
diffai model_old.pt model_new.pt

# Safetensors - 11のML分析が自動的に実行  
diffai checkpoint_v1.safetensors checkpoint_v2.safetensors

# 出力例：
learning_rate_analysis: old=0.001, new=0.0015, change=+50.0%, trend=increasing
optimizer_comparison: type=Adam, momentum_change=+2.1%, state_evolution=stable
gradient_analysis: flow_health=healthy, norm=0.021069, variance_change=+15.3%
quantization_analysis: mixed_precision=FP16+FP32, compression=12.5%, precision_loss=1.2%
convergence_analysis: status=converging, stability=0.92, plateau_detected=false
# ... + 他6つの分析
  ~ fc1.weight: mean=-0.0002->-0.0001, std=0.0514->0.0716
  ~ fc2.weight: mean=-0.0008->-0.0018, std=0.0719->0.0883
```

### 科学データ（基本分析）

```bash
# NumPy配列 - テンソル統計のみ
diffai experiment_v1.npy experiment_v2.npy

# MATLABファイル - テンソル統計のみ
diffai simulation_v1.mat simulation_v2.mat
```

## 出力形式

### JSON（MLOps統合）
```bash
diffai model1.safetensors model2.safetensors --output json
```

### YAML（人間が読みやすいレポート）
```bash
diffai model1.safetensors model2.safetensors --output yaml
```

### 詳細モード（診断情報）
```bash
diffai model1.safetensors model2.safetensors --verbose
```

## diffaiの特徴

### 自動ML分析
- **設定不要**：PyTorch/Safetensorsに対して11のML分析機能が自動的に実行
- **設定より規約**：lawkitパターンに従ったゼロセットアップ体験
- **diffx-coreベース**：実証済みの信頼性の高い差分処理

### AI/ML専門設計
- **ネイティブテンソルサポート**：PyTorch、Safetensors、NumPy、MATLAB形式を理解
- **統計分析**：自動テンソル統計（平均、標準偏差、形状、メモリ）
- **ML固有の洞察**：勾配分析、量子化検出、収束パターン

### 従来のツール vs diffai
```bash
# 従来のdiff
$ diff model_v1.pt model_v2.pt
Binary files model_v1.pt and model_v2.pt differ

# diffai
$ diffai model_v1.pt model_v2.pt
learning_rate_analysis: old=0.001, new=0.0015, change=+50.0%
gradient_analysis: flow_health=healthy, norm=0.021069
quantization_analysis: mixed_precision=FP16+FP32, compression=12.5%
# ... 包括的なML分析が自動的に実行
```

## 一般的なユースケース

### 研究開発
```bash
# ファインチューニング前後の比較（自動的に包括的分析）
diffai pretrained_model.safetensors finetuned_model.safetensors

# 出力：学習進捗、収束分析、パラメータ進化など
```

### MLOps & CI/CD
```bash
# CI/CDパイプラインでの自動モデル検証
diffai production_model.safetensors candidate_model.safetensors --output json

# jqや他のツールにパイプして自動処理
diffai baseline.pt improved.pt --output json | jq '.gradient_analysis'
```

### モデル最適化
```bash
# 量子化の影響を分析
diffai full_precision.pt quantized.pt
# 自動検出：混合精度、圧縮率、精度損失

# メモリ使用量分析
diffai large_model.safetensors optimized_model.safetensors --verbose
```

## 次のステップ

- **[使用例](examples/)** - 実際のdiffai出力とユースケースを確認
- **[ML分析](ml-analysis_ja.md)** - 11の自動分析機能を理解  
- **[APIリファレンス](reference/api-reference_ja.md)** - Rust/Python/JavaScriptコードでdiffaiを使用
- **[ファイル形式](formats_ja.md)** - サポートされているAI/MLファイル形式の詳細

## 主要オプション

```bash
# 基本比較オプション
--epsilon <FLOAT>           # 浮動小数点比較の許容誤差
--output <FORMAT>           # cli（デフォルト）、json、yaml
--verbose                   # 詳細な診断情報
--no-color                  # カラー出力を無効化

# パスフィルタリング
--path <PATH>               # 特定のパスで差分をフィルタ
--ignore-keys-regex <REGEX> # 正規表現に一致するキーを無視

# メモリ最適化（大規模モデル用）
# メモリ最適化は自動的に行われます - 設定は不要です
```

diffaiは**設定より規約**に従います：AI/MLファイルを検出すると、ML分析が自動的に実行され、セットアップなしで包括的な洞察を提供します。