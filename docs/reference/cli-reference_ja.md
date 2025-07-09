# CLIリファレンス

diffai v0.2.0のコマンドライン完全リファレンス - AI/ML特化差分ツール

## 概要

```
diffai [OPTIONS] <INPUT1> <INPUT2>
```

## 説明

diffaiは、AI/MLワークフローに特化した差分ツールで、モデル構造、テンソル統計、科学データを理解します。PyTorchモデル、Safetensorsファイル、NumPy配列、MATLAB行列、構造化データファイルを比較し、フォーマットの違いではなく意味的な変更に焦点を当てます。

## 引数

### 必須引数

#### `<INPUT1>`
比較する最初の入力ファイルまたはディレクトリ

- **型**: ファイルパスまたはディレクトリパス
- **形式**: PyTorch (.pt/.pth)、Safetensors (.safetensors)、NumPy (.npy/.npz)、MATLAB (.mat)、JSON、YAML、TOML、XML、INI、CSV
- **特殊**: 標準入力には`-`を使用

#### `<INPUT2>`
比較する2番目の入力ファイルまたはディレクトリ

- **型**: ファイルパスまたはディレクトリパス
- **形式**: INPUT1と同じ
- **特殊**: 標準入力には`-`を使用

**例**:
```bash
diffai model1.safetensors model2.safetensors
diffai data_v1.npy data_v2.npy
diffai experiment_v1.mat experiment_v2.mat
diffai config.json config_new.json
diffai - config.json < input.json
```

## オプション

### 基本オプション

#### `-f, --format <FORMAT>`
入力ファイル形式を明示的に指定

- **可能な値**: `json`, `yaml`, `toml`, `ini`, `xml`, `csv`, `safetensors`, `pytorch`, `numpy`, `npz`, `matlab`
- **デフォルト**: ファイル拡張子から自動検出
- **例**: `--format safetensors`

#### `-o, --output <OUTPUT>`
出力形式を選択

- **可能な値**: `cli`, `json`, `yaml`, `unified`
- **デフォルト**: `cli`
- **例**: `--output json`

#### `-r, --recursive`
ディレクトリを再帰的に比較

- **例**: `diffai dir1/ dir2/ --recursive`

#### `--stats`
MLモデルと科学データの詳細統計を表示

- **例**: `diffai model.safetensors model2.safetensors --stats`

### 高度なオプション

#### `--path <PATH>`
特定のパスで差分をフィルタリング

- **例**: `--path "config.users[0].name"`
- **形式**: JSONPath風の構文

#### `--ignore-keys-regex <REGEX>`
正規表現にマッチするキーを無視

- **例**: `--ignore-keys-regex "^id$"`
- **形式**: 標準正規表現パターン

#### `--epsilon <FLOAT>`
浮動小数点比較の許容値を設定

- **例**: `--epsilon 0.001`
- **デフォルト**: マシンイプシロン

#### `--array-id-key <KEY>`
配列要素識別用のキーを指定

- **例**: `--array-id-key "id"`
- **用途**: 構造化配列の比較

#### `--sort-by-change-magnitude`
変更の大きさでソート（MLモデルのみ）

- **例**: `diffai model1.pt model2.pt --sort-by-change-magnitude`

## ML分析機能

### 学習・収束分析

#### `--learning-progress`
学習進捗の追跡
- **説明**: チェックポイント間の学習進捗を分析
- **出力**: 改善トレンド、変化の大きさ、収束速度
- **例**: `diffai checkpoint_10.pt checkpoint_20.pt --learning-progress`

#### `--convergence-analysis`
収束分析
- **説明**: モデルの安定性と収束状態を評価
- **出力**: 収束ステータス、安定性指標、推奨アクション
- **例**: `diffai model_before.pt model_after.pt --convergence-analysis`

#### `--anomaly-detection`
異常検知
- **説明**: 訓練異常（勾配爆発・消失）を検出
- **出力**: 異常タイプ、重要度、影響層数
- **例**: `diffai normal.pt anomalous.pt --anomaly-detection`

#### `--gradient-analysis`
勾配分析
- **説明**: 勾配の特性とフローを分析
- **出力**: 勾配統計、フロー方向、安定性
- **例**: `diffai model1.pt model2.pt --gradient-analysis`

### アーキテクチャ・パフォーマンス分析

#### `--architecture-comparison`
アーキテクチャ比較
- **説明**: モデル構造と設計を比較
- **出力**: アーキテクチャタイプ、深度、違いの数
- **例**: `diffai resnet.pt transformer.pt --architecture-comparison`

#### `--param-efficiency-analysis`
パラメータ効率分析
- **説明**: モデル間のパラメータ効率を分析
- **出力**: 効率スコア、パラメータ密度、最適化提案
- **例**: `diffai baseline.pt optimized.pt --param-efficiency-analysis`

#### `--memory-analysis`
メモリ分析
- **説明**: メモリ使用量と最適化機会を分析
- **出力**: メモリ差分、GPU推定、効率比
- **例**: `diffai small.pt large.pt --memory-analysis`

#### `--inference-speed-estimate`
推論速度推定
- **説明**: 推論速度とパフォーマンス特性を推定
- **出力**: 速度推定、レイテンシ、スループット
- **例**: `diffai model1.pt model2.pt --inference-speed-estimate`

### MLOps・デプロイ支援

#### `--deployment-readiness`
デプロイ準備評価
- **説明**: デプロイ準備と互換性を評価
- **出力**: 準備スコア、戦略、リスクレベル
- **例**: `diffai production.pt candidate.pt --deployment-readiness`

#### `--regression-test`
回帰テスト
- **説明**: 自動回帰テストを実行
- **出力**: テスト結果、劣化検出、推奨アクション
- **例**: `diffai baseline.pt new_version.pt --regression-test`

#### `--risk-assessment`
リスク評価
- **説明**: デプロイリスクと安定性を評価
- **出力**: リスクレベル、要因、緩和策
- **例**: `diffai current.pt candidate.pt --risk-assessment`

#### `--hyperparameter-impact`
ハイパーパラメータ影響分析
- **説明**: ハイパーパラメータ変更の影響を分析
- **出力**: 影響度、最適化提案、感度分析
- **例**: `diffai model_lr001.pt model_lr0001.pt --hyperparameter-impact`

#### `--learning-rate-analysis`
学習率分析
- **説明**: 学習率の効果と最適化を分析
- **出力**: 最適学習率、収束パターン、調整提案
- **例**: `diffai model1.pt model2.pt --learning-rate-analysis`

#### `--alert-on-degradation`
劣化アラート
- **説明**: 閾値を超えた性能劣化でアラート
- **出力**: アラート状態、劣化率、推奨アクション
- **例**: `diffai baseline.pt new_model.pt --alert-on-degradation`

#### `--performance-impact-estimate`
性能影響推定
- **説明**: 変更の性能影響を推定
- **出力**: 影響推定、性能変化、最適化提案
- **例**: `diffai model1.pt model2.pt --performance-impact-estimate`

### 実験・文書化支援

#### `--generate-report`
レポート生成
- **説明**: 包括的な分析レポートを生成
- **出力**: 詳細レポート、統計、推奨事項
- **例**: `diffai model1.pt model2.pt --generate-report`

#### `--markdown-output`
Markdown出力
- **説明**: Markdown形式でレポートを出力
- **出力**: 構造化Markdown、テーブル、グラフ
- **例**: `diffai model1.pt model2.pt --markdown-output`

#### `--include-charts`
チャート生成
- **説明**: 出力にチャートと視覚化を含める
- **出力**: チャート、グラフ、視覚化
- **例**: `diffai model1.pt model2.pt --include-charts`

#### `--review-friendly`
レビュー向け出力
- **説明**: 人間のレビュー向けに最適化された出力
- **出力**: 読みやすい形式、ハイライト、要約
- **例**: `diffai model1.pt model2.pt --review-friendly`

### 高度分析機能

#### `--embedding-analysis`
埋め込み分析
- **説明**: 埋め込み層の変化と意味的ドリフトを分析
- **出力**: 埋め込み変化、意味的距離、ドリフト検出
- **例**: `diffai model1.pt model2.pt --embedding-analysis`

#### `--similarity-matrix`
類似度行列
- **説明**: モデル比較用の類似度行列を生成
- **出力**: 類似度行列、相関、クラスタリング
- **例**: `diffai model1.pt model2.pt --similarity-matrix`

#### `--clustering-change`
クラスタリング変化
- **説明**: モデル表現のクラスタリング変化を分析
- **出力**: クラスタ変化、分離度、構造変化
- **例**: `diffai model1.pt model2.pt --clustering-change`

#### `--attention-analysis`
アテンション分析
- **説明**: アテンション機構パターンを分析（Transformer）
- **出力**: アテンションパターン、重要度、特化度
- **例**: `diffai transformer1.pt transformer2.pt --attention-analysis`

#### `--head-importance`
ヘッド重要度
- **説明**: アテンションヘッドの重要度と特化を分析
- **出力**: ヘッド重要度、特化パターン、最適化提案
- **例**: `diffai model1.pt model2.pt --head-importance`

#### `--attention-pattern-diff`
アテンションパターン差分
- **説明**: モデル間のアテンションパターンを比較
- **出力**: パターン差分、変化検出、影響分析
- **例**: `diffai model1.pt model2.pt --attention-pattern-diff`

### 追加分析機能

#### `--quantization-analysis`
量子化分析
- **説明**: 量子化効果と効率を分析
- **出力**: 圧縮率、速度向上、精度損失、適合性
- **例**: `diffai fp32.pt quantized.pt --quantization-analysis`

#### `--change-summary`
変更要約
- **説明**: 変更の詳細要約を生成
- **出力**: 変更要約、統計、影響分析
- **例**: `diffai model1.pt model2.pt --change-summary`

## 出力例

### CLI出力（デフォルト）

```bash
$ diffai model_v1.safetensors model_v2.safetensors --stats
  ~ fc1.bias: mean=0.0018→0.0017, std=0.0518→0.0647
  ~ fc1.weight: mean=-0.0002→-0.0001, std=0.0514→0.0716
  + new_layer.weight: shape=[64, 64], dtype=f32, params=4096
  - old_layer.bias: shape=[256], dtype=f32, params=256
[CRITICAL] anomaly_detection: type=gradient_explosion, severity=critical, affected=2 layers
```

### JSON出力

```bash
$ diffai model1.pt model2.pt --output json --learning-progress
[
  {
    "TensorStatsChanged": [
      "fc1.bias",
      {"mean": 0.0018, "std": 0.0518},
      {"mean": 0.0017, "std": 0.0647}
    ]
  },
  {
    "LearningProgress": [
      "learning_progress",
      {"trend": "improving", "magnitude": 0.0543, "speed": 0.80}
    ]
  }
]
```

### YAML出力

```bash
$ diffai config1.yaml config2.yaml --output yaml
- TensorStatsChanged:
  - fc1.bias
  - mean: 0.0018
    std: 0.0518
  - mean: 0.0017
    std: 0.0647
```

## 環境変数

### `DIFFAI_CONFIG_PATH`
設定ファイルのパスを指定

- **デフォルト**: `~/.config/diffai/config.toml`
- **例**: `export DIFFAI_CONFIG_PATH=/path/to/config.toml`

### `DIFFAI_OUTPUT_FORMAT`
デフォルト出力形式を設定

- **可能な値**: `cli`, `json`, `yaml`, `unified`
- **例**: `export DIFFAI_OUTPUT_FORMAT=json`

### `DIFFAI_STATS_DEFAULT`
デフォルトで統計を表示

- **可能な値**: `true`, `false`
- **例**: `export DIFFAI_STATS_DEFAULT=true`

## 設定ファイル

### 場所
- **ユーザー設定**: `~/.config/diffai/config.toml`
- **プロジェクト設定**: `./diffai.toml`

### 形式

```toml
[default]
output = "cli"
stats = true
epsilon = 1e-6

[ml_analysis]
learning_progress = false
convergence_analysis = false
anomaly_detection = true

[output]
cli_colors = true
json_pretty = true
yaml_flow = false
```

## 終了コード

| コード | 意味 |
|--------|------|
| 0 | 成功（違いが見つかった場合も含む） |
| 1 | エラー（ファイルが見つからない、解析エラーなど） |
| 2 | 無効な引数またはオプション |
| 3 | 設定ファイルエラー |

## パフォーマンス考慮事項

### 大容量ファイル

```bash
# 大容量モデルでのメモリ使用量を削減
diffai large1.safetensors large2.safetensors --epsilon 1e-3

# 特定パスに分析を限定
diffai model1.pt model2.pt --path "classifier"
```

### 高速化のヒント

1. **イプシロン使用**: 小さな差分を無視して処理を高速化
2. **パスフィルタ**: 必要な部分のみを比較
3. **適切な出力形式**: 用途に応じた最適な形式を選択

## トラブルシューティング

### 一般的な問題

#### "Failed to parse" エラー
```bash
# ファイル形式を明示的に指定
diffai --format safetensors model1.safetensors model2.safetensors

# ファイルの整合性を確認
file model.safetensors
```

#### メモリ不足エラー
```bash
# より大きなイプシロンを使用
diffai --epsilon 1e-3 large1.pt large2.pt

# 特定レイヤーのみ分析
diffai --path "classifier" model1.pt model2.pt
```

#### 権限エラー
```bash
# 読み取り権限を確認
ls -la model.safetensors

# 必要に応じて権限を変更
chmod 644 model.safetensors
```

## 関連項目

- [ML モデル比較ガイド](../user-guide/ml-model-comparison_ja.md)
- [科学データ分析](../user-guide/scientific-data_ja.md)
- [出力フォーマット](output-formats_ja.md)
- [サポート形式](formats_ja.md)