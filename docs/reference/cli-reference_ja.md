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

### 現在利用可能（v0.2.0）

以下のML分析機能が現在実装されています：

#### `--stats`
MLモデルと科学データの詳細統計を表示

- **出力**: 各テンソルの平均、標準偏差、最小/最大、形状、データ型
- **例**: `diffai model.safetensors model2.safetensors --stats`

#### `--quantization-analysis`
量子化効果と効率を分析

- **出力**: 圧縮率、精度損失分析
- **例**: `diffai fp32_model.safetensors quantized_model.safetensors --quantization-analysis`

#### `--sort-by-change-magnitude`
優先順位付けのため変更量でソート

- **出力**: 変更量ソートされた差分リスト
- **例**: `diffai model1.pt model2.pt --sort-by-change-magnitude`

#### `--show-layer-impact`
レイヤー別の変更影響を分析

- **出力**: レイヤー別変更分析
- **例**: `diffai baseline.safetensors modified.safetensors --show-layer-impact`

### Phase 3機能（現在利用可能）

#### アーキテクチャ・パフォーマンス分析

##### `--architecture-comparison`
モデルアーキテクチャと構造変化の比較

- **出力**: アーキテクチャタイプ検出、レイヤー深度比較、移行難易度評価
- **例**: `diffai model1.safetensors model2.safetensors --architecture-comparison`

##### `--memory-analysis`
メモリ使用量と最適化機会の分析

- **出力**: メモリデルタ、ピーク使用量推定、GPU利用率、最適化推奨
- **例**: `diffai model1.safetensors model2.safetensors --memory-analysis`

##### `--anomaly-detection`
モデルパラメータの数値異常検出

- **出力**: NaN/Inf検出、勾配爆発・消失分析、死んだニューロン検出
- **例**: `diffai model1.safetensors model2.safetensors --anomaly-detection`

##### `--change-summary`
詳細な変更サマリの生成

- **出力**: 変更幅度、パターン、レイヤーランキング、構造vs パラメータ変更
- **例**: `diffai model1.safetensors model2.safetensors --change-summary`

#### 高度分析

##### `--convergence-analysis`
モデルパラメータの収束パターン分析

- **出力**: 収束状態、パラメータ安定性、早期停止推奨
- **例**: `diffai model1.safetensors model2.safetensors --convergence-analysis`

##### `--gradient-analysis`
パラメータ変更から推定される勾配情報の分析

- **出力**: 勾配フロー健全性、ノルム推定、問題レイヤー、クリッピング推奨
- **例**: `diffai model1.safetensors model2.safetensors --gradient-analysis`

##### `--similarity-matrix`
モデル比較用類似度行列の生成

- **出力**: レイヤー間類似度、クラスタリング係数、外れ値検出
- **例**: `diffai model1.safetensors model2.safetensors --similarity-matrix`

## 出力例

### CLI出力（デフォルト）

```bash
$ diffai model_v1.safetensors model_v2.safetensors --stats
  ~ fc1.bias: mean=0.0018->0.0017, std=0.0518->0.0647
  ~ fc1.weight: mean=-0.0002->-0.0001, std=0.0514->0.0716
  ~ fc2.weight: mean=-0.0008->-0.0018, std=0.0719->0.0883
```

### 組み合わせ分析出力

```bash
$ diffai baseline.safetensors improved.safetensors --stats --quantization-analysis --sort-by-change-magnitude
quantization_analysis: compression=0.25, precision_loss=minimal
  ~ fc2.weight: mean=-0.0008->-0.0018, std=0.0719->0.0883
  ~ fc1.weight: mean=-0.0002->-0.0001, std=0.0514->0.0716
  ~ fc1.bias: mean=0.0018->0.0017, std=0.0518->0.0647
```

### 科学データ分析

```bash
$ diffai experiment_data_v1.npy experiment_data_v2.npy --stats
  ~ data: shape=[1000, 256], mean=0.1234->0.1456, std=0.9876->0.9654, dtype=float64
```

### MATLABファイル比較

```bash
$ diffai simulation_v1.mat simulation_v2.mat --stats
  ~ results: var=results, shape=[500, 100], mean=2.3456->2.4567, std=1.2345->1.3456, dtype=double
  + new_variable: var=new_variable, shape=[100], dtype=single, elements=100, size=0.39KB
```

### JSON出力

```bash
$ diffai model_v1.safetensors model_v2.safetensors --output json
[
  {
    "TensorStatsChanged": [
      "fc1.bias",
      {"mean": 0.0018, "std": 0.0518, "shape": [64], "dtype": "f32"},
      {"mean": 0.0017, "std": 0.0647, "shape": [64], "dtype": "f32"}
    ]
  }
]
```

### YAML出力

```bash
$ diffai model_v1.safetensors model_v2.safetensors --output yaml
- TensorStatsChanged:
  - fc1.bias
  - mean: 0.0018
    std: 0.0518
    shape: [64]
    dtype: f32
  - mean: 0.0017
    std: 0.0647
    shape: [64]
    dtype: f32
```

## 終了コード

- **0**: 成功 - 差分が見つかったまたは差分なし
- **1**: エラー - 無効な引数またはファイルアクセス問題
- **2**: 致命的エラー - 内部処理失敗

## 環境変数

- **DIFFAI_CONFIG**: 設定ファイルのパス
- **DIFFAI_LOG_LEVEL**: ログレベル (error, warn, info, debug)
- **DIFFAI_MAX_MEMORY**: 最大メモリ使用量 (MB単位)

## 設定ファイル

diffaiはTOML形式の設定ファイルに対応しています。設定の場所：

- Unix: `~/.config/diffx/config.toml`
- Windows: `%APPDATA%/diffx/config.toml`
- カレントディレクトリ: `.diffx.toml`

設定例:
```toml
[diffai]
default_output = "cli"
default_format = "auto"
epsilon = 0.001
sort_by_magnitude = false

[ml_analysis]
enable_all = false
learning_progress = true
convergence_analysis = true
memory_analysis = true
```

## パフォーマンス考慮事項

- **大容量ファイル**: diffaiはGB+ファイルにストリーミング処理を使用
- **メモリ使用量**: `DIFFAI_MAX_MEMORY`で設定可能なメモリ制限
- **並列処理**: 複数ファイル比較の自動並列化
- **キャッシュ**: 繰り返し比較のためのインテリジェントキャッシュ

## トラブルシューティング

### 一般的な問題

1. **"Binary files differ"メッセージ**: `--format`でファイル型を指定
2. **メモリ不足**: `DIFFAI_MAX_MEMORY`環境変数を設定
3. **処理が遅い**: 大きなモデルでは必要時のみ`--stats`を使用
4. **依存関係不足**: Rustツールチェーンが適切にインストールされていることを確認

### デバッグモード

デバッグ出力を有効にする：
```bash
DIFFAI_LOG_LEVEL=debug diffai model1.safetensors model2.safetensors
```

## 関連項目

- [基本使用ガイド](../user-guide/basic-usage_ja.md)
- [MLモデル比較ガイド](../user-guide/ml-model-comparison_ja.md)
- [科学データ分析ガイド](../user-guide/scientific-data_ja.md)
- [出力フォーマットリファレンス](output-formats_ja.md)
- [サポートフォーマットリファレンス](formats_ja.md)