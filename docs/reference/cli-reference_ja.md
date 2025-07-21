# CLIリファレンス

diffai v0.3.4のコマンドライン完全リファレンス - 簡素化されたインターフェースを持つAI/ML特化差分ツール

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

#### `-v, --verbose`
パフォーマンス指標、設定詳細、診断出力を含む詳細な処理情報を表示します。

- **例**: `diffai model1.safetensors model2.safetensors --verbose`
- **用途**: 分析プロセスとパフォーマンスのデバッグ

#### `--no-color`
スクリプト、パイプライン、ANSIカラーをサポートしていない端末との互換性を向上させるため、カラー出力を無効にします。

- **例**: `diffai config.json config.new.json --no-color`
- **用途**: カラーフォーマットなしのプレーンテキスト出力
- **注記**: CI/CD環境や自動化スクリプトで特に有用

## ML分析機能

### ML分析（PyTorch/Safetensorsファイルで自動実行）

**PyTorch（.pt/.pth）およびSafetensors（.safetensors）ファイルに対して、diffaiは自動的に以下を含む包括的な分析を実行します：**

#### 包括的分析スイート（30以上の機能）

- **基本統計**: 各テンソルの平均、標準偏差、最小/最大値、形状、データ型
- **量子化分析**: 圧縮率、精度損失分析
- **アーキテクチャ比較**: 構造検出、レイヤー深度比較、移行評価
- **メモリ分析**: メモリデルタ、ピーク使用量推定、最適化推奨
- **異常検出**: NaN/Inf検出、勾配爆発・消失分析
- **収束分析**: パラメータ安定性、早期停止推奨
- **勾配分析**: 勾配フロー健全性、ノルム推定、問題レイヤー
- **変更サマリ**: 変更幅度分析、パターン、レイヤーランキング
- **類似度行列**: レイヤー間類似度、クラスタリング係数
- **デプロイ準備**: 本番デプロイメント安全性評価
- **リスク評価**: 変更影響評価
- **パフォーマンス影響**: 速度と効率分析
- **パラメータ効率**: 最適化機会
- **回帰テスト**: 品質保証検証
- **学習進度**: トレーニング進度追跡
- **埋め込み分析**: セマンティックドリフト検出
- **アテンション分析**: Transformerアテンションパターン分析
- **統計的有意性**: 変更有意性テスト
- **転移学習分析**: ファインチューニング効果
- **アンサンブル分析**: 複数モデル比較
- **ハイパーパラメータ影響**: 設定変更効果
- **学習率分析**: 最適化スケジュール効果
- **その他多数...**

**🎯 フラグ不要** - 最適なユーザー体験のため、すべての分析が自動的に実行されます。

**例**: `diffai model1.safetensors model2.safetensors` を実行するだけで包括的な分析が得られます。

## 出力例

### CLI出力（デフォルト - 完全分析）

```bash
$ diffai model_v1.safetensors model_v2.safetensors
anomaly_detection: type=none, severity=none, action="continue_training"
architecture_comparison: type1=feedforward, type2=feedforward, deployment_readiness=ready
convergence_analysis: status=converging, stability=0.92
gradient_analysis: flow_health=healthy, norm=0.021069
memory_analysis: delta=+0.0MB, efficiency=1.000000
quantization_analysis: compression=0.0%, speedup=1.8x, precision_loss=1.5%
regression_test: passed=true, degradation=-2.5%, severity=low
deployment_readiness: readiness=0.92, risk=low, timeline=ready_for_immediate_deployment
  ~ fc1.bias: mean=0.0018->0.0017, std=0.0518->0.0647
  ~ fc1.weight: mean=-0.0002->-0.0001, std=0.0514->0.0716
  ~ fc2.bias: mean=-0.0076->-0.0257, std=0.0661->0.0973
  ~ fc2.weight: mean=-0.0008->-0.0018, std=0.0719->0.0883
  ~ fc3.bias: mean=-0.0074->-0.0130, std=0.1031->0.1093
  ~ fc3.weight: mean=-0.0035->-0.0010, std=0.0990->0.1113
```

### 包括的分析の利点

- **30以上の分析機能**が自動実行
- **オプション選択不要** - デフォルトですべての洞察を取得
- **同じ処理時間** - パフォーマンス負荷なし
- **本番対応の洞察** - デプロイ準備、リスク評価など

### 科学データ分析（自動）

```bash
$ diffai experiment_data_v1.npy experiment_data_v2.npy
  ~ data: shape=[1000, 256], mean=0.1234->0.1456, std=0.9876->0.9654, dtype=float64
```

### MATLABファイル比較（自動）

```bash
$ diffai simulation_v1.mat simulation_v2.mat
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

- **DIFFAI_LOG_LEVEL**: ログレベル (error, warn, info, debug)
- **DIFFAI_MAX_MEMORY**: 最大メモリ使用量 (MB単位)

## パフォーマンス考慮事項

- **大容量ファイル**: diffaiはGB+ファイルにストリーミング処理を使用
- **メモリ使用量**: `DIFFAI_MAX_MEMORY`で設定可能なメモリ制限
- **並列処理**: 複数ファイル比較の自動並列化
- **キャッシュ**: 繰り返し比較のためのインテリジェントキャッシュ

## トラブルシューティング

### 一般的な問題

1. **"Binary files differ"メッセージ**: `--format`でファイル型を指定
2. **メモリ不足**: `DIFFAI_MAX_MEMORY`環境変数を設定
3. **処理が遅い**: 分析は大きなモデルに対して自動的に最適化されます
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