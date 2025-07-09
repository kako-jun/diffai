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

### 出力オプション

#### `-o, --output <FORMAT>`
出力フォーマットを指定。

- **型**: 列挙型
- **値**: `cli`, `json`, `yaml`, `unified`
- **デフォルト**: `cli`

**例**:
```bash
# 人間が読みやすいCLI出力（デフォルト）
diffai model1.safetensors model2.safetensors

# 機械可読JSON
diffai model1.safetensors model2.safetensors --output json

# 設定用YAML形式
diffai config1.yaml config2.yaml --output yaml

# Git対応unified diff
diffai data1.json data2.json --output unified
```

**出力フォーマット詳細**:

| フォーマット | 用途 | 機能 |
|--------|----------|----------|
| `cli` | 人間によるレビュー | 色付き出力、ML記号、階層表示 |
| `json` | 自動化 | 機械可読、構造化データ、API統合 |
| `yaml` | 設定 | 人間が読みやすい構造化フォーマット |
| `unified` | Git統合 | 従来のdiff形式、マージツール対応 |

### 比較オプション

#### `--epsilon <VALUE>`
浮動小数点数の比較における許容誤差を設定。

- **型**: 浮動小数点数
- **デフォルト**: 厳密比較（許容誤差なし）
- **範囲**: 任意の正の浮動小数点数

**例**:
```bash
# 微小な差異を無視（数値ノイズ）
diffai model1.safetensors model2.safetensors --epsilon 1e-6

# 量子化分析（大きな許容誤差）
diffai fp32_model.safetensors int8_model.safetensors --epsilon 0.01

# 訓練進捗（中程度の許容誤差）
diffai checkpoint1.pt checkpoint2.pt --epsilon 1e-4
```

**エプシロン値別の用途**:
| エプシロン | 用途 | 説明 |
|---------|----------|-------------|
| なし | 厳密比較 | 数値ノイズを含むすべての変更を検出 |
| `1e-8` | 高精度 | 科学計算、最小許容誤差 |
| `1e-6` | 標準ML | 通常のモデル訓練、浮動小数点エラーを無視 |
| `1e-4` | 訓練進捗 | 重要な学習変更に焦点 |
| `0.01` | 量子化 | 量子化モデルの精度損失を考慮 |
| `0.1` | アーキテクチャ重視 | 小さな重み変更を無視し、構造に焦点 |

### フィルタリング オプション

#### `--path <PATH>`
特定のパスで差分をフィルタリング。

- **型**: 文字列
- **形式**: ネストされたオブジェクトにはドット記法、配列には角括弧記法
- **デフォルト**: すべてのパスを表示

**例**:
```bash
# 分類器レイヤーのみに焦点
diffai model1.safetensors model2.safetensors --path "tensor.classifier"

# 特定の設定セクション
diffai config1.json config2.json --path "database.connection"

# IDによる配列要素
diffai users1.json users2.json --path "users[id=123]"
```

**パス構文**:
```
# オブジェクトのプロパティ
config.database.host

# 配列のインデックス
users[0].name

# IDによる配列要素（--array-id-key使用時）
users[id=123].email

# ネストされた構造
model.layers[0].weights.data

# MLモデルのテンソル
tensor.bert.encoder.layer.11.attention.self.query.weight
```

#### `--ignore-keys-regex <REGEX>`
正規表現にマッチするキーを無視。

- **型**: 正規表現文字列
- **デフォルト**: なし（すべてのキーを比較）

**例**:
```bash
# タイムスタンプフィールドを無視
diffai config1.json config2.json --ignore-keys-regex "^timestamp$"

# 複数のメタデータフィールドを無視
diffai model1.safetensors model2.safetensors --ignore-keys-regex "^(_metadata|timestamp|run_id)$"

# バージョン情報を無視
diffai package1.json package2.json --ignore-keys-regex "^version"

# 一時的または生成されたフィールドを無視
diffai data1.json data2.json --ignore-keys-regex "^(tmp_|generated_|_temp)"
```

**一般的な正規表現パターン**:
| パターン | マッチ | 用途 |
|---------|---------|----------|
| `^timestamp$` | 正確に "timestamp" キー | タイムスタンプを無視 |
| `^_.*` | アンダースコアで始まるキー | プライベートフィールドを無視 |
| `.*_temp$` | "_temp" で終わるキー | 一時データを無視 |
| `^(id\|uuid)$` | "id" または "uuid" キー | 識別子を無視 |
| `version` | "version" を含むキー | バージョンフィールドを無視 |

#### `--array-id-key <KEY>`
配列要素の識別に使用するキー。

- **型**: 文字列
- **デフォルト**: インデックスベースの比較
- **目的**: 配列要素の追跡を改善

**例**:
```bash
# IDフィールドでユーザーを追跡
diffai users1.json users2.json --array-id-key "id"

# UUIDでタスクを追跡
diffai tasks1.json tasks2.json --array-id-key "uuid" 

# 名前でモデルレイヤーを追跡
diffai config1.json config2.json --array-id-key "layer_name"
```

**array-id-keyなし（インデックスベース）**:
```json
// 変更がインデックス変更として表示
[0].name: "Alice" -> "Bob"
[1]: {"name": "Charlie", "age": 30} (追加)
```

**array-id-key="id"あり（IDベース）**:
```json
// 変更が意味的な意味で表示
[id=1].name: "Alice" -> "Bob"  
[id=3]: {"id": 3, "name": "Charlie", "age": 30} (追加)
```

### ディレクトリ オプション

#### `-r, --recursive`
ディレクトリを再帰的に比較。

- **型**: ブール値フラグ
- **デフォルト**: `false`（単一ファイル比較）

**例**:
```bash
# ディレクトリ内のすべてのファイルを比較
diffai config_dir1/ config_dir2/ --recursive

# フォーマット指定と組み合わせ
diffai experiments_v1/ experiments_v2/ --recursive --format json

# フィルタリングと組み合わせ
diffai models_old/ models_new/ --recursive --ignore-keys-regex "^timestamp$"
```

**ディレクトリ比較の動作**:
- 同じ相対パスのファイルを比較
- 各ディレクトリに固有のファイルを表示
- 各ファイルペアの形式を自動検出
- すべてのフィルタリングオプションを各比較に適用

### ヘルプとバージョン

#### `-h, --help`
ヘルプ情報を表示。

```bash
diffai --help
diffai -h
```

#### `-V, --version`
バージョン情報を表示。

```bash
diffai --version
diffai -V
```

## オプションの組み合わせ

### MLモデル解析

```bash
# 包括的なモデル比較
diffai model1.safetensors model2.safetensors \
  --epsilon 1e-6 \
  --output json \
  --path "tensor.classifier" \
  > analysis.json

# 訓練進捗解析
diffai checkpoint_epoch_10.pt checkpoint_epoch_50.pt \
  --epsilon 1e-4 \
  --ignore-keys-regex "^(optimizer_state|scheduler_state)" \
  --output yaml
```

### 設定管理

```bash
# 環境設定の比較
diffai config_dev.json config_prod.json \
  --ignore-keys-regex "^(timestamp|environment|debug)" \
  --output yaml \
  --path "database"

# 再帰的ディレクトリ解析
diffai config_v1/ config_v2/ \
  --recursive \
  --format json \
  --ignore-keys-regex "^_.*" \
  --output json > config_diff.json
```

### MLOps統合

```bash
# CI/CDモデル検証
diffai baseline_model.safetensors candidate_model.safetensors \
  --epsilon 1e-5 \
  --output json \
  --ignore-keys-regex "^(training_metadata|timestamp)" | \
  jq '.[] | select(.TensorShapeChanged)' > architecture_changes.json

# 実験追跡
diffai experiment_run_123.json experiment_run_124.json \
  --array-id-key "metric_name" \
  --ignore-keys-regex "^(start_time|end_time|run_id)" \
  --path "results.metrics"
```

## 終了コード

| コード | 意味 | 説明 |
|------|---------|-------------|
| `0` | 成功 | 比較が正常に完了 |
| `1` | 一般エラー | 不正な引数、ファイルが見つからない等 |
| `2` | 解析エラー | 入力ファイルの解析不可 |
| `3` | I/Oエラー | ファイル読み取り/書き込み権限の問題 |

## 例

### 基本的な使用法

```bash
# 単純なファイル比較
diffai config.json config_new.json

# 許容誤差を設定したモデル比較
diffai model.safetensors model_v2.safetensors --epsilon 1e-6

# ディレクトリ比較
diffai experiments_old/ experiments_new/ --recursive
```

### 高度なフィルタリング

```bash
# 特定のモデルレイヤーに焦点
diffai transformer_v1.safetensors transformer_v2.safetensors \
  --path "tensor.encoder.layer"

# メタデータとタイムスタンプを無視
diffai config1.json config2.json \
  --ignore-keys-regex "^(_meta|timestamp|created_at)$"

# IDで配列要素を追跡
diffai users_before.json users_after.json \
  --array-id-key "user_id"
```

### 出力フォーマット

```bash
# 自動化用JSON
diffai data1.json data2.json --output json | jq '.[] | .Added'

# ドキュメント用YAML
diffai config.yaml config_new.yaml --output yaml > changes.yaml

# git用unified diff
diffai old.json new.json --output unified | git apply
```

### ML特化ワークフロー

```bash
# ファインチューニング解析
diffai pretrained.safetensors finetuned.safetensors \
  --epsilon 1e-6 \
  --output json | \
  jq '[.[] | select(.TensorStatsChanged)] | length'

# 量子化影響
diffai fp32_model.safetensors int8_model.safetensors \
  --epsilon 0.01 \
  --path "tensor" \
  --output yaml

# 訓練チェックポイント比較
diffai checkpoint_1.pt checkpoint_10.pt \
  --epsilon 1e-4 \
  --ignore-keys-regex "^optimizer" \
  --output json > training_progress.json
```

## 注意事項と制限

### ファイルサイズ制限
- **推奨最大値**: ファイル1つあたり10GB
- **メモリ使用量**: 処理中は約ファイルサイズの2倍
- **大きなファイル**: `--path` フィルタリングの使用を検討

### フォーマットサポート
- **バイナリ形式**: PyTorchとSafetensorsのみ
- **テキスト形式**: すべてのエンコーディングタイプをサポート
- **圧縮ファイル**: 直接サポートなし（先に展開）

### パフォーマンス上の考慮事項
- **大きなモデル**: 適切なエプシロン値を使用
- **深い構造**: 大量のメモリが必要な場合あり
- **ディレクトリ比較**: ファイルを順次処理

### 正規表現の注意点
- Rustの正規表現構文を使用（Perlに類似）
- デフォルトで大文字小文字を区別
- 完全なUnicodeサポート
- 一般的なパターンに最適化されたパフォーマンス

---

**関連項目**: 
- [API リファレンス](api-reference.md) - Rustライブラリのドキュメント
- [MLモデル比較](../user-guide/ml-model-comparison.md) - 詳細なML使用ガイド
- [例](../user-guide/examples.md) - 実際の使用場面