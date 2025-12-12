# diffai CLI 仕様書

バージョン: 0.3.17
最終更新: 2025-12-12
ステータス: **確定**

## 概要

diffai は AI/MLモデルファイル（PyTorch, Safetensors, NumPy, MATLAB）の差分を検出するCLIツール。
バイナリファイルから**意味的な差分**を抽出し、テンソル統計と自動ML分析を提供する。

## 基本構文

```
diffai [OPTIONS] <FILE1> <FILE2>
diffai [OPTIONS] -r <DIR1> <DIR2>
```

## 終了コード

| コード | 意味 |
|--------|------|
| 0 | 差分なし |
| 1 | 差分あり |
| 2 | 引数エラー、パースエラー等 |
| 3 | ファイルI/Oエラー |

## 引数

### FILE1, FILE2

- 両方必須
- ファイルパスまたはディレクトリパス
- ディレクトリの場合は `-r` が必須

---

## オプション

### 入力制御

#### `-f, --format <FORMAT>`

入力ファイルのフォーマットを明示的に指定する。

| 値 | 対応拡張子 |
|----|-----------|
| `pytorch` | .pt, .pth |
| `safetensors` | .safetensors |
| `numpy` | .npy, .npz |
| `matlab` | .mat |

**デフォルト**: 拡張子から自動検出

**動作**:
- 指定時: 両ファイルを指定フォーマットとしてパース
- 未指定時: 各ファイルの拡張子から判定

---

### 出力制御

#### `-o, --output <FORMAT>`

出力フォーマットを指定する。

| 値 | 説明 |
|----|------|
| `text` | 人間可読な差分表示（デフォルト） |
| `json` | JSON配列形式 |
| `yaml` | YAML配列形式 |

**text形式の記号**:
- `+` 追加: テンソル/パラメータが追加された
- `-` 削除: テンソル/パラメータが削除された
- `~` 変更: テンソル統計が変更された

**出力例（text形式）**:
```
learning_rate_analysis: old=0.001, new=0.0015, change=+50.0%
gradient_analysis: flow_health=healthy, norm=0.021
~ fc1.weight: mean=-0.0002->-0.0001, std=0.0514->0.0716
~ fc2.weight: mean=-0.0008->-0.0018, std=0.0719->0.0883
```

**JSON出力例**:
```json
[
  {"LearningRateChanged": ["learning_rate", 0.001, 0.0015]},
  {"TensorStatsChanged": ["fc1.weight", {"mean": -0.0002, "std": 0.0514}, {"mean": -0.0001, "std": 0.0716}]}
]
```

---

#### `--no-color`

出力の色付けを無効にする。

**動作**: ANSIエスケープシーケンスを出力しない

---

#### `-q, --quiet`

出力を抑制し、終了コードのみを返す。

**動作**:
- stdout に何も出力しない
- 終了コードで結果を判定

**例**:
```bash
diffai -q model1.pt model2.pt && echo "同じ" || echo "異なる"
```

---

#### `--brief`

差分の有無のみを報告する。

**動作**:
- 差分あり: `Files FILE1 and FILE2 differ` を出力
- 差分なし: 何も出力しない

---

#### `-v, --verbose`

詳細なML分析情報を表示する。

**出力内容**:
```
learning_rate_analysis:
  old: 0.001
  new: 0.0015
  change: +50.0%
  trend: increasing
gradient_analysis:
  flow_health: healthy
  norm: 0.021
  variance_change: +15.3%
...
```

---

### 比較オプション

#### `--epsilon <EPSILON>`

数値比較時の許容誤差。

**動作**: `|a - b| <= epsilon` なら同値とみなす

**適用対象**: テンソル統計値（mean, std, min, max）

**例**:
```bash
diffai --epsilon 0.001 model1.pt model2.pt
# mean の差が 0.001 以下なら同値
```

---

#### `--ignore-keys-regex <PATTERN>`

指定した正規表現にマッチするキー/レイヤー名を無視する。

**動作**:
- マッチしたキーとその値を比較対象から除外
- ネストしたオブジェクトにも再帰的に適用

**例**:
```bash
diffai --ignore-keys-regex "^optimizer" model1.pt model2.pt
# optimizer関連のパラメータを無視
```

---

#### `--array-id-key <KEY>`

配列要素をインデックスではなく指定キーで対応付ける。

**動作**:
- 配列内のオブジェクトを指定キーの値でマッチング
- 順序の変更は差分として検出しない

---

#### `--path <PATH>`

指定文字列を含むパスの差分のみを表示する。

**動作**:
- 全ての差分を計算した後にフィルタリング
- パスに指定文字列が含まれる場合のみ出力

**例**:
```bash
diffai --path "conv1" model1.pt model2.pt
# conv1 レイヤーの差分のみ表示
```

---

### ディレクトリ比較

#### `-r, --recursive`

ディレクトリを再帰的に比較する。

**必須条件**: 両引数がディレクトリの場合

**動作**:
1. 両ディレクトリのモデルファイルを再帰的に収集
2. 相対パスでマッチング
3. 片方にのみ存在 → Added/Removed
4. 両方に存在 → ファイル内容を比較

---

### ユーティリティ

#### `-h, --help`

ヘルプを表示。

#### `-V, --version`

バージョンを表示。形式: `diffai X.Y.Z`

---

## 対応フォーマット

| フォーマット | 拡張子 | 機能 |
|--------------|--------|------|
| PyTorch | .pt, .pth | フルML分析 + テンソル統計 |
| Safetensors | .safetensors | フルML分析 + テンソル統計 |
| NumPy | .npy, .npz | テンソル統計 |
| MATLAB | .mat | テンソル統計 |

**PyTorch/Safetensors固有機能**:
- 自動ML分析（11種類）
- オプティマイザ状態の比較
- 学習履歴の追跡

---

## 差分の種類

### 基本差分

| 種類 | 記号 | 説明 |
|------|------|------|
| Added | `+` | テンソル/パラメータが追加された |
| Removed | `-` | テンソル/パラメータが削除された |
| Modified | `~` | 値が変更された |
| TypeChanged | `!` | データ型が変更された |

### ML固有差分

| 種類 | 説明 |
|------|------|
| TensorStatsChanged | テンソル統計（mean, std, min, max）が変更 |
| TensorShapeChanged | テンソル形状が変更 |
| LearningRateChanged | 学習率が変更 |
| OptimizerStateChanged | オプティマイザ状態が変更 |
| ConvergenceChanged | 収束状態が変更 |
| GradientFlowChanged | 勾配フローが変更 |
| QuantizationChanged | 量子化設定が変更 |

---

## 自動ML分析

PyTorch/Safetensorsファイル比較時、以下の分析を自動実行：

1. **学習率分析** - 学習率の変化とトレンド検出
2. **オプティマイザ比較** - オプティマイザ状態の変化
3. **損失追跡** - 損失値の推移
4. **精度追跡** - 精度メトリクスの推移
5. **モデルバージョン分析** - アーキテクチャ変更検出
6. **勾配分析** - 勾配フローの健全性
7. **量子化分析** - 精度と圧縮の変化
8. **収束分析** - 学習の収束状態
9. **活性化関数分析** - 活性化パターンの変化
10. **アテンション分析** - Transformerアテンションの変化
11. **アンサンブル分析** - アンサンブルモデルの変化

---

## CI/CDでの使用例

```bash
# モデル変更検知
if ! diffai production.pt candidate.pt --quiet; then
  echo "モデルが変更されています"
  diffai production.pt candidate.pt --output json > changes.json
fi

# 特定レイヤーのみ確認
diffai model1.pt model2.pt --path "classifier"

# 学習率の大きな変化を検出
diffai checkpoint1.pt checkpoint2.pt --epsilon 0.0001
```

---

## 変更履歴

- 2025-12-12: v0.3.17 仕様確定
  - diffx-core 0.6.x への依存更新
  - Python/JSバインディングを別リポジトリに分離
  - 仕様書の精緻化
