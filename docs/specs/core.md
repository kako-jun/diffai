# diffai-core API 仕様書

バージョン: 0.3.17
最終更新: 2025-12-12
ステータス: **確定**

## 概要

`diffai-core` は AI/MLモデルファイルの意味的差分を検出するRustライブラリ。
CLIツール (`diffai`) および将来の言語バインディングの基盤。

## クレート構造

```
diffai-core/
├── src/
│   ├── lib.rs          # 公開API再エクスポート
│   ├── types.rs        # 型定義
│   ├── diff.rs         # 差分検出ロジック
│   ├── output.rs       # 出力フォーマット
│   ├── parsers/        # ファイルパーサー
│   │   ├── mod.rs
│   │   ├── pytorch.rs
│   │   ├── safetensors.rs
│   │   ├── numpy.rs
│   │   └── matlab.rs
│   └── ml_analysis/    # ML分析モジュール
│       ├── mod.rs
│       ├── learning_rate.rs
│       ├── gradient/
│       ├── convergence/
│       ├── quantization/
│       └── ...
└── tests/
```

---

## 公開API

### 主要関数

#### `diff_paths`

ファイルパスを指定して差分を検出する。

```rust
pub fn diff_paths(
    old_path: &str,
    new_path: &str,
    options: Option<&DiffOptions>,
) -> Result<Vec<DiffResult>>
```

**引数**:
- `old_path`: 比較元のパス
- `new_path`: 比較先のパス
- `options`: 比較オプション（省略可）

**戻り値**: `Result<Vec<DiffResult>>`

**動作**:
1. 拡張子からファイルフォーマットを自動検出
2. 各ファイルをパースして内部表現に変換
3. 差分を検出（基本差分 + テンソル差分）
4. PyTorch/Safetensorsの場合、ML分析を実行して結果に追加

**例**:
```rust
use diffai_core::{diff_paths, DiffOptions};

let results = diff_paths("model_v1.pt", "model_v2.pt", None)?;
for result in &results {
    println!("{:?}", result);
}
```

---

#### `diff`

パース済みの `serde_json::Value` を直接比較する。

```rust
pub fn diff(
    old: &Value,
    new: &Value,
    options: Option<&DiffOptions>,
) -> Result<Vec<DiffResult>>
```

**引数**:
- `old`: 比較元の値
- `new`: 比較先の値
- `options`: 比較オプション（省略可）

---

### パーサー関数

#### `detect_format_from_path`

ファイルパスの拡張子からフォーマットを検出する。

```rust
pub fn detect_format_from_path(path: &Path) -> Option<FileFormat>
```

**マッピング**:
| 拡張子 | FileFormat |
|--------|------------|
| `.pt`, `.pth` | `Pytorch` |
| `.safetensors` | `Safetensors` |
| `.npy`, `.npz` | `Numpy` |
| `.mat` | `Matlab` |
| その他 | `None` |

---

#### `parse_file_by_format`

指定フォーマットでファイルをパースする。

```rust
pub fn parse_file_by_format(path: &Path, format: FileFormat) -> Result<Value>
```

---

#### 個別パーサー

```rust
pub fn parse_pytorch_model(path: &Path) -> Result<Value>
pub fn parse_safetensors_model(path: &Path) -> Result<Value>
pub fn parse_numpy_file(path: &Path) -> Result<Value>
pub fn parse_matlab_file(path: &Path) -> Result<Value>
```

各パーサーはファイルを読み込み、以下の構造の `serde_json::Value` を返す：

**PyTorch/Safetensors**:
```json
{
  "tensors": {
    "layer1.weight": {
      "shape": [512, 256],
      "dtype": "float32",
      "data_summary": {
        "mean": 0.0012,
        "std": 0.0514,
        "min": -0.15,
        "max": 0.18
      }
    }
  },
  "metadata": {
    "format": "pytorch",
    "file_size": 12345678
  }
}
```

**NumPy/MATLAB**:
```json
{
  "arrays": {
    "variable_name": {
      "shape": [100, 100],
      "dtype": "float64",
      "data_summary": {...}
    }
  }
}
```

---

## 型定義

### `DiffResult`

差分の結果を表すenum。

```rust
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum DiffResult {
    // 基本差分（diffx-coreから継承）
    Added(String, Value),              // パス, 追加された値
    Removed(String, Value),            // パス, 削除された値
    Modified(String, Value, Value),    // パス, 旧値, 新値
    TypeChanged(String, Value, Value), // パス, 旧値, 新値

    // テンソル専用差分
    TensorShapeChanged(String, Vec<usize>, Vec<usize>),  // パス, 旧形状, 新形状
    TensorStatsChanged(String, TensorStats, TensorStats), // パス, 旧統計, 新統計
    TensorDataChanged(String, f64, f64),                  // パス, 旧mean, 新mean

    // ML分析結果
    ModelArchitectureChanged(String, String, String),     // パス, 旧arch, 新arch
    WeightSignificantChange(String, f64),                 // パス, 変化量
    ActivationFunctionChanged(String, String, String),    // パス, 旧fn, 新fn
    LearningRateChanged(String, f64, f64),                // パス, 旧LR, 新LR
    OptimizerChanged(String, String, String),             // パス, 旧opt, 新opt
    LossChange(String, f64, f64),                         // パス, 旧loss, 新loss
    AccuracyChange(String, f64, f64),                     // パス, 旧acc, 新acc
    ModelVersionChanged(String, String, String),          // パス, 旧ver, 新ver
}
```

**基本差分** (4種類):
- `Added`: キー/テンソルが追加された
- `Removed`: キー/テンソルが削除された
- `Modified`: 値が変更された（同一型）
- `TypeChanged`: 型が変更された

**テンソル専用差分** (3種類):
- `TensorShapeChanged`: テンソル形状が変更された
- `TensorStatsChanged`: テンソル統計（mean, std, min, max）が有意に変更された（1%以上の相対変化）
- `TensorDataChanged`: テンソルデータが変更された（統計変化が有意でない場合）

**ML分析結果** (8種類):
- `ModelArchitectureChanged`: モデル構造の変更を検出
- `WeightSignificantChange`: 重みの有意な変更を検出（0.05以上の変化、または50%以上の相対変化）
- `ActivationFunctionChanged`: 活性化関数の変更を検出
- `LearningRateChanged`: 学習率の変更を検出
- `OptimizerChanged`: オプティマイザの変更を検出
- `LossChange`: 損失値の変更を検出
- `AccuracyChange`: 精度の変更を検出
- `ModelVersionChanged`: モデル/フレームワークバージョンの変更を検出

---

### `TensorStats`

テンソルの統計情報。

```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TensorStats {
    pub mean: f64,      // 平均値
    pub std: f64,       // 標準偏差
    pub min: f64,       // 最小値
    pub max: f64,       // 最大値
    pub shape: Vec<usize>, // 形状
    pub dtype: String,  // データ型 ("float32", "float16", "int8" など)
    pub element_count: usize, // 要素数
}
```

**統計の有意な変化の判定**:
- `|old.mean - new.mean| / max(|old.mean|, 1e-8) > 0.01` (1%以上の相対変化)
- または `|old.std - new.std| / max(|old.std|, 1e-8) > 0.01`

---

### `DiffOptions`

差分検出のオプション。

```rust
#[derive(Debug, Clone, Default)]
pub struct DiffOptions {
    pub epsilon: Option<f64>,             // 数値許容誤差
    pub array_id_key: Option<String>,     // 配列要素のIDキー
    pub ignore_keys_regex: Option<Regex>, // 無視するキーの正規表現
    pub path_filter: Option<String>,      // パスフィルタ（部分一致）
    pub output_format: Option<OutputFormat>, // 出力フォーマット
}
```

---

### `OutputFormat`

出力フォーマット。

```rust
#[derive(Debug, Clone, Copy, Default)]
pub enum OutputFormat {
    #[default]
    Diffai,  // diffai形式（人間可読）
    Json,    // JSON配列
    Yaml,    // YAML配列
}
```

CLI で `--format diffai` (デフォルト), `--format json`, `--format yaml` で指定。

---

### `FileFormat`

入力ファイルフォーマット。

```rust
#[derive(Debug, Clone, Copy)]
pub enum FileFormat {
    Pytorch,     // .pt, .pth
    Safetensors, // .safetensors
    Numpy,       // .npy, .npz
    Matlab,      // .mat
}
```

---

## 差分検出アルゴリズム

### 基本差分検出

1. 両オブジェクトのキーを列挙
2. `ignore_keys_regex` にマッチするキーはスキップ
3. 片方にのみ存在するキー → Added/Removed
4. 両方に存在するキー → 値を再帰比較

### テンソル差分検出

テンソル構造（shape, dtype, data_summary を持つオブジェクト）を検出した場合：

1. 形状比較: `old.shape != new.shape` → `TensorShapeChanged`
2. 統計比較: 統計値が有意に変化 → `TensorStatsChanged`
3. 両方変化した場合は両方を出力

### 数値比較

**`epsilon` 指定時**:
```rust
(old_f - new_f).abs() <= epsilon
```

**未指定時**: 厳密比較（`old != new`）

### ML分析

PyTorch/Safetensorsファイル比較時に自動実行：

1. **学習率分析**: `learning_rate`, `lr` キーを検索
2. **オプティマイザ分析**: `optimizer`, `opt_state` キーを検索
3. **勾配分析**: 重みの統計変化から勾配健全性を推定
4. **収束分析**: 損失・精度の推移を分析

---

## 出力フォーマット

### Diffai形式（デフォルト）

```
+ added_key: value
- removed_key: value
~ modified_key: old_value -> new_value
! type_changed_key: old (OldType) -> new (NewType)
~ tensor.weight shape: [512, 256] -> [1024, 256]
~ tensor.weight stats: mean 0.001 -> 0.002
```

### JSON形式

```json
[
  {"Added": ["key", "value"]},
  {"TensorShapeChanged": ["layer.weight", [512, 256], [1024, 256]]},
  {"TensorStatsChanged": ["layer.weight", {"mean": 0.001, "std": 0.05, ...}, {"mean": 0.002, "std": 0.06, ...}]},
  {"LearningRateChanged": ["optimizer.lr", 0.001, 0.0001]},
  {"WeightSignificantChange": ["layer.weight", 0.15]}
]
```

---

## エラー処理

全ての公開関数は `anyhow::Result` を返す。

**主なエラー**:
- ファイルが存在しない: `"File not found: {path}"`
- 未対応フォーマット: `"Unsupported format for file: {path}"`
- パースエラー: `"Failed to parse {format} file: {details}"`
- 破損ファイル: `"Corrupted or invalid {format} file"`

---

## 使用例

### 基本的な使用

```rust
use diffai_core::{diff_paths, DiffResult};

fn main() -> anyhow::Result<()> {
    let results = diff_paths("model_v1.pt", "model_v2.pt", None)?;

    for result in &results {
        match result {
            DiffResult::TensorStatsChanged(path, old, new) => {
                println!("Tensor {} changed: mean {:.4} -> {:.4}",
                    path, old.mean, new.mean);
            }
            DiffResult::TensorShapeChanged(path, old, new) => {
                println!("Tensor {} reshaped: {:?} -> {:?}", path, old, new);
            }
            _ => println!("{:?}", result),
        }
    }

    Ok(())
}
```

### オプション付き

```rust
use diffai_core::{diff_paths, DiffOptions};
use regex::Regex;

let options = DiffOptions {
    epsilon: Some(0.001),
    ignore_keys_regex: Some(Regex::new("^optimizer").unwrap()),
    path_filter: Some("conv1".to_string()),
    ..Default::default()
};

let results = diff_paths("model1.pt", "model2.pt", Some(&options))?;
```

---

## 変更履歴

- 2025-12-12: v0.3.17 仕様更新（実装に合わせて修正）
  - DiffResult を 15 種類に拡張（TensorDataChanged, WeightSignificantChange, ActivationFunctionChanged, LossChange, AccuracyChange, ModelVersionChanged を追加）
  - OutputFormat: Text → Diffai にリネーム
  - 閾値を明記（TensorStatsChanged: 1%相対変化、WeightSignificantChange: 0.05絶対/50%相対）

- 2025-12-12: v0.3.17 仕様確定
  - diffx-core 0.6.x 統合
  - DiffResult enum の整理
  - テンソル差分検出仕様の明確化
  - ML分析の自動実行仕様
