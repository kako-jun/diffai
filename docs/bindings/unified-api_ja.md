# diffai 統一API リファレンス

*diffai-python および diffai-js 言語バインディング API ドキュメント*

## 概要

diffai は AI/MLモデルファイルとテンソルを比較するための統一APIを提供します。PyTorch（.pt、.pth）、Safetensors、NumPy（.npy、.npz）、MATLAB（.mat）形式をサポートし、機械学習ユースケースに特化した分析を行います。

## メイン関数

### `diff(old, new, options)`

2つのAI/MLモデル構造またはテンソルを比較し、ML固有の分析を含む差分を返します。

#### パラメータ

- `old` (Value): 元の/古いモデルまたはテンソルデータ
- `new` (Value): 新しい/更新されたモデルまたはテンソルデータ
- `options` (DiffOptions, optional): 比較の設定オプション

#### 戻り値

- `Result<Vec<DiffResult>, Error>`: ML固有の変更を含む差分のベクター

#### 例

```rust
use diffai_core::{diff, DiffOptions};
use serde_json::json;

// モデルメタデータ比較の例
let old = json!({
    "model_name": "bert-base",
    "layers": {
        "encoder.layer.0.attention.self.query.weight": [768, 768],
        "encoder.layer.0.attention.self.query.bias": [768]
    }
});

let new = json!({
    "model_name": "bert-base-finetuned",
    "layers": {
        "encoder.layer.0.attention.self.query.weight": [768, 768],
        "encoder.layer.0.attention.self.query.bias": [768]
    }
});

let options = DiffOptions {
    ml_analysis_enabled: Some(true),
    scientific_precision: Some(true),
    ..Default::default()
};

let results = diff(&old, &new, Some(&options))?;
```

## オプション

### DiffOptions 構造体

```rust
pub struct DiffOptions {
    // 数値比較
    pub epsilon: Option<f64>,
    
    // 配列比較
    pub array_id_key: Option<String>,
    
    // フィルタリング
    pub ignore_keys_regex: Option<String>,
    pub path_filter: Option<String>,
    
    // 出力制御
    pub output_format: Option<OutputFormat>,
    pub show_unchanged: Option<bool>,
    pub show_types: Option<bool>,
    
    // メモリ最適化
    pub use_memory_optimization: Option<bool>,
    pub batch_size: Option<usize>,
    
    // diffai固有オプション
    pub ml_analysis_enabled: Option<bool>,
    pub tensor_comparison_mode: Option<String>,
    pub model_format: Option<String>,
    pub scientific_precision: Option<bool>,
    pub weight_threshold: Option<f64>,
    pub gradient_analysis: Option<bool>,
    pub statistical_summary: Option<bool>,
    pub verbose: Option<bool>,
    pub no_color: Option<bool>,
}
```

### オプション詳細

#### ML固有オプション

- **`ml_analysis_enabled`**: ML固有の分析を有効にする（重み変化、勾配フロー等）
  - デフォルト: `true`
  
- **`tensor_comparison_mode`**: テンソルの比較方法
  - オプション: `"element-wise"`, `"statistical"`, `"structural"`
  - デフォルト: `"element-wise"`
  
- **`model_format`**: 最適化された解析のための期待されるモデル形式
  - オプション: `"pytorch"`, `"safetensors"`, `"numpy"`, `"matlab"`, `"auto"`
  - デフォルト: `"auto"`
  
- **`scientific_precision`**: 数値出力に科学記法を使用
  - デフォルト: `false`
  
- **`weight_threshold`**: 報告する最小重み変化（ノイズのフィルタリングに役立つ）
  - デフォルト: `1e-6`
  
- **`gradient_analysis`**: 勾配関連テンソルを特別に分析
  - デフォルト: `false`
  
- **`statistical_summary`**: テンソル変化の統計サマリーを含める
  - デフォルト: `false`

#### 共通オプション（統一APIから継承）

- **`epsilon`**: 数値比較の許容誤差
  - デフォルト: `1e-9`（MLユースケースのより高い精度）
  
- **`ignore_keys_regex`**: 無視するキー（タイムスタンプ、ランダムシード等に有用）
  - 例: `"^(timestamp|random_seed|training_step)"`
  
- **`show_unchanged`**: 出力に変更されていないレイヤーを含める
  - デフォルト: `false`
  
- **`use_memory_optimization`**: 大型モデル（>1GB）で有効にする
  - デフォルト: ファイルサイズ>100MBで `true`

## 結果タイプ

### DiffResult 列挙型（ML拡張）

```rust
pub enum DiffResult {
    // 標準的な差分
    Added(String, Value),
    Removed(String, Value),
    Modified(String, Value, Value),
    TypeChanged(String, String, String),
    
    // ML固有の差分
    TensorShapeChanged(String, Vec<usize>, Vec<usize>),
    WeightSignificantChange(String, f64, Statistics),
    LayerAdded(String, LayerInfo),
    LayerRemoved(String, LayerInfo),
    ArchitectureChanged(String, String),
    PrecisionChanged(String, String, String),
}
```

### ML固有結果タイプ

- **`TensorShapeChanged(path, old_shape, new_shape)`**: テンソル次元の変更
- **`WeightSignificantChange(path, magnitude, stats)`**: 統計を含む重要な重み変化
- **`LayerAdded/Removed(path, info)`**: ニューラルネットワークレイヤーの変更
- **`ArchitectureChanged(old_arch, new_arch)`**: モデルアーキテクチャの変更
- **`PrecisionChanged(path, old_precision, new_precision)`**: データ型の変更（例：float32からfloat16）

### Statistics 構造体

```rust
pub struct Statistics {
    pub mean_change: f64,
    pub std_dev: f64,
    pub max_change: f64,
    pub min_change: f64,
    pub changed_elements: usize,
    pub total_elements: usize,
}
```

## 言語バインディング

### Python

```python
import diffai_python

# 基本的なモデル比較（ユーザーがモデルを自分でロード）
results = diffai_python.diff(old_model, new_model)

# ML固有オプション付き
results = diffai_python.diff(
    old_model,
    new_model,
    ml_analysis_enabled=True,
    tensor_comparison_mode="statistical",
    weight_threshold=1e-5,
    statistical_summary=True,
    scientific_precision=True
)

# ユーザーは適切なライブラリを使用してモデルをロードする（torch等）
# old_model = torch.load("model_epoch_1.pt")
# new_model = torch.load("model_epoch_10.pt")
# results = diffai_python.diff(old_model, new_model)
```

### TypeScript/JavaScript

```typescript
import { diff, DiffOptions } from 'diffai-js';

// 基本使用法 - ユーザーがモデルを自分でロード
const results = await diff(oldModel, newModel);

// ML固有オプション付き
const options: DiffOptions = {
    diffaiOptions: {
        mlAnalysisEnabled: true,
        tensorComparisonMode: 'statistical',
        scientificPrecision: true
    },
    epsilon: 1e-5,
    showTypes: true
};
const results = await diff(oldModel, newModel, options);
```

## 例

### PyTorchモデルの比較

```rust
use diffai_core::{diff, DiffOptions};

// ユーザーは適切なMLライブラリを使用してモデルをロードする
let old_model = /* PyTorch/candle/tchライブラリを使用してロード */;
let new_model = /* PyTorch/candle/tchライブラリを使用してロード */;

let options = DiffOptions {
    ml_analysis_enabled: Some(true),
    weight_threshold: Some(0.001),
    statistical_summary: Some(true),
    ..Default::default()
};

let results = diff(&old_model, &new_model, Some(&options))?;
```

### 訓練進捗の分析

```rust
let options = DiffOptions {
    tensor_comparison_mode: Some("statistical".to_string()),
    gradient_analysis: Some(true),
    show_unchanged: Some(false),
    ..Default::default()
};

// チェックポイントを比較して訓練進捗を確認
// ユーザーは適切なMLライブラリを使用してチェックポイントをロードする
let checkpoint_1 = /* PyTorch/candle/tchライブラリを使用してロード */;
let checkpoint_10 = /* PyTorch/candle/tchライブラリを使用してロード */;

let results = diff(&checkpoint_1, &checkpoint_10, Some(&options))?;
```

### 異なる精度の比較

```rust
let options = DiffOptions {
    epsilon: Some(1e-3), // 精度差の高い許容誤差
    scientific_precision: Some(true),
    ..Default::default()
};

// ユーザーは適切なMLライブラリを使用してモデルをロードする
let float32_model = /* PyTorch/candle/tchライブラリを使用してロード */;
let float16_model = /* PyTorch/candle/tchライブラリを使用してロード */;

let results = diff(&float32_model, &float16_model, Some(&options))?;
```

## パフォーマンスの考慮事項

- **大型モデル**: 1GBを超えるモデルには `use_memory_optimization` を有効にする
- **バッチ処理**: 利用可能メモリに基づいて `batch_size` を調整（デフォルト：1000テンソル）
- **統計モード**: 大型テンソルの高速比較には `tensor_comparison_mode: "statistical"` を使用
- **フィルタリング**: 特定のレイヤーやコンポーネントに焦点を当てるには `path_filter` を使用

## エラーハンドリング

ライブラリは以下に対して詳細なエラーを提供します：
- サポートされていないモデル形式
- 破損したモデルファイル
- メモリ割り当て失敗
- 互換性のないテンソル形状
- 精度損失の警告

## ベストプラクティス

1. **適切なイプシロンの設定**: 異なる精度のモデルを比較する際は高い値（1e-3）を使用
2. **重み閾値の使用**: 重要でない変化をフィルタリングして重要な差分に焦点を当てる
3. **統計サマリーの有効化**: 大型モデルでは、統計サマリーがより良い洞察を提供
4. **メモリ最適化**: 本番モデルでは常に有効にする
5. **レイヤーフィルタリング**: デバッグ中は特定のレイヤーを検査するために `path_filter` を使用