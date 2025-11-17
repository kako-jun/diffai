# APIリファレンス - diffai-core

AI/MLモデルの差分機能を提供する`diffai-core` Rustクレートの完全なAPIドキュメント。

## 概要

`diffai-core`クレートは、diffaiエコシステムの心臓部であり、AI/MLモデルファイルとテンソル向けの専門的な差分操作を提供します。他のRustアプリケーションに組み込んでML固有の比較機能を追加できます。

**統一API設計**：コアAPIは、自動包括分析を備えたすべての比較操作用の単一メイン関数`diff()`を公開します。

## インストール

`Cargo.toml`に`diffai-core`を追加：

```toml
[dependencies]
diffai-core = "0.2.0"
```

### 機能フラグ

```toml
[dependencies]
diffai-core = { version = "0.2.0", features = ["all-formats"] }
```

利用可能な機能：
- `pytorch`（デフォルト） - PyTorchモデルサポート
- `safetensors`（デフォルト） - Safetensors形式サポート  
- `numpy`（デフォルト） - NumPy配列サポート
- `matlab` - MATLABファイルサポート
- `all-formats` - すべての形式パーサーを有効化

## パブリックAPI

### コア型

#### `DiffResult`

2つのAI/MLモデルまたはテンソル間の単一の差異を表現。

```rust
#[derive(Debug, PartialEq, Serialize)]
pub enum DiffResult {
    // 標準的な差異
    Added(String, Value),
    Removed(String, Value),
    Modified(String, Value, Value),
    TypeChanged(String, String, String),
    
    // ML固有の差異
    TensorShapeChanged(String, Vec<usize>, Vec<usize>),
    WeightSignificantChange(String, f64, Statistics),
    LayerAdded(String, LayerInfo),
    LayerRemoved(String, LayerInfo),
    ArchitectureChanged(String, String),
    PrecisionChanged(String, String, String),
}
```

### コア関数

#### `diff()`

2つのAI/MLモデルまたはテンソル間の差異を計算するための主要関数。これはすべての比較操作の統一APIエントリポイントです。

```rust
pub fn diff(
    old: &Value,
    new: &Value,
    options: Option<&DiffOptions>,
) -> Result<Vec<DiffResult>, Error>
```

**パラメータ：**
- `old`：元の/ベースラインモデルまたはテンソルデータ
- `new`：新しい/更新されたモデルまたはテンソルデータ
- `options`：比較のオプション設定パラメータ

**戻り値：** 見つかったすべての差異を表す`Result<Vec<DiffResult>, Error>`

#### DiffOptions構造体

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
    
    // diffai固有のオプション
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

**例：**
```rust
use diffai_core::{diff, DiffOptions, DiffResult};
use serde_json::{json, Value};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let old_model = json!({
        "model_name": "bert-base",
        "layers": {
            "encoder.layer.0.attention.self.query.weight": [768, 768],
            "encoder.layer.0.attention.self.query.bias": [768]
        }
    });
    
    let new_model = json!({
        "model_name": "bert-base-finetuned",
        "layers": {
            "encoder.layer.0.attention.self.query.weight": [768, 768],
            "encoder.layer.0.attention.self.query.bias": [768]
        }
    });
    
    let options = DiffOptions {
        ml_analysis_enabled: Some(true),
        weight_threshold: Some(0.001),
        statistical_summary: Some(true),
        ..Default::default()
    };
    
    let differences = diff(&old_model, &new_model, Some(&options))?;
    
    for diff_result in differences {
        match diff_result {
            DiffResult::WeightSignificantChange(path, magnitude, stats) => {
                println!("{}で重要な重み変化: magnitude={}", path, magnitude);
                println!("統計: mean_change={}, std_dev={}", stats.mean_change, stats.std_dev);
            }
            _ => {}
        }
    }
    
    Ok(())
}
```

## 高度な使用法

### カスタム比較ロジック

#### ML固有の分析

機械学習固有の分析機能を有効化：

```rust
use diffai_core::{diff, DiffOptions};
use serde_json::json;

let old_checkpoint = json!({
    "epoch": 1,
    "model_state_dict": { /* モデル重み */ },
    "optimizer_state_dict": { /* オプティマイザ状態 */ }
});

let new_checkpoint = json!({
    "epoch": 10,
    "model_state_dict": { /* 更新された重み */ },
    "optimizer_state_dict": { /* 更新された状態 */ }
});

let options = DiffOptions {
    ml_analysis_enabled: Some(true),
    tensor_comparison_mode: Some("statistical".to_string()),
    gradient_analysis: Some(true),
    statistical_summary: Some(true),
    ..Default::default()
};

let differences = diff(&old_checkpoint, &new_checkpoint, Some(&options))?;
```

#### 精度認識比較

異なる数値精度のモデルを処理：

```rust
let options = DiffOptions {
    epsilon: Some(1e-3), // 精度差異のためのより高い許容値
    scientific_precision: Some(true),
    weight_threshold: Some(1e-4),
    ..Default::default()
};

let differences = diff(&float32_model, &float16_model, Some(&options))?;
```

### 異なるモデル形式での作業

#### モデルの読み込みと比較

```rust
use diffai_core::{diff, DiffOptions, DiffResult};
use serde_json::Value;
use std::fs;

// ユーザーは適切なMLライブラリを使用してモデルを読み込む必要があります
fn compare_pytorch_models(
    model1_path: &str,
    model2_path: &str,
    options: Option<&DiffOptions>
) -> Result<Vec<DiffResult>, Box<dyn std::error::Error>> {
    // 例：ユーザーはcandle、tch、または他のPyTorchバインディングを使用
    // 実際のモデルデータをserde_json::Valueに読み込む
    
    // これは単なるプレースホルダー - 実際の実装はMLライブラリを使用
    let old_content = fs::read_to_string(model1_path)?;
    let new_content = fs::read_to_string(model2_path)?;
    
    let old: Value = serde_json::from_str(&old_content)?;
    let new: Value = serde_json::from_str(&new_content)?;
    
    Ok(diff(&old, &new, options)?)
}
```

### 統合パターン

#### 訓練進捗分析

```rust
use diffai_core::{diff, DiffOptions, DiffResult};
use serde_json::Value;

struct TrainingAnalyzer {
    pub weight_changes: Vec<(String, f64)>,
    pub architecture_changes: Vec<String>,
    pub precision_changes: Vec<(String, String, String)>,
}

impl TrainingAnalyzer {
    pub fn analyze_checkpoints(
        &mut self,
        checkpoint1: &Value,
        checkpoint2: &Value
    ) -> Result<(), Box<dyn std::error::Error>> {
        let options = DiffOptions {
            ml_analysis_enabled: Some(true),
            tensor_comparison_mode: Some("statistical".to_string()),
            statistical_summary: Some(true),
            ..Default::default()
        };
        
        let differences = diff(checkpoint1, checkpoint2, Some(&options))?;
        
        for diff_result in differences {
            match diff_result {
                DiffResult::WeightSignificantChange(path, magnitude, _) => {
                    self.weight_changes.push((path, magnitude));
                }
                DiffResult::ArchitectureChanged(old_arch, new_arch) => {
                    self.architecture_changes.push(
                        format!("{} -> {}", old_arch, new_arch)
                    );
                }
                DiffResult::PrecisionChanged(path, old_prec, new_prec) => {
                    self.precision_changes.push((path, old_prec, new_prec));
                }
                _ => {}
            }
        }
        
        Ok(())
    }
}
```

#### 非同期モデル比較

```rust
use diffai_core::{diff, DiffOptions, DiffResult};
use serde_json::Value;
use tokio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tasks = vec![
        compare_models_async("model_v1.pt", "model_v2.pt"),
        compare_models_async("model_v2.pt", "model_v3.pt"),
    ];
    
    let results = futures::future::try_join_all(tasks).await?;
    
    for (i, diffs) in results.into_iter().enumerate() {
        println!("モデルペア {}: {} の差異", i + 1, diffs.len());
    }
    
    Ok(())
}

async fn compare_models_async(
    file1: &str,
    file2: &str
) -> Result<Vec<DiffResult>, Box<dyn std::error::Error>> {
    let content1 = tokio::fs::read_to_string(file1).await?;
    let content2 = tokio::fs::read_to_string(file2).await?;
    
    let result = tokio::task::spawn_blocking(move || {
        // 実際の使用では、MLライブラリを使用してモデルファイルを解析
        let old: Value = serde_json::from_str(&content1)?;
        let new: Value = serde_json::from_str(&content2)?;
        
        let options = DiffOptions {
            ml_analysis_enabled: Some(true),
            use_memory_optimization: Some(true),
            ..Default::default()
        };
        
        diff(&old, &new, Some(&options))
    }).await??;
    
    Ok(result)
}
```

## エラーハンドリング

### エラー型

ライブラリはエラーハンドリングに`anyhow::Error`を使用：

```rust
use diffai_core::{diff, DiffOptions};
use anyhow::Result;

fn handle_model_errors() -> Result<()> {
    // ... モデルを読み込み ...
    
    match diff(&old_model, &new_model, None) {
        Ok(differences) => {
            println!("{}の差異を発見", differences.len());
        }
        Err(e) => {
            eprintln!("モデル比較エラー: {}", e);
            
            // 特定のエラー型をチェック
            if e.to_string().contains("memory") {
                eprintln!("メモリ最適化の有効化を検討してください");
            }
        }
    }
    
    Ok(())
}
```

## パフォーマンスの考慮事項

### メモリ使用量

大きなモデルの場合：

```rust
use diffai_core::{diff, DiffOptions, DiffResult};

fn process_large_models(
    old: &Value,
    new: &Value
) -> Result<Vec<DiffResult>, Box<dyn std::error::Error>> {
    let options = DiffOptions {
        use_memory_optimization: Some(true),
        batch_size: Some(500), // 大きなテンソル用の小さなバッチ
        tensor_comparison_mode: Some("statistical".to_string()),
        ..Default::default()
    };
    
    Ok(diff(old, new, Some(&options))?)
}
```

### 最適化のヒント

1. **メモリ最適化を使用** - 1GB以上のモデルに
2. **適切なイプシロンを設定** - 精度要件に応じて
3. **統計モードを使用** - 大きなテンソルのより高速な比較
4. **パスをフィルタ** - 特定のレイヤーやコンポーネントに焦点
5. **バッチサイズを調整** - 利用可能なメモリに基づいて

## テスト

### ユニットテスト

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    
    #[test]
    fn test_weight_change_detection() {
        let old = json!({
            "weights": {
                "layer1": [1.0, 2.0, 3.0],
                "layer2": [4.0, 5.0, 6.0]
            }
        });
        
        let new = json!({
            "weights": {
                "layer1": [1.1, 2.1, 3.1],
                "layer2": [4.0, 5.0, 6.0]
            }
        });
        
        let options = DiffOptions {
            ml_analysis_enabled: Some(true),
            weight_threshold: Some(0.05),
            ..Default::default()
        };
        
        let diffs = diff(&old, &new, Some(&options)).unwrap();
        
        // layer1で重要な変化を検出する必要がある
        assert!(diffs.iter().any(|d| matches!(d, 
            DiffResult::WeightSignificantChange(path, _, _) if path.contains("layer1")
        )));
    }
}
```

## 自動ML分析

**設定より規約**：diffai-coreは、PyTorch（.pt/.pth）またはSafetensors（.safetensors）ファイルを検出すると、11の専門的なML分析機能を自動的に実行します：

1. **learning_rate_analysis** - 学習率追跡と動態
2. **optimizer_comparison** - オプティマイザ状態比較
3. **loss_tracking** - 損失関数進化分析
4. **accuracy_tracking** - パフォーマンス指標監視
5. **model_version_analysis** - バージョンとチェックポイント検出
6. **gradient_analysis** - 勾配フローと安定性分析
7. **quantization_analysis** - 混合精度検出（FP32/FP16/INT8/INT4）
8. **convergence_analysis** - 学習曲線と収束パターン
9. **activation_analysis** - 活性化関数使用分析
10. **attention_analysis** - トランスフォーマーと注意機構
11. **ensemble_analysis** - アンサンブルモデル検出

**トリガー条件：**
- **PyTorch/Safetensors**：11の分析すべてが自動実行
- **NumPy/MATLAB**：基本的なテンソル統計のみ
- **その他の形式**：標準的な構造比較

## バージョン互換性

- **0.3.16**：自動ML分析付きの現在の安定版
- **基盤**：実証済みの差分信頼性のためのdiffx-core v0.6.x
- **最小Rustバージョン**：1.70.0
- **依存関係**：現在のバージョンについては`Cargo.toml`を参照

## 関連項目

- コマンドライン使用については[CLIリファレンス](cli-reference_ja.md)
- 実用例については[MLモデル比較ガイド](../user-guide/ml-model-comparison_ja.md)
- 言語バインディングについては[統一APIリファレンス](../bindings/unified-api_ja.md)