# diffai ML専用オプション制約条件・相互関係分析

## 📋 オプション一覧と制約

### 1. `--ml-analysis` (基盤オプション)
- **制約**: 無し（全形式で有効）
- **依存**: 他の全ML機能の前提条件
- **自動化**: 常に有効

### 2. `--tensor-mode <TENSOR_MODE>` [shape, data, both]
- **制約**: 無し（全形式で有効）
- **排他**: 3つの値は相互排他
- **自動化**: "both"で統一（最大情報量）

### 3. `--model-format <MODEL_FORMAT>` [pytorch, safetensors, numpy, matlab, auto]
- **制約**: 無し
- **排他**: 指定時は自動検出を無効化
- **自動化**: auto（自動検出優先）

### 4. `--scientific-precision`
- **制約**: 無し（全形式で有効）
- **用途**: 数値出力の精度制御
- **自動化**: 常に有効（ML用途では高精度が必要）

### 5. `--weight-threshold <WEIGHT_THRESHOLD>`
- **制約**: 無し（全形式で有効）
- **用途**: 重み変化検出の閾値
- **自動化**: 0.01（経験的妥当値）

### 6. `--activation-analysis`
- **制約**: ⚠️ **PyTorch/Safetensorsのみ**
- **理由**: 活性化関数情報が必要
- **依存**: ml-analysis必須
- **自動化**: フォーマット依存

### 7. `--learning-rate-tracking`
- **制約**: ⚠️ **PyTorch/Safetensorsのみ**
- **理由**: 学習メタデータが必要
- **依存**: ml-analysis必須
- **自動化**: フォーマット依存

### 8. `--optimizer-comparison`
- **制約**: ⚠️ **PyTorchのみ**
- **理由**: オプティマイザ状態が必要
- **依存**: ml-analysis必須
- **自動化**: フォーマット依存

### 9. `--loss-tracking`
- **制約**: ⚠️ **PyTorch/Safetensorsのみ**
- **理由**: 損失関数情報が必要
- **依存**: ml-analysis必須
- **自動化**: フォーマット依存

### 10. `--accuracy-tracking`
- **制約**: ⚠️ **PyTorchのみ**
- **理由**: モデル評価情報が必要
- **依存**: ml-analysis必須
- **自動化**: フォーマット依存

### 11. `--model-version-check`
- **制約**: 無し（全形式で有効）
- **用途**: メタデータ比較
- **自動化**: 常に有効

## 🔗 フォーマット依存マトリクス

| オプション | PyTorch | Safetensors | NumPy | MATLAB |
|------------|---------|-------------|--------|--------|
| ml-analysis | ✅ | ✅ | ✅ | ✅ |
| tensor-mode | ✅ | ✅ | ✅ | ✅ |
| model-format | ✅ | ✅ | ✅ | ✅ |
| scientific-precision | ✅ | ✅ | ✅ | ✅ |
| weight-threshold | ✅ | ✅ | ✅ | ✅ |
| activation-analysis | ✅ | ✅ | ❌ | ❌ |
| learning-rate-tracking | ✅ | ✅ | ❌ | ❌ |
| optimizer-comparison | ✅ | ❌ | ❌ | ❌ |
| loss-tracking | ✅ | ✅ | ❌ | ❌ |
| accuracy-tracking | ✅ | ❌ | ❌ | ❌ |
| model-version-check | ✅ | ✅ | ✅ | ✅ |

## 🎯 自動発動ルール設計

### Universal（全形式共通）
```rust
ml_analysis_enabled: true,
tensor_comparison_mode: "both",
model_format: None, // auto-detect
scientific_precision: true,
weight_threshold: 0.01,
model_version_check: true,
```

### PyTorch (.pt/.pth)
```rust
// + Universal
activation_analysis: true,
learning_rate_tracking: true,
optimizer_comparison: true,
loss_tracking: true,
accuracy_tracking: true,
```

### Safetensors (.safetensors)
```rust
// + Universal
activation_analysis: true,
learning_rate_tracking: true,
loss_tracking: true,
// optimizer_comparison: false, // 状態情報なし
// accuracy_tracking: false,    // 評価情報なし
```

### NumPy (.npy/.npz)
```rust
// Universal のみ
// 学習関連機能は全て無効
```

### MATLAB (.mat)
```rust
// Universal のみ
// 学習関連機能は全て無効
```

## ⚠️ 修正が必要な現在の実装

現在のCLI実装では全機能を無差別に有効化している：

```rust
// 問題のある現在の実装
let diffai_options = Some(DiffaiSpecificOptions {
    ml_analysis_enabled: Some(true),
    tensor_comparison_mode: Some("both".to_string()),
    model_format: None,
    scientific_precision: Some(true),
    weight_threshold: Some(0.01),
    activation_analysis: Some(true), // ❌ NumPy/MATLABで無意味
    learning_rate_tracking: Some(true), // ❌ NumPy/MATLABで無意味
    optimizer_comparison: Some(true), // ❌ Safetensors/NumPy/MATLABで無意味
    loss_tracking: Some(true), // ❌ NumPy/MATLABで無意味
    accuracy_tracking: Some(true), // ❌ Safetensors/NumPy/MATLABで無意味
    model_version_check: Some(true),
});
```

## 🎯 改善案：フォーマット適応型設定

```rust
fn build_format_aware_diffai_options(format: FileFormat) -> DiffaiSpecificOptions {
    let mut options = DiffaiSpecificOptions {
        // Universal options
        ml_analysis_enabled: Some(true),
        tensor_comparison_mode: Some("both".to_string()),
        model_format: None,
        scientific_precision: Some(true),
        weight_threshold: Some(0.01),
        model_version_check: Some(true),
        
        // Format-specific defaults
        activation_analysis: None,
        learning_rate_tracking: None,
        optimizer_comparison: None,
        loss_tracking: None,
        accuracy_tracking: None,
    };
    
    match format {
        FileFormat::PyTorch => {
            options.activation_analysis = Some(true);
            options.learning_rate_tracking = Some(true);
            options.optimizer_comparison = Some(true);
            options.loss_tracking = Some(true);
            options.accuracy_tracking = Some(true);
        }
        FileFormat::Safetensors => {
            options.activation_analysis = Some(true);
            options.learning_rate_tracking = Some(true);
            options.loss_tracking = Some(true);
        }
        FileFormat::NumPy | FileFormat::Matlab => {
            // Universal options only
        }
    }
    
    options
}
```