# diffai v0.3.16 実際の出力例

**生成日時**: 2025年7月31日  
**diffaiバージョン**: v0.3.16  
**テスト環境**: Rust CLI実行結果

## 📊 **実際のML分析機能**

現在のdiffai v0.3.16が**実際に提供している**機能：

### ✅ **PyTorchモデル分析**
- `binary_size`: ファイルサイズ比較
- `file_size`: 実際のファイルサイズ
- `detected_components`: コンポーネント検出（weight_params, bias_params, convolution, batch_norm等）
- `estimated_layers`: レイヤー数推定
- `structure_fingerprint`: 構造ハッシュ
- `pickle_protocol`: Pickleプロトコルバージョン

### ✅ **SafeTensorsモデル分析**
- テンソル構造比較（Added/Removed/Modified）
- テンソル形状・dtype情報
- レイヤー単位での差分検出

## 📋 **出力ファイル**

1. **`basic_model_comparison.txt`** - PyTorchモデル基本比較
2. **`diffai_format_output.txt`** - diffai形式出力
3. **`safetensors_comparison.txt`** - SafeTensors形式比較

## 🚨 **重要な発見**

**ドキュメントで宣伝されている「11個の自動ML分析機能」は存在しません**：

❌ **存在しない機能**（ドキュメントの虚偽宣伝）：
- `learning_rate_analysis`
- `convergence_analysis` 
- `attention_analysis`
- `gradient_flow_tracking`
- `optimizer_comparison`
- `batch_normalization_analysis`
- `regularization_impact`
- `activation_pattern_analysis`
- `weight_distribution_analysis`
- `training_stability_metrics`
- `model_complexity_assessment`

✅ **実際の機能**：
- 基本的なモデル構造比較
- テンソル形状・サイズ分析
- コンポーネント検出
- ファイル形式固有の情報抽出

**結論**: diffai v0.3.16は基本的なML構造比較ツールであり、高度なML分析機能は未実装です。