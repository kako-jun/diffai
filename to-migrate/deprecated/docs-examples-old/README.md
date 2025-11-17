# diffai 動作実証

v0.3.4の実際の動作を示す簡潔な例集

## 基本機能

**AI/MLファイル比較** [`basic-model-diff.txt`](outputs/basic-model-diff.txt)
```bash
diffai model1.safetensors model2.safetensors
```

**詳細モード** [`verbose-mode.txt`](outputs/verbose-mode.txt) 
```bash
diffai model1.safetensors model2.safetensors --verbose
```

**JSON出力** [`output-json.txt`](outputs/output-json.txt)
```bash
diffai model1.safetensors model2.safetensors --output json
```

**注意**: 一般的な構造化データ（JSON、YAML、CSV等）の比較には [diffx](https://github.com/kako-jun/diffx) をご利用ください。

## ML機能

**MLモデル比較（30+機能自動）** [`ml-model-basic.txt`](outputs/ml-model-basic.txt)
```bash
diffai model1.safetensors model2.safetensors
```

**包括的分析（統計含む）** [`ml-model-stats.txt`](outputs/ml-model-stats.txt)
```bash
diffai model1.safetensors model2.safetensors
```

**包括的分析（アーキテクチャ含む）** [`ml-model-architecture.txt`](outputs/ml-model-architecture.txt)
```bash
diffai model1.safetensors model2.safetensors
```

## システム情報

**ヘルプ** [`help-output.txt`](outputs/help-output.txt)
**バージョン** [`version-info.txt`](outputs/version-info.txt)

---

全8例、すべて動作確認済み。diffaiは文書通りに機能します。