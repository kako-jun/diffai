# diffai 動作実証

v0.3.4の実際の動作を示す簡潔な例集

## 基本機能

**JSON比較** [`basic-json-diff.txt`](outputs/basic-json-diff.txt)
```bash
diffai config_v1.json config_v2.json
```

**詳細モード** [`verbose-mode.txt`](outputs/verbose-mode.txt) 
```bash
diffai config_v1.json config_v2.json --verbose
```

**JSON出力** [`output-json.txt`](outputs/output-json.txt)
```bash
diffai config_v1.json config_v2.json --output json
```

## ML機能

**MLモデル比較** [`ml-model-basic.txt`](outputs/ml-model-basic.txt)
```bash
diffai model1.safetensors model2.safetensors
```

**統計分析** [`ml-model-stats.txt`](outputs/ml-model-stats.txt)
```bash
diffai model1.safetensors model2.safetensors --stats
```

**アーキテクチャ解析** [`ml-model-architecture.txt`](outputs/ml-model-architecture.txt)
```bash
diffai model1.safetensors model2.safetensors --architecture-comparison
```

## システム情報

**ヘルプ** [`help-output.txt`](outputs/help-output.txt)
**バージョン** [`version-info.txt`](outputs/version-info.txt)

---

全8例、すべて動作確認済み。diffaiは文書通りに機能します。