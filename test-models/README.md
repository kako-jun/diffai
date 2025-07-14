# Real Models Test

このディレクトリは、diffaiの実際のMLモデルでの動作検証用です。

## 🎯 目的

- HuggingFaceから実際の小さなMLモデルをダウンロード
- diffaiの高度な分析機能を実際のモデルで検証
- PyTorchとSafetensors両方の形式でテスト

## 🚀 使用方法

### 1. 依存関係のインストール

```bash
cd real_models_test/
uv sync
```

### 2. モデルのダウンロード

```bash
uv run python download_models.py
```

### 3. diffaiでのテスト

```bash
# Safetensorsモデル間の比較
diffai distilbert_base/model.safetensors gpt2_small/model.safetensors

# 高度な分析機能
diffai distilbert_base/model.safetensors gpt2_small/model.safetensors \
  --learning-progress --convergence-analysis --architecture-comparison

# JSON出力でMLOpsツール連携
diffai distilbert_base/model.safetensors gpt2_small/model.safetensors \
  --deployment-readiness --output json
```

## 📦 ダウンロードされるモデル

| モデル | 形式 | サイズ | 説明 |
|--------|------|--------|------|
| DistilBERT-base | Safetensors | ~260MB | BERT系の小さなモデル |
| DialoGPT-small | PyTorch | ~117MB | 対話型GPT（PyTorch形式テスト用） |
| GPT-2 small | Safetensors | ~500MB | OpenAIのGPT-2小型版 |
| Tiny GPT-2 | PyTorch | ~11MB | 極小テスト用GPT-2 |
| DistilGPT-2 | Safetensors | ~350MB | GPT-2の蒸留版 |

## 🔧 SSL証明書問題への対応

このスクリプトはSSL証明書の検証を無効化しています：

```python
os.environ['HF_HUB_DISABLE_SSL_VERIFY'] = '1'
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
```

**注意**: 本番環境では適切なSSL証明書設定を使用してください。

## 📋 テストデータ管理ポリシー

- **このディレクトリは.gitignoreに含まれています**
- ダウンロードしたモデルファイルはリポジトリにコミットされません
- 必要に応じてスクリプトを実行してモデルを再ダウンロード

## 🎯 テスト用途

1. **基本機能テスト**: モデル形式の読み込み確認
2. **統計分析テスト**: 実際のテンソル統計計算
3. **高度機能テスト**: 学習進捗・収束分析等
4. **パフォーマンステスト**: 大きなファイルでの動作確認
5. **MLOps統合テスト**: JSON/YAML出力でのツール連携

## 💡 活用例

### 研究開発での使用
```bash
# 異なるアーキテクチャの比較
diffai distilbert_base/model.safetensors dialogpt_small/pytorch_model.bin \
  --architecture-comparison --param-efficiency-analysis
```

### CI/CD統合での使用
```bash
# デプロイメント準備確認
diffai baseline_model.safetensors candidate_model.safetensors \
  --regression-test --deployment-readiness --alert-on-degradation
```

### 実験記録での使用
```bash
# Markdown形式でのレポート生成
diffai model_before.safetensors model_after.safetensors \
  --generate-report --markdown-output --include-charts
```