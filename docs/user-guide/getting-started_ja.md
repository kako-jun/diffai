# diffaiを始める

このガイドでは、`diffai`を素早く効果的に使い始める方法を詳しく解説します。

## diffaiとは

`diffai`は、AI技術を活用した次世代の差分比較ツールです。従来の単純な差分表示を超えて、機械学習による意味理解と高度な分析機能を組み合わせ、構造化データファイルの変更に対する深い洞察を提供します。

### 主な特徴

- **AI駆動の分析**: 機械学習により変更パターンや異常を自動検出
- **意味的理解**: データ変更の文脈と意味を理解して分析
- **幅広いフォーマット対応**: JSON、YAML、TOML、XML、CSV等に対応
- **インテリジェントな洞察**: 変更パターンをML技術で深く分析
- **高度な統計機能**: データ分布とトレンドの詳細な統計分析

## 事前準備

始める前に、`diffai`がインストールされていることを確認してください。詳しいインストール方法は[インストールガイド](installation_ja.md)をご覧ください。

クイックインストール:
```bash
cargo install diffai
```

## 基本的な使い方

### シンプルなファイル比較

最も基本的な使い方は、2つのファイルを比較することです：

```bash
diffai file1.json file2.json
```

このコマンドは変更点を色分けして表示します：
- **緑**: 追加された内容
- **赤**: 削除された内容
- **黄**: 変更された内容

### 出力例

```bash
$ diffai config_old.json config_new.json
  ~ version: "1.0.0" -> "1.1.0"
  + features.new_feature: true
  - features.deprecated_feature
  ~ settings.timeout: 30 -> 60
```

## 実践的な例

### 設定ファイルの比較

開発環境と本番環境の設定を比較：

```bash
diffai config/development.json config/production.json
```

### APIレスポンスの検証

APIのバージョン間の変更を確認：

```bash
diffai api/v1/response.json api/v2/response.json --output json
```

### データ移行の確認

データベースエクスポートの整合性チェック：

```bash
diffai old_export.csv new_export.csv --array-id-key "user_id"
```

## 高度な使用方法

### 特定のキーを無視

タイムスタンプなど、比較したくないフィールドを除外：

```bash
diffai file1.json file2.json --ignore-keys-regex "timestamp|updated_at"
```

### 浮動小数点の許容誤差

数値の微小な差異を許容：

```bash
diffai data1.json data2.json --epsilon 0.001
```

### 配列要素の追跡

IDベースで配列要素を比較：

```bash
diffai users_old.json users_new.json --array-id-key "id"
```

## 出力形式

### JSON形式での出力

プログラムで処理しやすい形式：

```bash
diffai file1.json file2.json --output json > diff_result.json
```

### YAML形式での出力

人間が読みやすい構造化形式：

```bash
diffai file1.json file2.json --output yaml
```

## MLモデル比較（特別機能）

### Safetensorsモデルの比較

```bash
diffai model_v1.safetensors model_v2.safetensors
```

自動的に以下の分析が実行されます：
- テンソル統計の比較
- 量子化分析
- アーキテクチャ比較
- 異常検出
- メモリ効率分析

### PyTorchモデルの比較

```bash
diffai checkpoint_epoch_10.pt checkpoint_epoch_20.pt
```

## トラブルシューティング

### よくある問題

**Q: ファイルが大きすぎてメモリ不足になる**
A: `--streaming`オプションを使用してください：
```bash
diffai large_file1.json large_file2.json --streaming
```

**Q: 日本語を含むファイルで文字化けする**
A: UTF-8エンコーディングを確認してください。diffaiはUTF-8を標準でサポートしています。

**Q: 特定の差分だけを見たい**
A: `--path`オプションで特定のパスに絞り込めます：
```bash
diffai file1.json file2.json --path "config.database"
```

## 次のステップ

基本的な使い方を理解したら、以下のガイドでさらに詳しく学びましょう：

1. [基本使用ガイド](basic-usage_ja.md) - より詳細な使用方法
2. [MLモデル比較ガイド](ml-model-comparison_ja.md) - 機械学習モデルの高度な比較
3. [科学データ分析](scientific-data_ja.md) - NumPyやMATLABファイルの比較
4. [CLIリファレンス](../reference/cli-reference_ja.md) - 全コマンドオプションの詳細

## ヒントとベストプラクティス

### 1. 設定ファイルの活用

頻繁に使用するオプションは`.diffai.toml`に保存：

```toml
[default]
ignore_keys_regex = "timestamp|_id"
epsilon = 0.001
output = "json"
```

### 2. CI/CDへの統合

GitHub Actionsでの使用例：

```yaml
- name: Compare configurations
  run: |
    diffai config/prod.json config/staging.json --output json > diff.json
    if [ -s diff.json ]; then
      echo "設定に差異があります"
      cat diff.json
      exit 1
    fi
```

### 3. パフォーマンスの最適化

大規模ファイルの比較時：
- `--parallel`オプションで並列処理を有効化
- `--cache`オプションで結果をキャッシュ
- 必要に応じて`--memory-limit`でメモリ使用量を制限

## コミュニティとサポート

- **ドキュメント**: https://diffai.dev/docs
- **GitHub**: https://github.com/your-org/diffai
- **問題報告**: GitHubのIssuesページへ
- **ディスカッション**: GitHubのDiscussionsで質問や提案を共有

diffaiを使って、より効率的で洞察に富んだデータ比較を始めましょう！