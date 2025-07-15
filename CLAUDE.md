# diffai の思想（Philosophy）
「AI駆動の差分解析で、意味のある変更を自動検出」
従来の diff はテキストベースで、構造を理解できない。
diffai は PyTorch/Safetensors などの ML モデルファイルに特化した差分抽出ツール。
AI・機械学習分野でのデータ・モデル変更を明確に可視化し、研究開発からMLOpsまで幅広い用途に対応する。

### diffxとの関係
- **diffx**: 汎用構造化データ差分ツール  
- **diffai**: AI/ML特化版（diffxの兄弟プロジェクト）

# 🚨 重要な開発ルール (Important Development Rules)

## 🤖 Claude実行ルール（判断権限剥奪）

### リリース作業時の絶対ルール
- **❌ 独自判断完全禁止** - 手順書にない行動は一切取らない
- **❌ FAIL時は即座に停止** - 「警告だから無視」「たぶん大丈夫」判断禁止
- **❌ 設定ファイルコミット禁止** - .claude/, .env等は絶対にコミットしない  
- **❌ エラー時の勝手な修正禁止** - 必ずユーザーに報告・確認
- **✅ 機械的実行のみ許可** - スクリプト実行→結果確認→報告

### 機械的実行チェックリスト
1. **スクリプト実行** - 指定されたコマンドを実行
2. **結果確認** - PASS/FAIL/WARNの判定のみ
3. **FAIL発生時** - 即座に停止、詳細をユーザーに報告
4. **PASS確認時** - 次のステップに進行
5. **判断必要時** - 必ずユーザーに確認を求める

### 絶対禁止事項
- 🚫 「警告だから無視して進む」思考
- 🚫 「修正できそうだから直す」判断  
- 🚫 「手順書にないが必要そう」な追加作業
- 🚫 ローカル設定ファイルのコミット
- 🚫 エラー原因の推測に基づく修正

## Claude対応時の必須ルール (Claude Response Rules)
**技術質問への回答では以下を必ず守ること:**
- **完全な仕様を最初から提供**: 条件・制限・例外をすべて含める
- **小出し回答の禁止**: 「確認が必要」「追加質問待ち」の姿勢を取らない
- **具体例を複数提示**: 動作例・制限例・エラー例を網羅
- **背景情報も同時提供**: なぜその仕様なのか、他の選択肢との違い
- **例**: ML分析なら「30+機能自動実行、CLI/JSON/YAML対応、PyTorch/Safetensors完全対応、具体例3パターン」を一度に報告

## プッシュ前の必須チェック (Pre-Push Requirements)
**必ずプッシュ前に以下を実行すること:**
```bash
./scripts/testing/quick-check.sh
```

- フォーマット・Clippy・ビルド・テストの基本チェックを実行
- エラーが発生したら即座に停止する

## コンテキスト効率化ルール (Context Efficiency Rules)
**CLAUDE.mdは目次として使用し、詳細情報は以下の専用ファイルを参照:**

- **📋 タスクリスト**: `.claude/tasks.md` を参照
- **🚀 リリース手順**: `.claude/release-guide.md` を参照

**重要**: 詳細が必要な時のみ該当ファイルを読むこと。CLAUDE.md自体は最小限に保つ。

---

# 📦 現在の状況 (Current Status)

## 🎯 プロジェクト完成度
**diffai は ML特化差分解析・科学データ対応・マルチプラットフォーム公開が完了**

- **✅ AI/ML対応**: PyTorch/Safetensors完全サポート
- **✅ 科学データ対応**: NumPy/MATLAB配列解析
- **✅ ML分析機能**: 統計・アーキテクチャ・異常検知等11機能
- **✅ 3言語エコシステム**: Rust(crates.io), JavaScript(npm), Python(PyPI)
- **✅ 包括的テスト**: TDD + ドキュメント例テスト完備

## 📦 最新リリース: v0.3.5 (2025-07-15)
- **🤖 Intelligent ML Recommendations**: 11軸評価マトリックス・3段階優先度システム
- **🎯 Interface Simplification**: 35個のMLフラグ削除・自動包括分析
- **🚀 Universal Release Automation**: 9ステップ自動リリースワークフロー
- **📚 Enhanced Testing**: 150統合テスト + 68単体テスト完全通過
- **🐛 Critical Fixes**: テスト構造修正・統合テストパス問題解決

## 💻 提供形態（✅完全動作確認済み）
- **🦀 Rust (crates.io)**: diffai-core/diffai-cli v0.3.5
- **📦 npm (diffai-js)**: クロスプラットフォームバイナリ同梱 v0.3.5
- **🐍 Python (PyPI)**: maturin製wheel配布 v0.3.5

---

# 🚀 開発ガイド (Development Guide)

## リリース手順
```bash
# 詳細手順は以下を参照
cat .claude/release-guide.md
```

## Python環境管理
```bash
# 必ずuvでvenv作成
uv venv && source .venv/bin/activate
```