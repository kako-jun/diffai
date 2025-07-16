# TODOリスト

## 🎉 v0.3.4リリース完了 (2025-07-14)

### ✅ 完了済み構造現代化
- [x] `.claude/` ディレクトリ構造作成
- [x] CLAUDE.md目次化（詳細を専用ファイルに移動）
- [x] scripts/ 分類（release/testing/utils完備）
- [x] Python maturin化（バイナリ同梱配布成功）
- [x] npm ユニバーサル化（全OS同梱済み）
- [x] GitHub Actions 2幕構成移植（Act1/Act2完全動作）
- [x] スクリプト動的バージョン管理（ハードコード根絶）
- [x] PyPI distribution修復（maturinエラー解決）
- [x] 全プラットフォーム統一リリース（crates.io/npm/PyPI）

### 🚀 次期優先事項
**Infrastructure Modernization完了により、ML機能拡張にフォーカス可能**

## 🔧 Phase 4: ML分析機能拡張 (新優先度)

### 🧠 フォーマット対応拡張
- [ ] TensorFlow SavedModel対応
- [ ] ONNX形式完全サポート
- [ ] Hugging Face Hub直接統合
- [ ] PyTorch Lightning checkpoint対応

### 📊 高度可視化・解析
- [ ] インタラクティブ差分可視化（Web UI）
- [ ] モデルアーキテクチャ比較図生成
- [ ] パフォーマンス回帰自動検出
- [ ] 大容量モデル最適化（streaming diff）

### 🔗 MLOps統合
- [ ] MLflow experiment tracking連携
- [ ] Weights & Biases統合
- [ ] DVC (Data Version Control) 対応
- [ ] Docker/Kubernetes デプロイ支援

## 🧹 品質・エコシステム強化

### 📊 品質改善
- [ ] テストカバレッジ向上
- [ ] ドキュメント整合性チェック自動化
- [ ] エラーハンドリング改善

### 🧪 テスト統合の現状 (2025-07-17)
**リリーステストとプロジェクトテストの分離状況**

#### 現在のテスト体系
- **tests/フォルダ**: Rust内部テスト（unit/integration）
  - `cargo test`で実行される標準テスト
  - npm/Python関数テストは`#[ignore]`で無効化
- **04スクリプト**: Act1テスト（Rust crate機能確認）
  - 現在は外部プロジェクト作成で関数呼び出しテスト実装
  - **問題**: 無限ループ発生、08スクリプトと統一すべき
- **05スクリプト**: Act2テスト（npm/Python関数確認）
  - 実際のインストール→インポート→関数呼び出しテスト
- **08スクリプト**: 公開パッケージテスト
  - 各エコシステムでの実際の動作確認

#### 統一化の方向性
- **04スクリプト**: 08スクリプトと同様に`cargo test`アプローチに変更
- **05スクリプト**: 08スクリプトと同レベルのテスト内容を実装済み
- **実際の関数テスト**: 04/05/08スクリプトで実装、tests/フォルダは内部テストのみ

### 🌟 エコシステム拡張
- [ ] Homebrew Formula作成
- [ ] Docker Hub公開
- [ ] VS Code拡張機能検討

---

## 📈 プロジェクト現状サマリー
- **✅ Infrastructure Modernization完了**: v0.3.4で全プラットフォーム安定配布達成
- **✅ 技術負債解消**: 動的バージョン管理・CI/CD・パッケージ配布の完全自動化
- **🚀 次フェーズ**: MLエコシステム拡張とユーザー体験向上にフォーカス可能

**戦略**: 安定した基盤の上で、AI/ML特化機能の革新的拡張を推進する。