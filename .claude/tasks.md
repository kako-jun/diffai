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

### 🧪 テスト統合完了 (2025-07-17)
**共通テストフレームワークによる統一テスト体系確立**

#### 新しいテスト体系
- **tests/フォルダ**: 純粋にRust内部テスト（unit/integration）
  - `cargo test`で実行される標準テスト
  - npm/Pythonテストは完全に削除済み
- **共通テストフレームワーク**: `.github/rust-cli-kiln/scripts/testing/common/`
  - `test-utils.sh` - 共通ユーティリティ（ログ、結果追跡）
  - `test-rust-binary.sh` - Rustバイナリテスト
  - `test-rust-crate.sh` - Rustクレートテスト（cargo test含む）
  - `test-npm-package.sh` - npmパッケージテスト
  - `test-python-package.sh` - Pythonパッケージテスト

#### リファクタリング後のスクリプト
- **04スクリプト**: Rustエコシステムテスト（135行、50%削減）
  - 共通フレームワークを使用、無限ループ問題解決
- **05スクリプト**: npm/Pythonテスト（213行、42%削減）
  - 共通フレームワークを使用、統一されたテストロジック
- **08スクリプト**: 公開パッケージテスト（47行、80%削減）
  - 完全に共通フレームワークに依存

#### 3プロジェクト共通化完了
- **diffai/diffx/lawkit**: 全く同じテストコードを使用可能
- **PROJECT_NAME変数**: 自動的に各プロジェクトに対応
- **完全なDRY原則**: テストロジックの重複を完全に排除

#### 統一された6ファイルテスト構造 (2025-07-17)
**全プロジェクト・全言語パッケージで同一のテスト構造を適用**

##### npm/Pythonパッケージ統一構造
**npm**: `tests/` ディレクトリに6ファイル
- `cli.test.js` - CLI基本機能（--version, --help）
- `basic.test.js` - 基本機能・出力フォーマット（JSON/YAML）
- `binary.test.js` - 同梱バイナリ直接実行
- `formats.test.js` - ファイル形式サポート（JSON/YAML/CSV/TXT）
- `errors.test.js` - エラーハンドリング（非存在ファイル、不正形式等）
- `features.test.js` - 機能特化（ML分析、統計、色出力等）

**Python**: `tests/` ディレクトリに6ファイル（同一構造）
- `test_cli.py` - CLI基本機能
- `test_basic.py` - 基本機能・出力フォーマット
- `test_binary.py` - 同梱バイナリ直接実行
- `test_formats.py` - ファイル形式サポート
- `test_errors.py` - エラーハンドリング
- `test_features.py` - 機能特化

##### 実装完了状況
- ✅ **diffai-npm**: 6ファイル構造完全実装（`npm test`で統一実行）
- ✅ **diffai-python**: 6ファイル構造完全実装（`python test.py`で統一実行）
- 📋 **計画**: Rustテスト構造統一 + diffx/lawkitプロジェクト展開

##### 設計方針
- **業界標準コマンド**: `npm test`, `python test.py`, `cargo test`のみ使用（fallback削除）
- **一時ファイル管理**: 全テストで適切なクリーンアップ実装
- **結果オブジェクト**: `{passed, total}`形式で統一
- **npx実行**: npmパッケージテストは`npx diffai`でCLI実行
- **プラットフォーム対応**: バイナリ名の自動検出機能

##### 次期計画: Rustテスト構造統一 (優先度: 中)
**現状**: `/home/d131/repos/42/2025/diffai/tests/` は unit/ と integration/ 混在
**目標**: 6ファイル構造でRustテストも統一

**Rust統一構造 (tests/ ディレクトリ)**:
- `cli_tests.rs` - CLIコマンドライン実行（`std::process::Command`でバイナリ実行）
- `core_tests.rs` - diffai-coreクレート関数直接呼び出し（`use diffai_core::*`）
- `basic_tests.rs` - 基本機能・出力フォーマット（JSON/YAML）
- `formats_tests.rs` - ファイル形式サポート（JSON/YAML/CSV/TXT）
- `errors_tests.rs` - エラーハンドリング（非存在ファイル、不正形式等）
- `features_tests.rs` - 機能特化（ML分析、統計、色出力等）

**メリット**:
- 3言語で論理的に同一のテスト分類（言語特性を考慮した適切な分割）
- `cargo test`での統一実行
- テストファイル数最適化（現在の unit/integration 細分化から統合）
- メンテナンス性向上

**言語別テスト対応表**:
| 分類 | Rust | JavaScript | Python |
|------|------|------------|--------|
| CLI実行 | `cli_tests.rs` | `cli.test.js` | `test_cli.py` |
| ライブラリ/バイナリ | `core_tests.rs` | `binary.test.js` | `test_binary.py` |
| 基本機能 | `basic_tests.rs` | `basic.test.js` | `test_basic.py` |
| フォーマット | `formats_tests.rs` | `formats.test.js` | `test_formats.py` |
| エラー処理 | `errors_tests.rs` | `errors.test.js` | `test_errors.py` |
| 機能特化 | `features_tests.rs` | `features.test.js` | `test_features.py` |

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