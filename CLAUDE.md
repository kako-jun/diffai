# diffai の思想（Philosophy）
「AI/MLデータの差分を、誰でも、どこでも、簡単に」

diffaiは、AI・機械学習分野でのデータ差分抽出に特化したツールです。
従来の diff はテキストベースで、構造を理解できない。
diffai は JSON/YAML/TOMLなどの構造化データに特化し、AI/MLのデータセット、設定ファイル、モデルパラメータの変更を明確に可視化する。

## 名前「diffai」に込めた意味
diff + ai の「ai」は：

- **AI/ML focused** AI・機械学習に特化
- **intelligent** 賢い差分抽出
- **automated** 自動化された分析
- **accessible** 誰でもアクセス可能
- **advanced** 高度な機能

## diffaiとdiffxの関係
diffaiは、diffxプロジェクトの兄弟プロジェクトです。
diffxは汎用的な構造化データ差分ツールであり、diffaiはAI/ML分野に特化したバージョンです。

### 分離の理由
- **対象分野の特化**: diffxは汎用、diffaiはAI/ML特化
- **機能の最適化**: AI/MLデータに特化した機能追加
- **開発の独立性**: それぞれの分野に最適化した開発

## 技術仕様
diffxの技術基盤をそのまま継承し、以下の追加機能を予定：

- **モデルパラメータ差分**: 深層学習モデルの重み変更追跡
- **データセット差分**: 訓練データの変更検出
- **実験設定差分**: ハイパーパラメータの変更追跡
- **評価指標差分**: モデル性能の変化可視化

## 対応フォーマット
- JSON（設定ファイル、APIレスポンス）
- YAML（MLOps設定、Kubernetes設定）
- TOML（プロジェクト設定）
- XML（レガシーシステム連携）
- INI（設定ファイル）
- CSV（データセット、実験結果）

## 将来的な展望
- **MLOps統合**: CI/CDパイプラインでの自動検証
- **実験追跡**: MLflow、Weights & Biasesとの連携
- **モデルバージョニング**: Git LFS、DVC統合
- **可視化**: Jupyter Notebook、Tensorboard連携