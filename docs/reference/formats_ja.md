# サポートされている形式

diffaiがサポートするファイル形式とその仕様。

## 概要

diffaiは、機械学習モデルから科学データ、構造化設定ファイルまで、さまざまな用途に最適化された幅広いファイル形式をサポートします。

## 機械学習モデル形式

### PyTorchモデル
- **拡張子**: `.pt`, `.pth`
- **形式**: CandleライブラリとのPickle形式統合
- **サポート**: テンソル統計、形状変更、パラメータ追加/削除
- **データ型**: F32, F64, F16, BF16, I64, U32, U8

**例**:
```bash
diffai model1.pt model2.pt
```

### Safetensorsモデル
- **拡張子**: `.safetensors`
- **形式**: HuggingFace Safetensors形式（推奨）
- **サポート**: 高速読み込み、メモリ効率、セキュア
- **データ型**: F32, F64, F16, BF16, I64, U32, U8

**例**:
```bash
diffai model1.safetensors model2.safetensors
```

## 科学データ形式

### NumPy配列
- **拡張子**: `.npy`, `.npz`
- **形式**: NumPy配列形式
- **サポート**: 統計分析、形状比較、データ型検証
- **データ型**: 全NumPyデータ型

**例**:
```bash
diffai data1.npy data2.npy
diffai archive1.npz archive2.npz
```

### MATLAB行列
- **拡張子**: `.mat`
- **形式**: MATLAB行列ファイル
- **サポート**: 複素数、変数名、多次元配列
- **データ型**: double, single, int8-64, uint8-64, logical

**例**:
```bash
diffai simulation1.mat simulation2.mat
```

## 構造化データ形式

### JSON
- **拡張子**: `.json`
- **形式**: JavaScript Object Notation
- **サポート**: ネストされたオブジェクト、配列、基本データ型
- **用途**: 設定ファイル、APIレスポンス、実験結果

### YAML
- **拡張子**: `.yaml`, `.yml`
- **形式**: YAML Ain't Markup Language
- **サポート**: 階層構造、コメント、複数ドキュメント
- **用途**: 設定ファイル、CI/CD、Kubernetes

### TOML
- **拡張子**: `.toml`
- **形式**: Tom's Obvious, Minimal Language
- **サポート**: 型安全、設定指向構造
- **用途**: Rust設定、プロジェクト設定

### XML
- **拡張子**: `.xml`
- **形式**: eXtensible Markup Language
- **サポート**: 属性、名前空間、階層構造
- **用途**: 設定ファイル、データ交換

### INI
- **拡張子**: `.ini`
- **形式**: Initialization file
- **サポート**: セクション、キー値ペア
- **用途**: 設定ファイル、レガシーアプリケーション

### CSV
- **拡張子**: `.csv`
- **形式**: Comma-Separated Values
- **サポート**: 表形式データ、ヘッダー行
- **用途**: データ分析、スプレッドシート

## 形式自動検出

diffaiは以下の優先順位で形式を検出します：

1. **--format** オプション（明示的指定）
2. **ファイル拡張子** ベースの自動検出
3. **設定ファイル** デフォルト形式
4. **エラー**（判定不可能な場合）

## 実装状況

### フェーズ1-2（完了）
- ✅ PyTorch (.pt, .pth)
- ✅ Safetensors (.safetensors)
- ✅ NumPy (.npy, .npz)
- ✅ MATLAB (.mat)
- ✅ JSON (.json)
- ✅ YAML (.yaml, .yml)
- ✅ TOML (.toml)
- ✅ XML (.xml)
- ✅ INI (.ini)
- ✅ CSV (.csv)

### フェーズ3（計画中）
- ⏳ TensorFlow (.pb, .h5, SavedModel)
- ⏳ ONNX (.onnx)
- ⏳ HDF5 (.h5, .hdf5)

## 形式比較

### MLモデル形式

| 形式 | 速度 | セキュリティ | メモリ効率 | 推奨 |
|------|------|------------|-----------|------|
| Safetensors | 高 | 高 | 高 | 推奨 |
| PyTorch | 中 | 低 | 中 | 利用可能 |

### 科学データ形式

| 形式 | 精度 | 圧縮 | 複素数 | 推奨用途 |
|------|------|------|--------|----------|
| NumPy | 高 | 中 | サポート | Python科学計算 |
| MATLAB | 高 | 低 | サポート | MATLAB環境 |

## 関連ドキュメント

- [CLIリファレンス](cli-reference_ja.md) - 完全なコマンドラインオプション
- [ML分析機能](ml-analysis_ja.md) - 機械学習特化機能
- [出力形式](output-formats_ja.md) - 出力形式仕様

