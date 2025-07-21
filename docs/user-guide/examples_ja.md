# 実例集

このページでは、様々なシナリオでのdiffaiの使用例を紹介し、AI駆動の分析機能を実際のケースで解説します。

## 基本的な使用例

### JSON設定ファイル

アプリケーション設定ファイルをAIの洞察と共に比較：

```bash
# 基本的なJSON比較
diffai config-dev.json config-prod.json

# ML分析付きの詳細出力
diffai --verbose config-dev.json config-prod.json

# 特定の設定パスに焦点を当てる
diffai --path "database.settings" config-dev.json config-prod.json
```

**サンプルファイル:**
```json
// config-dev.json
{
  "app_name": "myapp",
  "version": "1.0.0",
  "database": {
    "host": "localhost",
    "port": 5432,
    "ssl": false,
    "pool_size": 10
  },
  "features": {
    "debug_mode": true,
    "cache_enabled": false,
    "rate_limiting": false
  }
}

// config-prod.json
{
  "app_name": "myapp",
  "version": "1.0.1",
  "database": {
    "host": "db.production.com",
    "port": 5432,
    "ssl": true,
    "pool_size": 50
  },
  "features": {
    "debug_mode": false,
    "cache_enabled": true,
    "rate_limiting": true
  }
}
```

**出力例:**
```
configuration_risk_assessment: environment_appropriate
  ~ version: "1.0.0" -> "1.0.1"
  ~ database.host: "localhost" -> "db.production.com"
  ~ database.ssl: false -> true
  ~ database.pool_size: 10 -> 50 (5x increase - production scaling)
  ~ features.debug_mode: true -> false (production-safe)
  ~ features.cache_enabled: false -> true (performance optimization)
  ~ features.rate_limiting: false -> true (security hardening)
```

### YAML Kubernetesマニフェスト

デプロイメント設定の変更を分析：

```bash
# 基本的な比較
diffai k8s-deployment-v1.yaml k8s-deployment-v2.yaml

# セキュリティ影響の分析付き
diffai --security-analysis k8s-deployment-v1.yaml k8s-deployment-v2.yaml

# 特定のコンテナに焦点
diffai --path "spec.containers[0]" k8s-deployment-v1.yaml k8s-deployment-v2.yaml
```

**サンプルファイル:**
```yaml
# k8s-deployment-v1.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web-app
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: app
        image: myapp:1.0
        resources:
          requests:
            memory: "128Mi"
            cpu: "100m"
          limits:
            memory: "256Mi"
            cpu: "200m"

# k8s-deployment-v2.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web-app
spec:
  replicas: 5
  template:
    spec:
      containers:
      - name: app
        image: myapp:1.1
        resources:
          requests:
            memory: "256Mi"
            cpu: "200m"
          limits:
            memory: "512Mi"
            cpu: "500m"
```

**AI分析出力:**
```
scaling_analysis: horizontal_and_vertical_scaling_detected
  ~ spec.replicas: 3 -> 5 (66% increase)
  ~ spec.template.spec.containers[0].image: "myapp:1.0" -> "myapp:1.1"
  ~ resources.requests.memory: "128Mi" -> "256Mi" (2x increase)
  ~ resources.limits.cpu: "200m" -> "500m" (2.5x increase)
AI Insight: リソース要求の大幅な増加。パフォーマンステストの実施を推奨
```

## 高度な使用例

### MLモデル比較

#### PyTorchチェックポイント

学習の進行状況を追跡：

```bash
# エポック間の比較
diffai checkpoint_epoch_10.pt checkpoint_epoch_20.pt

# 詳細な統計分析
diffai --detailed-stats checkpoint_epoch_10.pt checkpoint_epoch_20.pt

# 特定のレイヤーに焦点
diffai --layer-filter "transformer.encoder" model_v1.pt model_v2.pt
```

**出力例:**
```
model_evolution_analysis: converging_normally
training_progress: healthy
  ~ transformer.encoder.layer_0.weight:
    mean: -0.002 -> -0.001 (stabilizing)
    std: 0.045 -> 0.032 (variance reduction)
  ~ optimizer.learning_rate: 0.001 -> 0.0001 (scheduled decay)
AI Insight: モデルは正常に収束中。学習率の調整が適切
```

#### Safetensorsモデル

量子化の影響を分析：

```bash
# 量子化前後の比較
diffai model_fp32.safetensors model_int8.safetensors

# 詳細な量子化分析
diffai --quantization-analysis model_fp32.safetensors model_int8.safetensors
```

**出力例:**
```
quantization_impact_analysis:
  compression_ratio: 4.0x
  accuracy_impact: minimal (-0.2%)
  inference_speedup: 2.8x
  memory_reduction: 75%
推奨: エッジデバイスへのデプロイメントに適している
```

### データ移行検証

#### CSVデータの整合性チェック

```bash
# ID列を使用した比較
diffai old_users.csv new_users.csv --array-id-key "user_id"

# 許容誤差を設定した数値比較
diffai financial_data_old.csv financial_data_new.csv \
  --array-id-key "transaction_id" \
  --epsilon 0.01
```

**サンプルデータ:**
```csv
# old_users.csv
user_id,name,email,status
1,Alice,alice@example.com,active
2,Bob,bob@example.com,active
3,Charlie,charlie@example.com,inactive

# new_users.csv
user_id,name,email,status,created_at
1,Alice,alice@example.com,active,2024-01-01
2,Bob,bob@company.com,active,2024-01-01
4,David,david@example.com,active,2024-03-15
```

**出力例:**
```
data_migration_analysis: partial_migration_with_changes
  ~ [user_id=2].email: "bob@example.com" -> "bob@company.com"
  - [user_id=3]: Charlie (レコード削除)
  + [user_id=4]: David (新規レコード)
  + 全レコードに created_at カラムが追加
警告: 3件中1件のレコードが削除されています
```

### API応答比較

#### RESTful APIレスポンス

バージョン間の互換性チェック：

```bash
# タイムスタンプを無視して比較
diffai api_v1_response.json api_v2_response.json \
  --ignore-keys-regex "timestamp|request_id"

# 破壊的変更の検出
diffai --breaking-changes api_v1_response.json api_v2_response.json
```

**サンプルレスポンス:**
```json
// api_v1_response.json
{
  "status": "success",
  "data": {
    "users": [
      {"id": 1, "name": "Alice", "role": "admin"},
      {"id": 2, "name": "Bob", "role": "user"}
    ],
    "total": 2
  },
  "timestamp": "2024-01-01T10:00:00Z"
}

// api_v2_response.json
{
  "status": "success",
  "data": {
    "users": [
      {"id": 1, "username": "Alice", "role": "admin", "permissions": ["read", "write"]},
      {"id": 2, "username": "Bob", "role": "user", "permissions": ["read"]}
    ],
    "total_count": 2,
    "page_info": {"page": 1, "per_page": 10}
  },
  "timestamp": "2024-03-01T10:00:00Z"
}
```

**互換性分析:**
```
api_compatibility_analysis: breaking_changes_detected
  ! data.users[].name -> data.users[].username (フィールド名変更)
  + data.users[].permissions (新規必須フィールド)
  ! data.total -> data.total_count (フィールド名変更)
  + data.page_info (新規オブジェクト - 後方互換性あり)
推奨: クライアントコードの更新が必要
```

## 実践的なワークフロー

### CI/CDパイプライン統合

#### GitHub Actions

```yaml
name: Configuration Drift Detection
on:
  pull_request:
    paths:
      - 'config/**'

jobs:
  config-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Install diffai
        run: cargo install diffai
      
      - name: Compare configurations
        run: |
          diffai config/production.json config/staging.json \
            --output json \
            --threshold 0.1 > config_diff.json
          
      - name: Post analysis comment
        if: always()
        uses: actions/github-script@v6
        with:
          script: |
            const fs = require('fs');
            const diff = JSON.parse(fs.readFileSync('config_diff.json', 'utf8'));
            
            let comment = '## 設定差分分析\n\n';
            if (diff.risk_level === 'high') {
              comment += '⚠️ **高リスクの変更が検出されました**\n\n';
            }
            
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: comment + '```json\n' + JSON.stringify(diff, null, 2) + '\n```'
            });
```

### 監視とアラート

#### 設定ドリフト検出

```bash
#!/bin/bash
# config-drift-monitor.sh

BASELINE="/etc/myapp/config.json"
CURRENT="/tmp/current-config.json"

# 現在の設定を取得
kubectl get configmap myapp-config -o json | jq '.data' > "$CURRENT"

# 比較実行
DIFF_OUTPUT=$(diffai "$BASELINE" "$CURRENT" --output json)

# リスクレベルをチェック
RISK=$(echo "$DIFF_OUTPUT" | jq -r '.risk_assessment.level')

if [ "$RISK" = "high" ]; then
  # アラート送信
  curl -X POST "$SLACK_WEBHOOK" \
    -H 'Content-Type: application/json' \
    -d "{
      \"text\": \"⚠️ 設定ドリフト検出: 高リスクの変更\",
      \"attachments\": [{
        \"color\": \"danger\",
        \"text\": \`\`\`$DIFF_OUTPUT\`\`\`
      }]
    }"
fi
```

### データ品質保証

#### ETLパイプライン検証

```python
# etl_validation.py
import subprocess
import json
import sys

def validate_etl_output(source_file, target_file):
    """ETL処理の出力を検証"""
    
    # diffaiを使用して比較
    result = subprocess.run([
        'diffai',
        source_file,
        target_file,
        '--output', 'json',
        '--array-id-key', 'record_id',
        '--epsilon', '0.001'
    ], capture_output=True, text=True)
    
    diff_data = json.loads(result.stdout)
    
    # データ品質メトリクスをチェック
    if diff_data.get('data_quality', {}).get('missing_records', 0) > 0:
        print(f"エラー: {diff_data['data_quality']['missing_records']}件のレコードが欠落")
        return False
    
    if diff_data.get('data_quality', {}).get('type_mismatches', 0) > 0:
        print(f"警告: {diff_data['data_quality']['type_mismatches']}件の型不一致")
    
    return True

if __name__ == "__main__":
    if not validate_etl_output(sys.argv[1], sys.argv[2]):
        sys.exit(1)
```

## パフォーマンス最適化

### 大規模ファイルの処理

```bash
# ストリーミングモードで大規模ファイルを比較
diffai large_dataset1.json large_dataset2.json \
  --streaming \
  --memory-limit 1GB

# 並列処理を有効化
diffai data_dir1/ data_dir2/ \
  --recursive \
  --parallel 8

# 結果をキャッシュして再実行を高速化
diffai --cache-dir /tmp/diffai-cache \
  model1.safetensors model2.safetensors
```

## トラブルシューティング例

### メモリ不足エラー

```bash
# 問題: メモリ不足エラー
# Error: Out of memory

# 解決策1: チャンク処理
diffai huge_file1.json huge_file2.json \
  --chunk-size 100MB

# 解決策2: 特定のパスのみ比較
diffai huge_file1.json huge_file2.json \
  --path "data.important_section"
```

### エンコーディング問題

```bash
# 問題: 文字化け
# 解決策: エンコーディングを明示的に指定
diffai --encoding utf-8 japanese_file1.json japanese_file2.json

# BOM付きUTF-8の場合
diffai --encoding utf-8-sig file1.json file2.json
```

これらの例を参考に、あなたのユースケースに合わせてdiffaiを活用してください。より詳細な情報は[CLIリファレンス](../reference/cli-reference_ja.md)をご覧ください。