# 示例

本页面提供了在各种场景中使用diffai的综合示例，展示其AI驱动的分析能力。

## 基本用法示例

### JSON配置文件

使用AI洞察比较应用程序配置文件：

```bash
# 基本JSON比较
diffai config-dev.json config-prod.json

# 带有ML分析的输出
diffai --verbose config-dev.json config-prod.json

# 专注于特定配置路径
diffai --path "database.settings" config-dev.json config-prod.json
```

**示例文件:**
```json
// config-dev.json
{
  "app_name": "myapp",
  "version": "1.0.0",
  "database": {
    "host": "localhost",
    "port": 5432,
    "name": "dev_db"
  },
  "debug": true
}

// config-prod.json  
{
  "app_name": "myapp",
  "version": "1.0.1",
  "database": {
    "host": "prod-server.com",
    "port": 5432,
    "name": "prod_db"
  },
  "debug": false
}
```

### YAML Docker Compose文件

```bash
# 比较Docker Compose配置
diffai docker-compose.yml docker-compose.prod.yml

# 忽略时间戳相关的更改
diffai --ignore-keys-regex "created_at|updated_at" docker-compose.yml docker-compose.new.yml
```

### CSV数据分析

使用统计分析比较数据集：

```bash
# 使用ML洞察比较销售数据
diffai --verbose --epsilon 0.05 sales-q1.csv sales-q2.csv

# 专注于特定列
diffai --path "revenue,profit" financial-data-old.csv financial-data-new.csv
```

## 高级AI分析示例

### 异常检测

```bash
# 检测用户行为数据中的异常
diffai --epsilon 0.01 --verbose user-metrics-baseline.json user-metrics-current.json

# 使用智能ID匹配进行数组比较
diffai --array-id-key "user_id" users-jan.json users-feb.json
```

### 机器学习模型比较

```bash
# 比较ML模型配置
diffai --verbose --path "hyperparameters" model-v1.json model-v2.json

# 使用统计容差比较训练结果
diffai --epsilon 0.001 training-results-baseline.json training-results-new.json
```

### 科学数据分析

```bash
# 比较实验数据集
diffai --verbose --epsilon 0.01 experiment-control.csv experiment-test.csv

# 研究目录的递归比较
diffai --recursive --output analysis-report.json research-v1/ research-v2/
```

## 格式特定示例

### XML配置文件

```bash
# 比较Spring Boot配置
diffai application-dev.xml application-prod.xml

# Maven POM文件比较
diffai pom.xml pom.xml.backup
```

### TOML Cargo文件

```bash
# 比较Rust项目依赖项
diffai Cargo.toml Cargo.toml.new

# 仅关注依赖项更改
diffai --path "dependencies" Cargo.toml Cargo.toml.updated
```

## 集成示例

### CI/CD管道集成

```bash
#\!/bin/bash
# 自动化配置漂移检测
diffai --format json --output config-drift.json \
  production-config.json staging-config.json

# 如果检测到重大更改则以错误退出
if [ $(jq '.changes | length' config-drift.json) -gt 10 ]; then
  echo "检测到重大配置漂移！"
  exit 1
fi
```

### API响应监控

```bash
# 回归测试的API响应比较
diffai --ignore-keys-regex "timestamp|request_id" \
  api-response-baseline.json api-response-current.json

# 仅关注数据负载更改
diffai --path "data" api-v1-response.json api-v2-response.json
```

## 现实世界场景

### 系统配置监控

```bash
# 日常配置漂移检查
diffai --recursive --format json \
  /etc/production-config/ /etc/staging-config/ > daily-drift-report.json
```

### 数据质量保证

```bash
# 比较数据质量指标
diffai --verbose --epsilon 0.05 \
  quality-metrics-baseline.json quality-metrics-current.json
```

## 提示和最佳实践

### 选择正确的选项

- 对科学数据中的数值容差使用`--epsilon`
- 对时间戳等动态字段使用`--ignore-keys-regex`
- 使用`--path`将分析集中在特定数据部分
- 对智能数组元素匹配使用`--array-id-key`

### 性能优化

- 从基本比较开始，仅在需要时添加`--verbose`
- 对大型数据集使用`--format json`以减少内存使用
- 对于非常大的文件考虑使用`--path`过滤

### 集成模式

- 与`jq`结合进行JSON输出处理
- 在脚本中使用退出代码进行自动化决策
- 在CI/CD中生成报告以进行变更跟踪

