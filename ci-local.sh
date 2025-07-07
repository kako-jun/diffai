#!/bin/bash
set -e

# Same environment as GitHub Actions CI
export CARGO_TERM_COLOR=always

echo "🔄 Running complete CI simulation locally (matching GitHub Actions exactly)..."

echo "📝 Step 1: Check formatting"
cargo fmt --all -- --check

echo "🔍 Step 2: Run Clippy"
cargo clippy --all-targets --all-features -- -D warnings

echo "🏗️ Step 3: Build"
cargo build --verbose

echo "🧪 Step 4: Run tests"
cargo test --verbose

echo "📚 Step 5: Run doc tests"
cargo test --doc

echo "🚀 Step 6: Release build"
cargo build --release --verbose

echo "🧪 Step 7: Run tests in release mode"
cargo test --release --verbose

echo "🎯 Step 8: Test core CLI functionality"
# Test basic ML model comparison (must succeed)
echo "Testing CLI with sample data..."

# Create test JSON files for basic functionality
echo '{"model": {"layers": [{"name": "conv1", "params": 1000}]}}' > /tmp/test_model1.json
echo '{"model": {"layers": [{"name": "conv1", "params": 1200}]}}' > /tmp/test_model2.json

# Test JSON diff (must succeed)
cargo run --bin diffai -- /tmp/test_model1.json /tmp/test_model2.json > /dev/null

# Test YAML diff (must succeed)
echo 'config:
  learning_rate: 0.01
  batch_size: 32' > /tmp/test_config1.yaml
echo 'config:
  learning_rate: 0.001
  batch_size: 64' > /tmp/test_config2.yaml

cargo run --bin diffai -- /tmp/test_config1.yaml /tmp/test_config2.yaml > /dev/null

# Test stdin processing (must succeed)
echo '{"experiment": {"accuracy": 0.95}}' | cargo run --bin diffai -- - /tmp/test_model1.json > /dev/null

# Cleanup
rm -f /tmp/test_model1.json /tmp/test_model2.json /tmp/test_config1.yaml /tmp/test_config2.yaml

echo "✅ All CI steps completed successfully!"
echo "🚀 Ready to push to remote repository"
echo "📋 Note: This script reproduces the exact GitHub Actions CI pipeline"