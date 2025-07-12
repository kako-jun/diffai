#!/bin/bash
set -euo pipefail

# Same environment as GitHub Actions CI
export CARGO_TERM_COLOR=always
export RUST_BACKTRACE=1

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_step() {
    echo -e "${GREEN}$1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Trap to handle errors
trap 'print_error "CI failed at step: ${CURRENT_STEP:-unknown}"' ERR

echo "🔄 Running complete CI simulation locally (matching GitHub Actions exactly)..."

CURRENT_STEP="formatting check"
print_step "📝 Step 1: Check formatting"
cargo fmt --all -- --check

CURRENT_STEP="clippy analysis"
print_step "🔍 Step 2: Run Clippy (workspace-wide)"
cargo clippy --workspace --all-targets --all-features -- -D warnings

CURRENT_STEP="workspace build"
print_step "🏗️ Step 3: Build workspace"
cargo build --workspace --verbose

CURRENT_STEP="workspace tests"
print_step "🧪 Step 4: Run tests (workspace-wide)"
cargo test --workspace --verbose

CURRENT_STEP="doc tests"
print_step "📚 Step 5: Run doc tests (workspace-wide)"
cargo test --workspace --doc

CURRENT_STEP="release build"
print_step "🚀 Step 6: Release build (workspace)"
cargo build --workspace --release --verbose

CURRENT_STEP="release tests"
print_step "🧪 Step 7: Run tests in release mode (workspace)"
cargo test --workspace --release --verbose

CURRENT_STEP="security audit"
print_step "🔒 Step 8: Security audit"
if command -v cargo-audit &> /dev/null; then
    cargo audit
else
    print_warning "cargo-audit not installed, skipping security audit"
    print_warning "Install with: cargo install cargo-audit"
fi

CURRENT_STEP="CLI functionality test"
print_step "🎯 Step 9: Test core CLI functionality"

# Create temporary directory for test files
TEST_DIR=$(mktemp -d)
trap 'rm -rf "$TEST_DIR"' EXIT

# Create test JSON files for basic functionality
cat > "$TEST_DIR/test_model1.json" << 'EOF'
{"model": {"layers": [{"name": "conv1", "params": 1000}]}}
EOF

cat > "$TEST_DIR/test_model2.json" << 'EOF'
{"model": {"layers": [{"name": "conv1", "params": 1200}]}}
EOF

# Test JSON diff (must succeed)
print_step "  Testing JSON diff..."
cargo run --bin diffai -- "$TEST_DIR/test_model1.json" "$TEST_DIR/test_model2.json" > /dev/null

# Test YAML diff (must succeed)
cat > "$TEST_DIR/test_config1.yaml" << 'EOF'
config:
  learning_rate: 0.01
  batch_size: 32
EOF

cat > "$TEST_DIR/test_config2.yaml" << 'EOF'
config:
  learning_rate: 0.001
  batch_size: 64
EOF

print_step "  Testing YAML diff..."
cargo run --bin diffai -- "$TEST_DIR/test_config1.yaml" "$TEST_DIR/test_config2.yaml" > /dev/null

# Test stdin processing (must succeed)
print_step "  Testing stdin processing..."
echo '{"experiment": {"accuracy": 0.95}}' | cargo run --bin diffai -- - "$TEST_DIR/test_model1.json" > /dev/null

# Clear the trap since we'll clean up manually
trap - ERR

print_step "✅ All CI steps completed successfully!"
print_step "🚀 Ready to push to remote repository"
echo "📋 Note: This script reproduces the exact GitHub Actions CI pipeline"