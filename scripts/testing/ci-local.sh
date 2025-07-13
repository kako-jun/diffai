#!/bin/bash
set -euo pipefail

# CI-local: GitHub Actions CI checks reproduction (NO BUILDS)
# Reproduces CI checks without heavy compilation

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_step() {
    echo -e "${GREEN}$1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Trap errors
trap 'print_error "CI check failed at: ${CURRENT_STEP:-unknown}"' ERR

# Same environment as GitHub Actions
export CARGO_TERM_COLOR=always
export RUST_BACKTRACE=1

echo "🔄 CI-local: Fast checks (NO compilation)"
print_info "Platform: $(uname -s) $(uname -m)"
print_info "Rust: $(rustc --version)"
echo ""

# =============================================================================
# FAST CHECKS ONLY
# =============================================================================

CURRENT_STEP="formatting check"
print_step "📝 Check formatting"
cargo fmt --all -- --check

CURRENT_STEP="clippy analysis"
print_step "🔍 Run clippy (workspace-wide)"
cargo clippy --workspace --all-targets --all-features -- -D warnings

CURRENT_STEP="dependency verification"
print_step "🔗 Verify dependencies (catches cross-platform issues)"

# Check for problematic dependencies that cause GitHub Actions failures
print_info "Checking for Python binding dependencies..."
if grep -q "^[^#]*pyo3\|^[^#]*numpy.*=" Cargo.toml diffai-*/Cargo.toml 2>/dev/null; then
    print_error "Found Python binding dependencies!"
    print_error "These cause PyO3/Python version issues on macOS/Windows"
    print_info "Check: pyo3, numpy dependencies in Cargo.toml files"
    exit 1
fi

# Verify package name for release workflow
print_info "Checking package names for release workflow..."
if ! grep -q 'name = "diffai"' diffai-cli/Cargo.toml; then
    print_error "Package name issue: diffai-cli/Cargo.toml should have name = \"diffai\""
    exit 1
fi

# Check workflow files for incorrect package names
print_info "Checking workflow files for package name consistency..."
if grep -q "diffai-cli" .github/workflows/*.yml 2>/dev/null; then
    print_error "Found 'diffai-cli' in workflow files!"
    print_error "Should be 'diffai' (actual package name)"
    print_info "Check: .github/workflows/*.yml files"
    exit 1
fi

# Clear trap
trap - ERR

print_step "✅ ALL CI CHECKS PASSED!"
print_info "✅ No Python binding dependencies found"
print_info "✅ Package names correct for release workflow"
print_info "✅ Ready for GitHub Actions push"