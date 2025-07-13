#!/bin/bash
set -euo pipefail

# Find the project root directory (where Cargo.toml exists)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Change to project root
cd "$PROJECT_ROOT"

# Exactly match GitHub Actions CI environment
export CARGO_TERM_COLOR=always
export RUST_BACKTRACE=1

# Stricter error handling to match CI
trap 'echo "Error occurred on line $LINENO. Exit code: $?" >&2' ERR

echo "Running complete CI simulation locally (matching GitHub Actions exactly)..."
echo "Project root: $PROJECT_ROOT"

echo "Step 1: Check formatting"
cargo fmt --all --check

echo "Step 2: Run Clippy"
cargo clippy --workspace --all-targets --all-features -- -D warnings

echo "Step 3: Build"
cargo build --workspace --verbose

echo "Step 4: Run tests"
cargo test --workspace --verbose

echo "Step 5: Release build test (matching GitHub Actions)"
# Test release build like GitHub Actions does - this will catch dependency version mismatches
echo "Testing release build for diffai CLI (this catches crates.io dependency issues)..."
cargo build --release --package diffai --verbose
echo "Testing release build for diffai-core..."
cargo build --release --package diffai-core --verbose
echo "Release build successful - performance optimizations applied"

echo "Step 5.1: Check for dependency version consistency"
# Check that workspace dependencies match individual package dependencies
echo "Verifying dependency versions are consistent across workspace..."
WORKSPACE_VERSION=$(grep 'version =' Cargo.toml | head -1 | sed 's/.*"\(.*\)".*/\1/')
CLI_DEP_VERSION=$(grep "diffai-core.*version.*=" diffai-cli/Cargo.toml | sed 's/.*version = "\(.*\)".*/\1/')

if [ "$WORKSPACE_VERSION" != "$CLI_DEP_VERSION" ]; then
    echo "ERROR: Version mismatch detected!" >&2
    echo "Workspace version: $WORKSPACE_VERSION" >&2
    echo "diffai-cli dependency version: $CLI_DEP_VERSION" >&2
    echo "This will cause GitHub Actions to fail during release build." >&2
    exit 1
fi
echo "✅ Dependency versions are consistent ($WORKSPACE_VERSION)"

echo "Step 5.2: Simulate GitHub Actions crates.io dependency resolution"
# Create a temporary directory and test building without local path dependencies
TEMP_TEST_DIR=$(mktemp -d)
trap 'rm -rf "$TEMP_TEST_DIR"' EXIT

echo "Simulating crates.io dependency resolution (like GitHub Actions)..."
# This simulates what happens in GitHub Actions where path dependencies aren't available
cd "$TEMP_TEST_DIR"
cargo init --name test-crates-io-deps --bin
echo "Testing if diffai-core version $CLI_DEP_VERSION would be available from crates.io..."

# Check if the required version exists on crates.io
if ! cargo search diffai-core | grep -q "diffai-core.*$CLI_DEP_VERSION"; then
    echo "WARNING: diffai-core version $CLI_DEP_VERSION not found on crates.io" >&2
    echo "This means GitHub Actions will fail if it tries to download this version" >&2
    echo "You may need to publish diffai-core first, or this is a new version about to be released" >&2
else
    echo "✅ diffai-core version $CLI_DEP_VERSION is available on crates.io"
fi

cd "$PROJECT_ROOT"

echo "Step 6: Test core CLI functionality"

# Create temp directory for test files (like CI would)
TEST_DIR=$(mktemp -d)
trap 'rm -rf "$TEST_DIR"' EXIT

# Test basic JSON diff (must succeed)
echo '{"a": 1}' > "$TEST_DIR/test1.json"
echo '{"a": 2}' > "$TEST_DIR/test2.json"
if ! cargo run --bin diffai -- "$TEST_DIR/test1.json" "$TEST_DIR/test2.json" > /dev/null 2>&1; then
    echo "ERROR: Basic JSON diff test failed" >&2
    exit 1
fi

# Test YAML diff (must succeed)
echo 'name: old' > "$TEST_DIR/test1.yaml"
echo 'name: new' > "$TEST_DIR/test2.yaml"
if ! cargo run --bin diffai -- "$TEST_DIR/test1.yaml" "$TEST_DIR/test2.yaml" > /dev/null 2>&1; then
    echo "ERROR: YAML diff test failed" >&2
    exit 1
fi

# Test stdin processing (must succeed)
if ! echo '{"b": 1}' | cargo run --bin diffai -- - "$TEST_DIR/test1.json" > /dev/null 2>&1; then
    echo "ERROR: Stdin processing test failed" >&2
    exit 1
fi

# Additional tests to ensure exact CI parity
echo "Step 7: Additional strict checks"

# Ensure no warnings in release mode
if ! cargo build --release --workspace 2>&1 | grep -v "Finished" | grep -v "Compiling" | grep -v "Building" | grep -q .; then
    echo "Release build completed without warnings"
else
    echo "ERROR: Release build produced warnings" >&2
    exit 1
fi

# Check for any TODO or FIXME comments (optional but good practice)
if grep -r "TODO\|FIXME" --include="*.rs" "$PROJECT_ROOT" | grep -v "target/"; then
    echo "WARNING: Found TODO/FIXME comments in code"
fi

# Verify Cargo.lock is committed and up to date
if ! git diff --quiet Cargo.lock; then
    echo "ERROR: Cargo.lock has uncommitted changes" >&2
    exit 1
fi

# Check for large files that shouldn't be committed
if find "$PROJECT_ROOT" -type f -size +1M -not -path "$PROJECT_ROOT/target/*" -not -path "$PROJECT_ROOT/.git/*" | grep -q .; then
    echo "WARNING: Found files larger than 1MB"
    find "$PROJECT_ROOT" -type f -size +1M -not -path "$PROJECT_ROOT/target/*" -not -path "$PROJECT_ROOT/.git/*" -exec ls -lh {} \;
fi

echo "All CI steps completed successfully!"
echo "Ready to push to remote repository"