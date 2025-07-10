#!/bin/bash

# Unified Publishing Script for diffai
# Publishes both npm and PyPI packages with coordinated versioning

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🚀 Publishing all diffai packages..."

# Color codes for better output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    print_error "Not in a git repository"
    exit 1
fi

# Check for uncommitted changes
if ! git diff-index --quiet HEAD --; then
    print_error "You have uncommitted changes. Please commit or stash them first."
    exit 1
fi

# Get version from Cargo.toml (main version source)
CARGO_VERSION=$(grep '^version = ' "$PROJECT_ROOT/Cargo.toml" | head -1 | sed 's/version = "\(.*\)"/\1/')
print_status "Main version (Cargo.toml): $CARGO_VERSION"

# Check npm package version
NPM_VERSION=$(node -p "require('$PROJECT_ROOT/diffai-npm/package.json').version" 2>/dev/null || echo "NOT_FOUND")
print_status "NPM version: $NPM_VERSION"

# Check Python package version
PYTHON_VERSION=$(grep '^version = ' "$PROJECT_ROOT/diffai-python/pyproject.toml" | sed 's/version = "\(.*\)"/\1/' 2>/dev/null || echo "NOT_FOUND")
print_status "Python version: $PYTHON_VERSION"

# Version consistency check
if [ "$NPM_VERSION" != "$CARGO_VERSION" ] || [ "$PYTHON_VERSION" != "$CARGO_VERSION" ]; then
    print_error "Version mismatch detected!"
    print_error "  Cargo:  $CARGO_VERSION"
    print_error "  NPM:    $NPM_VERSION"
    print_error "  Python: $PYTHON_VERSION"
    print_error "Please ensure all versions are synchronized before publishing."
    exit 1
fi

print_success "All package versions are synchronized: $CARGO_VERSION"

# Check if tag already exists
if git tag -l | grep -q "^v$CARGO_VERSION$"; then
    print_error "Git tag v$CARGO_VERSION already exists"
    exit 1
fi

# Pre-flight checks
print_status "Running pre-flight checks..."

# Check if Rust project builds
print_status "Building Rust project..."
if ! cargo build --release; then
    print_error "Rust build failed"
    exit 1
fi

# Check if tests pass
print_status "Running Rust tests..."
if ! cargo test; then
    print_error "Rust tests failed"
    exit 1
fi

# Check npm package
if [ -f "$PROJECT_ROOT/diffai-npm/package.json" ]; then
    print_status "Checking npm package..."
    cd "$PROJECT_ROOT/diffai-npm"
    npm pkg lint
    cd "$PROJECT_ROOT"
fi

# Check Python package
if [ -f "$PROJECT_ROOT/diffai-python/pyproject.toml" ]; then
    print_status "Checking Python package..."
    cd "$PROJECT_ROOT/diffai-python"
    if command -v python3 &> /dev/null; then
        python3 -m pip install --upgrade build twine
        python3 -m build
        python3 -m twine check dist/* || print_warning "Python package check had warnings"
        rm -rf dist/ build/ *.egg-info/
    fi
    cd "$PROJECT_ROOT"
fi

print_success "Pre-flight checks completed"

# Publishing options
echo
echo "📦 Publishing Options:"
echo "  1. Publish NPM package only"
echo "  2. Publish Python package only"
echo "  3. Publish both packages"
echo "  4. Create git tag only"
echo "  5. Cancel"
echo

read -p "Choose option (1-5): " -n 1 -r
echo

case $REPLY in
    1)
        print_status "Publishing NPM package..."
        "$SCRIPT_DIR/publish-npm.sh"
        ;;
    2)
        print_status "Publishing Python package..."
        "$SCRIPT_DIR/publish-pypi.sh"
        ;;
    3)
        print_status "Publishing both packages..."
        
        # Publish NPM first
        print_status "Step 1/2: Publishing NPM package..."
        "$SCRIPT_DIR/publish-npm.sh"
        
        print_status "Step 2/2: Publishing Python package..."
        "$SCRIPT_DIR/publish-pypi.sh"
        
        print_success "All packages published successfully!"
        ;;
    4)
        print_status "Creating git tag only..."
        git tag "v$CARGO_VERSION" -m "Release v$CARGO_VERSION"
        print_success "Created tag v$CARGO_VERSION"
        print_status "Push with: git push origin v$CARGO_VERSION"
        ;;
    5)
        print_status "Publishing cancelled"
        exit 0
        ;;
    *)
        print_error "Invalid option"
        exit 1
        ;;
esac

# Final steps
if [[ $REPLY =~ ^[123]$ ]]; then
    echo
    print_status "Post-publication steps:"
    
    # Create main release tag if it doesn't exist
    if ! git tag -l | grep -q "^v$CARGO_VERSION$"; then
        print_status "Creating main release tag..."
        git tag "v$CARGO_VERSION" -m "Release v$CARGO_VERSION"
        print_success "Created tag v$CARGO_VERSION"
    fi
    
    # Show next steps
    echo
    print_success "🎉 Publication completed!"
    echo
    print_status "Next steps:"
    echo "  1. Push git tags: git push origin --tags"
    echo "  2. Create GitHub release: https://github.com/diffai-team/diffai/releases/new"
    echo "  3. Update documentation if needed"
    echo "  4. Announce the release"
    echo
    print_status "Package URLs:"
    echo "  📦 NPM: https://www.npmjs.com/package/diffai"
    echo "  🐍 PyPI: https://pypi.org/project/diffai-python/"
    echo "  📖 GitHub: https://github.com/diffai-team/diffai"
fi