#!/bin/bash

# PyPI Package Publishing Script for diffai
# Publishes the diffai-python package to PyPI

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON_DIR="$PROJECT_ROOT/diffai-python"

echo "🐍 Publishing diffai-python package to PyPI..."

# Check if required tools are installed
for tool in python3 pip; do
    if ! command -v $tool &> /dev/null; then
        echo "❌ $tool is not installed. Please install Python and pip."
        exit 1
    fi
done

# Check if we're in the right directory
if [ ! -f "$PYTHON_DIR/pyproject.toml" ]; then
    echo "❌ pyproject.toml not found in $PYTHON_DIR"
    exit 1
fi

cd "$PYTHON_DIR"

# Get version from pyproject.toml
VERSION=$(python3 -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])" 2>/dev/null || \
          python3 -c "import toml; print(toml.load('pyproject.toml')['project']['version'])" 2>/dev/null || \
          grep '^version = ' pyproject.toml | sed 's/version = "\(.*\)"/\1/')

echo "📦 Package version: $VERSION"

# Install build dependencies
echo "📥 Installing build dependencies..."
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade build twine

# Install development dependencies
if grep -q "\[dev\]" pyproject.toml; then
    echo "🔧 Installing development dependencies..."
    python3 -m pip install -e ".[dev]"
fi

# Run tests if they exist
echo "🧪 Running tests..."
if [ -d "tests" ] && command -v pytest &> /dev/null; then
    python3 -m pytest
elif [ -f "test_integration.py" ]; then
    python3 test_integration.py
else
    echo "⚠️  No tests found, skipping..."
fi

# Lint code if tools are available
echo "🔍 Running code quality checks..."
if command -v black &> /dev/null; then
    python3 -m black --check src/ || echo "⚠️  Code formatting issues found"
fi

if command -v flake8 &> /dev/null; then
    python3 -m flake8 src/ || echo "⚠️  Linting issues found"
fi

if command -v mypy &> /dev/null; then
    python3 -m mypy src/ || echo "⚠️  Type checking issues found"
fi

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf dist/ build/ *.egg-info/

# Build package
echo "🔨 Building package..."
python3 -m build

# Check package
echo "🔍 Checking package..."
python3 -m twine check dist/*

# List package contents
echo "📋 Package contents:"
if command -v unzip &> /dev/null; then
    unzip -l dist/*.whl | head -20
else
    echo "  (install unzip to see wheel contents)"
fi

# Check if user is authenticated with PyPI
echo "👤 Checking PyPI authentication..."
if [ ! -f ~/.pypirc ] && [ -z "$TWINE_USERNAME" ] && [ -z "$TWINE_PASSWORD" ]; then
    echo "⚠️  No PyPI credentials found. You may need to authenticate during upload."
fi

# Test on TestPyPI first (optional)
read -p "🧪 Upload to TestPyPI first for testing? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "📤 Uploading to TestPyPI..."
    python3 -m twine upload --repository testpypi dist/*
    
    echo "✅ Uploaded to TestPyPI!"
    echo "🔗 Test installation: pip install --index-url https://test.pypi.org/simple/ diffai-python"
    
    read -p "✅ TestPyPI upload successful. Continue with production PyPI? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "❌ Production upload cancelled."
        exit 0
    fi
fi

# Final confirmation for production
read -p "🤔 Are you sure you want to publish diffai-python@$VERSION to PyPI? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Publication cancelled."
    exit 1
fi

# Upload to PyPI
echo "📢 Uploading to PyPI..."
python3 -m twine upload dist/*

if [ $? -eq 0 ]; then
    echo "✅ Successfully published diffai-python@$VERSION to PyPI!"
    echo "📦 Package URL: https://pypi.org/project/diffai-python/"
    echo "💾 Install with: pip install diffai-python"
    
    # Tag the git commit
    cd "$PROJECT_ROOT"
    if git rev-parse --git-dir > /dev/null 2>&1; then
        echo "🏷️  Creating git tag pypi-v$VERSION..."
        git tag "pypi-v$VERSION" -m "PyPI package release v$VERSION"
        echo "📤 Push tag with: git push origin pypi-v$VERSION"
    fi
    
    # Clean up build artifacts
    echo "🧹 Cleaning up..."
    rm -rf dist/ build/ *.egg-info/
    
else
    echo "❌ Failed to publish package"
    exit 1
fi