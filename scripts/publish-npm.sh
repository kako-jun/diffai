#!/bin/bash

# NPM Package Publishing Script for diffai
# Publishes the diffai npm package to npmjs.org

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
NPM_DIR="$PROJECT_ROOT/diffai-npm"

echo "🚀 Publishing diffai npm package..."

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "❌ npm is not installed. Please install Node.js and npm."
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "$NPM_DIR/package.json" ]; then
    echo "❌ package.json not found in $NPM_DIR"
    exit 1
fi

cd "$NPM_DIR"

# Get version from package.json
VERSION=$(node -p "require('./package.json').version")
echo "📦 Package version: $VERSION"

# Verify package contents
echo "📋 Verifying package contents..."
npm pack --dry-run

# Check if user is logged in to npm
if ! npm whoami > /dev/null 2>&1; then
    echo "❌ Not logged in to npm. Please run 'npm login' first."
    exit 1
fi

NPM_USER=$(npm whoami)
echo "👤 Publishing as: $NPM_USER"

# Run tests before publishing
echo "🧪 Running tests..."
if [ -f "test.js" ]; then
    node test.js
else
    echo "⚠️  No tests found, skipping..."
fi

# Build/prepare package
echo "🔨 Preparing package..."

# Create bin directory if it doesn't exist
mkdir -p bin

# Lint package.json
echo "🔍 Validating package.json..."
npm pkg lint

# Check for security vulnerabilities
echo "🔒 Checking for vulnerabilities..."
npm audit --audit-level moderate || echo "⚠️  Security audit completed with warnings"

# Publish package
echo "📢 Publishing to npm..."

# Ask for confirmation
read -p "🤔 Are you sure you want to publish diffai@$VERSION? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Publication cancelled."
    exit 1
fi

# Publish with public access
npm publish --access public

if [ $? -eq 0 ]; then
    echo "✅ Successfully published diffai@$VERSION to npm!"
    echo "📦 Package URL: https://www.npmjs.com/package/diffai"
    echo "💾 Install with: npm install diffai"
    
    # Tag the git commit
    cd "$PROJECT_ROOT"
    if git rev-parse --git-dir > /dev/null 2>&1; then
        echo "🏷️  Creating git tag npm-v$VERSION..."
        git tag "npm-v$VERSION" -m "npm package release v$VERSION"
        echo "📤 Push tag with: git push origin npm-v$VERSION"
    fi
else
    echo "❌ Failed to publish package"
    exit 1
fi