#!/bin/bash

# Version Consistency Checker for diffai
# Ensures all package versions are synchronized

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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

echo "🔍 Checking version consistency across all diffai packages..."

# Extract versions from different sources
get_cargo_version() {
    grep '^version = ' "$PROJECT_ROOT/Cargo.toml" | head -1 | sed 's/version = "\(.*\)"/\1/'
}

get_npm_version() {
    if [ -f "$PROJECT_ROOT/diffai-npm/package.json" ]; then
        node -p "require('$PROJECT_ROOT/diffai-npm/package.json').version" 2>/dev/null || echo "ERROR"
    else
        echo "NOT_FOUND"
    fi
}

get_python_version() {
    if [ -f "$PROJECT_ROOT/diffai-python/pyproject.toml" ]; then
        grep '^version = ' "$PROJECT_ROOT/diffai-python/pyproject.toml" | sed 's/version = "\(.*\)"/\1/' 2>/dev/null || echo "ERROR"
    else
        echo "NOT_FOUND"
    fi
}

get_changelog_version() {
    if [ -f "$PROJECT_ROOT/CHANGELOG.md" ]; then
        grep -m 1 '^## \[' "$PROJECT_ROOT/CHANGELOG.md" | sed 's/## \[\(.*\)\].*/\1/' 2>/dev/null || echo "NOT_FOUND"
    else
        echo "NOT_FOUND"
    fi
}

get_claude_md_version() {
    if [ -f "$PROJECT_ROOT/CLAUDE.md" ]; then
        grep -o 'v[0-9]\+\.[0-9]\+\.[0-9]\+' "$PROJECT_ROOT/CLAUDE.md" | head -1 | sed 's/v//' 2>/dev/null || echo "NOT_FOUND"
    else
        echo "NOT_FOUND"
    fi
}

# Get all versions
CARGO_VERSION=$(get_cargo_version)
NPM_VERSION=$(get_npm_version)
PYTHON_VERSION=$(get_python_version)
CHANGELOG_VERSION=$(get_changelog_version)
CLAUDE_VERSION=$(get_claude_md_version)

# Display versions
echo
print_status "Current versions:"
printf "  %-15s %s\n" "Cargo.toml:" "$CARGO_VERSION"
printf "  %-15s %s\n" "NPM package:" "$NPM_VERSION"
printf "  %-15s %s\n" "Python package:" "$PYTHON_VERSION"
printf "  %-15s %s\n" "CHANGELOG.md:" "$CHANGELOG_VERSION"
printf "  %-15s %s\n" "CLAUDE.md:" "$CLAUDE_VERSION"

# Check consistency
echo
ISSUES=0

if [ "$NPM_VERSION" != "NOT_FOUND" ] && [ "$NPM_VERSION" != "ERROR" ] && [ "$NPM_VERSION" != "$CARGO_VERSION" ]; then
    print_error "NPM version ($NPM_VERSION) doesn't match Cargo version ($CARGO_VERSION)"
    ISSUES=$((ISSUES + 1))
fi

if [ "$PYTHON_VERSION" != "NOT_FOUND" ] && [ "$PYTHON_VERSION" != "ERROR" ] && [ "$PYTHON_VERSION" != "$CARGO_VERSION" ]; then
    print_error "Python version ($PYTHON_VERSION) doesn't match Cargo version ($CARGO_VERSION)"
    ISSUES=$((ISSUES + 1))
fi

if [ "$CHANGELOG_VERSION" != "NOT_FOUND" ] && [ "$CHANGELOG_VERSION" != "$CARGO_VERSION" ]; then
    print_warning "CHANGELOG.md version ($CHANGELOG_VERSION) doesn't match Cargo version ($CARGO_VERSION)"
fi

if [ "$CLAUDE_VERSION" != "NOT_FOUND" ] && [ "$CLAUDE_VERSION" != "$CARGO_VERSION" ]; then
    print_warning "CLAUDE.md version ($CLAUDE_VERSION) doesn't match Cargo version ($CARGO_VERSION)"
fi

# Check for missing files
if [ "$NPM_VERSION" = "NOT_FOUND" ]; then
    print_warning "NPM package.json not found"
fi

if [ "$PYTHON_VERSION" = "NOT_FOUND" ]; then
    print_warning "Python pyproject.toml not found"
fi

if [ "$NPM_VERSION" = "ERROR" ]; then
    print_error "Error reading NPM package.json"
    ISSUES=$((ISSUES + 1))
fi

if [ "$PYTHON_VERSION" = "ERROR" ]; then
    print_error "Error reading Python pyproject.toml"
    ISSUES=$((ISSUES + 1))
fi

# Git tag check
if git rev-parse --git-dir > /dev/null 2>&1; then
    LATEST_TAG=$(git tag -l "v*" | sort -V | tail -1 | sed 's/v//')
    if [ -n "$LATEST_TAG" ]; then
        printf "  %-15s %s\n" "Latest git tag:" "$LATEST_TAG"
        if [ "$LATEST_TAG" != "$CARGO_VERSION" ]; then
            print_warning "Latest git tag (v$LATEST_TAG) doesn't match current version ($CARGO_VERSION)"
        fi
    fi
fi

# Summary
echo
if [ $ISSUES -eq 0 ]; then
    print_success "✅ All critical versions are consistent!"
    
    if [ "$NPM_VERSION" = "$CARGO_VERSION" ] && [ "$PYTHON_VERSION" = "$CARGO_VERSION" ]; then
        print_success "All package versions match: $CARGO_VERSION"
    fi
else
    print_error "❌ Found $ISSUES critical version inconsistencies"
    echo
    print_status "To fix version inconsistencies:"
    echo "  1. Update diffai-npm/package.json version to $CARGO_VERSION"
    echo "  2. Update diffai-python/pyproject.toml version to $CARGO_VERSION"
    echo "  3. Update CHANGELOG.md with new version entry"
    echo "  4. Update CLAUDE.md version references"
    exit 1
fi

# Additional checks
echo
print_status "Additional checks:"

# Check if version follows semantic versioning
if [[ ! $CARGO_VERSION =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    print_warning "Version doesn't follow semantic versioning pattern (x.y.z): $CARGO_VERSION"
fi

# Check if this is a development version
if [[ $CARGO_VERSION =~ -dev|-alpha|-beta|-rc ]]; then
    print_warning "This appears to be a development version: $CARGO_VERSION"
fi

# Check if git workspace is clean
if git rev-parse --git-dir > /dev/null 2>&1; then
    if ! git diff-index --quiet HEAD --; then
        print_warning "Git workspace has uncommitted changes"
    else
        print_success "Git workspace is clean"
    fi
fi

echo
print_status "Version check completed"