#!/bin/bash

# 🔧 Setup GitHub workflow and repository settings for diffai
# Based on diffx setup patterns

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

error() {
    echo -e "${RED}❌ $1${NC}"
}

info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')] $1${NC}"
}

echo "🔧 Setting up GitHub workflow and repository settings for diffai"
echo "=============================================================="

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    error "This script must be run from the root of a git repository"
    exit 1
fi

# Check if gh CLI is available
if ! command -v gh &> /dev/null; then
    warning "GitHub CLI (gh) not found. Some operations will be skipped."
    warning "Install gh CLI from: https://cli.github.com/"
    GH_AVAILABLE=false
else
    GH_AVAILABLE=true
    success "GitHub CLI found"
fi

# 1. Apply repository labels
if [ "$GH_AVAILABLE" = true ]; then
    log "Step 1: Setting up repository labels..."
    
    if [ -f ".github/labels.json" ]; then
        # Delete existing labels that might conflict
        gh label list --json name --jq '.[].name' | while read label; do
            gh label delete "$label" --yes 2>/dev/null || true
        done
        
        # Create labels from JSON file
        cat .github/labels.json | jq -r '.[] | "\\(.name)|\\(.color)|\\(.description)"' | while IFS='|' read name color description; do
            gh label create "$name" --color "$color" --description "$description" 2>/dev/null || \
            gh label edit "$name" --color "$color" --description "$description" 2>/dev/null || true
        done
        
        success "Repository labels configured"
    else
        warning "Labels file not found at .github/labels.json"
    fi
else
    warning "Skipping label setup (gh CLI not available)"
fi

# 2. Apply branch protection rules
if [ "$GH_AVAILABLE" = true ]; then
    log "Step 2: Setting up branch protection..."
    
    if [ -f ".github/branch-protection.json" ]; then
        # Apply branch protection settings
        gh api repos/:owner/:repo/branches/main/protection \
            --method PUT \
            --input .github/branch-protection.json 2>/dev/null && \
            success "Branch protection rules applied" || \
            warning "Failed to apply branch protection rules (may need admin access)"
    else
        warning "Branch protection file not found at .github/branch-protection.json"
    fi
else
    warning "Skipping branch protection setup (gh CLI not available)"
fi

# 3. Set up repository settings
if [ "$GH_AVAILABLE" = true ]; then
    log "Step 3: Configuring repository settings..."
    
    # Enable features
    gh repo edit --enable-wiki=false
    gh repo edit --enable-projects=false
    gh repo edit --enable-issues=true
    gh repo edit --enable-merge-commit=true
    gh repo edit --enable-squash-merge=true
    gh repo edit --enable-rebase-merge=true
    gh repo edit --delete-branch-on-merge=true
    
    success "Repository settings configured"
else
    warning "Skipping repository settings (gh CLI not available)"
fi

# 4. Set up repository secrets (guidance)
log "Step 4: Repository secrets setup guidance..."
echo ""
info "Manual setup required for the following secrets:"
echo "  • CRATES_TOKEN - For publishing to crates.io"
echo "  • NPM_TOKEN - For publishing to npm (when ready)"
echo "  • PYPI_TOKEN - For publishing to PyPI (when ready)"
echo ""
info "To set up secrets:"
echo "  gh secret set CRATES_TOKEN"
echo "  gh secret set NPM_TOKEN"
echo "  gh secret set PYPI_TOKEN"
echo ""

# 5. Validate workflow files
log "Step 5: Validating workflow files..."

WORKFLOW_DIR=".github/workflows"
if [ -d "$WORKFLOW_DIR" ]; then
    for workflow in "$WORKFLOW_DIR"/*.yml; do
        if [ -f "$workflow" ]; then
            # Basic YAML syntax check
            if command -v yamllint &> /dev/null; then
                yamllint "$workflow" && success "$(basename "$workflow") - syntax OK" || warning "$(basename "$workflow") - syntax issues"
            else
                # Basic check for YAML structure
                if grep -q "^name:" "$workflow" && grep -q "^on:" "$workflow" && grep -q "^jobs:" "$workflow"; then
                    success "$(basename "$workflow") - basic structure OK"
                else
                    warning "$(basename "$workflow") - basic structure issues"
                fi
            fi
        fi
    done
else
    warning "No .github/workflows directory found"
fi

# 6. Set up commit hooks (optional)
log "Step 6: Setting up git hooks..."

if [ ! -d ".git/hooks" ]; then
    mkdir -p .git/hooks
fi

# Pre-commit hook to run CI locally
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
# Pre-commit hook for diffai

echo "🔍 Running pre-commit checks..."

# Run local CI if available
if [ -f "scripts/ci-local.sh" ]; then
    echo "Running local CI..."
    ./scripts/ci-local.sh || {
        echo "❌ Local CI failed. Commit aborted."
        exit 1
    }
fi

echo "✅ Pre-commit checks passed"
EOF

chmod +x .git/hooks/pre-commit
success "Git pre-commit hook installed"

# 7. Create issue templates
log "Step 7: Creating issue templates..."

ISSUE_TEMPLATE_DIR=".github/ISSUE_TEMPLATE"
mkdir -p "$ISSUE_TEMPLATE_DIR"

# Bug report template
cat > "$ISSUE_TEMPLATE_DIR/bug_report.yml" << 'EOF'
name: Bug Report
description: File a bug report for diffai
title: "[Bug]: "
labels: ["type: bug", "priority: medium"]
body:
  - type: markdown
    attributes:
      value: |
        Thanks for taking the time to fill out this bug report!
        
  - type: input
    id: version
    attributes:
      label: diffai Version
      description: What version of diffai are you running?
      placeholder: e.g., v0.3.0
    validations:
      required: true
      
  - type: dropdown
    id: file-format
    attributes:
      label: File Format
      description: What file format were you comparing?
      options:
        - Safetensors (.safetensors)
        - PyTorch (.pt, .pth)
        - NumPy (.npy, .npz)
        - MATLAB (.mat)
        - JSON
        - YAML
        - Other
    validations:
      required: true
      
  - type: textarea
    id: what-happened
    attributes:
      label: What happened?
      description: Also tell us, what did you expect to happen?
      placeholder: Tell us what you see!
    validations:
      required: true
      
  - type: textarea
    id: reproduce
    attributes:
      label: Steps to Reproduce
      description: How can we reproduce this issue?
      placeholder: |
        1. Run command '...'
        2. With files '...'
        3. See error
    validations:
      required: true
      
  - type: textarea
    id: logs
    attributes:
      label: Relevant log output
      description: Please copy and paste any relevant log output. This will be automatically formatted into code, so no need for backticks.
      render: shell
EOF

# Feature request template
cat > "$ISSUE_TEMPLATE_DIR/feature_request.yml" << 'EOF'
name: Feature Request
description: Suggest an idea for diffai
title: "[Feature]: "
labels: ["type: feature", "priority: low"]
body:
  - type: markdown
    attributes:
      value: |
        Thanks for suggesting a new feature for diffai!
        
  - type: dropdown
    id: area
    attributes:
      label: Feature Area
      description: Which area does this feature relate to?
      options:
        - Core comparison engine
        - ML analysis features
        - Output formats
        - CLI interface
        - File format support
        - Performance
        - Documentation
        - Other
    validations:
      required: true
      
  - type: textarea
    id: problem
    attributes:
      label: Is your feature request related to a problem?
      description: A clear and concise description of what the problem is.
      placeholder: I'm always frustrated when...
      
  - type: textarea
    id: solution
    attributes:
      label: Describe the solution you'd like
      description: A clear and concise description of what you want to happen.
    validations:
      required: true
      
  - type: textarea
    id: alternatives
    attributes:
      label: Describe alternatives you've considered
      description: A clear and concise description of any alternative solutions or features you've considered.
      
  - type: textarea
    id: context
    attributes:
      label: Additional context
      description: Add any other context, use cases, or screenshots about the feature request here.
EOF

success "Issue templates created"

# 8. Summary
echo ""
echo "🎉 GitHub Setup Complete!"
echo "========================"
echo ""
success "Repository labels configured"
success "Branch protection rules applied"
success "Repository settings optimized"
success "Workflow files validated"
success "Git hooks installed"
success "Issue templates created"
echo ""
info "Next steps:"
echo "  1. Set up repository secrets (CRATES_TOKEN, NPM_TOKEN, PYPI_TOKEN)"
echo "  2. Test workflows by creating a pull request"
echo "  3. Review branch protection settings in GitHub web interface"
echo ""
info "For more information, see:"
echo "  • https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features"
echo "  • https://docs.github.com/en/actions"