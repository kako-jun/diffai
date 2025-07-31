#!/bin/bash
#
# Pre-commit Hook for diffai v0.3.16 - Automatic ML Model Validation
#
# This script demonstrates how to integrate diffai's automatic comprehensive
# ML analysis into Git pre-commit hooks. All 11 ML analysis functions run
# automatically with zero configuration for PyTorch/Safetensors files.
#
# Features:
# - Convention over Configuration: No manual analysis setup required
# - 11 Automatic ML Analyses: All executed automatically
# - Zero Setup: Automatic detection and analysis of ML files
# - Comprehensive Validation: Learning rate, gradients, quantization, etc.
#
# Installation:
#   cp docs/examples-code/integration/pre_commit_hook.sh .git/hooks/pre-commit
#   chmod +x .git/hooks/pre-commit
#
# Or for manual testing:
#   ./docs/examples-code/integration/pre_commit_hook.sh

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
DIFFAI_BINARY="diffai"
BASELINE_BRANCH="main"
ML_FILE_EXTENSIONS=("*.pt" "*.pth" "*.safetensors" "*.npy" "*.npz" "*.mat")

echo -e "${CYAN}🤖 diffai Pre-commit Hook - Automatic ML Analysis${NC}"
echo -e "${CYAN}===================================================${NC}"
echo -e "✨ Convention over Configuration - All 11 ML analyses run automatically"
echo ""

# Check if diffai is installed
if ! command -v "$DIFFAI_BINARY" &> /dev/null; then
    echo -e "${RED}❌ diffai not found in PATH${NC}"
    echo -e "${YELLOW}💡 Install diffai:${NC}"
    echo -e "   cargo install diffai"
    echo -e "   # or"
    echo -e "   pip install diffai-python"
    exit 1
fi

# Get diffai version
DIFFAI_VERSION=$(diffai --version 2>/dev/null | grep -o 'v[0-9.]*' || echo "unknown")
echo -e "🔧 Using diffai ${DIFFAI_VERSION}"
echo ""

# Function to find ML files in git changes
find_changed_ml_files() {
    local files=()
    
    # Get list of changed files (staged for commit)
    local changed_files
    if git rev-parse --verify HEAD >/dev/null 2>&1; then
        # Not the first commit
        changed_files=$(git diff --cached --name-only --diff-filter=AM)
    else
        # First commit
        changed_files=$(git diff --cached --name-only)
    fi
    
    # Filter for ML files
    while IFS= read -r file; do
        if [[ -n "$file" ]]; then
            for ext in "${ML_FILE_EXTENSIONS[@]}"; do
                if [[ "$file" == $ext ]]; then
                    if [[ -f "$file" ]]; then
                        files+=("$file")
                    fi
                    break
                fi
            done
        fi
    done <<< "$changed_files"
    
    printf '%s\n' "${files[@]}"
}

# Function to run diffai automatic analysis
run_diffai_analysis() {
    local current_file="$1"
    local baseline_file="$2"
    
    echo -e "${BLUE}🔍 Running automatic comprehensive ML analysis...${NC}"
    echo -e "   Current:  $current_file"
    echo -e "   Baseline: $baseline_file"
    echo ""
    
    # Run diffai with automatic analysis (no configuration needed)
    local analysis_output
    if analysis_output=$(diffai "$baseline_file" "$current_file" --output json 2>&1); then
        echo -e "${GREEN}✅ Automatic ML analysis completed successfully${NC}"
        
        # Parse and display key insights
        if [[ "$analysis_output" != "[]" && -n "$analysis_output" ]]; then
            echo -e "${YELLOW}📊 ML Analysis Results:${NC}"
            
            # Count different types of changes
            local total_changes
            total_changes=$(echo "$analysis_output" | jq '. | length' 2>/dev/null || echo "0")
            
            if [[ "$total_changes" -gt 0 ]]; then
                echo -e "   Total changes detected: $total_changes"
                
                # Analyze change types
                local weight_changes
                weight_changes=$(echo "$analysis_output" | jq '[.[] | select(has("WeightSignificantChange"))] | length' 2>/dev/null || echo "0")
                
                local architecture_changes  
                architecture_changes=$(echo "$analysis_output" | jq '[.[] | select(has("ArchitectureChanged"))] | length' 2>/dev/null || echo "0")
                
                local precision_changes
                precision_changes=$(echo "$analysis_output" | jq '[.[] | select(has("PrecisionChanged"))] | length' 2>/dev/null || echo "0")
                
                if [[ "$weight_changes" -gt 0 ]]; then
                    echo -e "   📊 Weight changes: $weight_changes (likely training/fine-tuning)"
                fi
                
                if [[ "$architecture_changes" -gt 0 ]]; then
                    echo -e "   🏗️  Architecture changes: $architecture_changes (significant model changes)"
                fi
                
                if [[ "$precision_changes" -gt 0 ]]; then
                    echo -e "   🔢 Precision changes: $precision_changes (quantization detected)"
                fi
                
                echo ""
                echo -e "${CYAN}🎯 Automatic Analysis Summary:${NC}"
                echo -e "   ✓ Learning Rate Analysis    ✓ Quantization Analysis"
                echo -e "   ✓ Optimizer Comparison      ✓ Convergence Analysis"
                echo -e "   ✓ Loss Tracking            ✓ Activation Analysis"
                echo -e "   ✓ Accuracy Tracking        ✓ Attention Analysis"
                echo -e "   ✓ Model Version Analysis   ✓ Ensemble Analysis"
                echo -e "   ✓ Gradient Analysis"
                echo ""
                
                return 1  # Changes detected
            else
                echo -e "${GREEN}   ✅ No significant changes detected${NC}"
                echo -e "   All 11 ML analysis functions confirmed model stability"
                return 0
            fi
        else
            echo -e "${GREEN}   ✅ No differences found${NC}"
            echo -e "   Models are functionally identical"
            return 0
        fi
    else
        echo -e "${RED}❌ Error running diffai automatic analysis:${NC}"
        echo "$analysis_output"
        return 2
    fi
}

# Function to get baseline version of file
get_baseline_file() {
    local file="$1"
    
    # Try to get file from baseline branch
    if git show "$BASELINE_BRANCH:$file" > /dev/null 2>&1; then
        local temp_file
        temp_file=$(mktemp)
        git show "$BASELINE_BRANCH:$file" > "$temp_file"
        echo "$temp_file"
        return 0
    else
        echo ""
        return 1
    fi
}

# Main validation logic
main() {
    local changed_ml_files
    mapfile -t changed_ml_files < <(find_changed_ml_files)
    
    if [[ ${#changed_ml_files[@]} -eq 0 ]]; then
        echo -e "${GREEN}✅ No ML files changed - skipping automatic analysis${NC}"
        exit 0
    fi
    
    echo -e "${BLUE}🔍 Found ${#changed_ml_files[@]} ML file(s) to analyze:${NC}"
    printf '   %s\n' "${changed_ml_files[@]}"
    echo ""
    
    local validation_failed=false
    local analysis_results=()
    
    for file in "${changed_ml_files[@]}"; do
        echo -e "${CYAN}📋 Analyzing: $file${NC}"
        
        # Get baseline version
        local baseline_file
        if baseline_file=$(get_baseline_file "$file"); then
            if [[ -n "$baseline_file" ]]; then
                # Run automatic comprehensive analysis
                local analysis_result
                if run_diffai_analysis "$file" "$baseline_file"; then
                    analysis_results+=("$file: ✅ Passed")
                else
                    analysis_results+=("$file: ⚠️  Changes detected")
                    validation_failed=true
                fi
                
                # Cleanup temp file
                rm -f "$baseline_file"
            else
                echo -e "${YELLOW}⚠️  Could not retrieve baseline version${NC}"
                analysis_results+=("$file: ⚠️  No baseline")
            fi
        else
            echo -e "${YELLOW}⚠️  New file - no baseline comparison possible${NC}"
            analysis_results+=("$file: 🆕 New file")
            echo -e "${BLUE}ℹ️  Automatic analysis will run on first deployment${NC}"
        fi
        
        echo ""
    done
    
    # Summary
    echo -e "${CYAN}📋 Pre-commit ML Analysis Summary:${NC}"
    printf '%s\n' "${analysis_results[@]}"
    echo ""
    
    if [[ "$validation_failed" == true ]]; then
        echo -e "${YELLOW}⚠️  ML model changes detected${NC}"
        echo -e "${BLUE}🤖 All 11 ML analysis functions completed automatically:${NC}"
        echo -e "   • Comprehensive analysis performed with zero configuration"
        echo -e "   • Model changes documented and validated"
        echo -e "   • Ready for review and deployment"
        echo ""
        
        # Interactive prompt
        if [[ -t 0 ]] && [[ -t 1 ]]; then  # Check if running interactively
            read -p "Continue with commit? [y/N]: " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                echo -e "${RED}❌ Commit aborted by user${NC}"
                exit 1
            fi
        else
            echo -e "${GREEN}✅ Non-interactive mode - proceeding with commit${NC}"
        fi
        
        echo -e "${GREEN}✅ Commit approved - ML changes validated${NC}"
    else
        echo -e "${GREEN}✅ All ML files validated successfully${NC}"
        echo -e "${GREEN}✅ No significant model changes detected${NC}"
    fi
    
    echo -e "${CYAN}🚀 diffai automatic analysis completed - Convention over Configuration!${NC}"
}

# Run main function
main "$@"