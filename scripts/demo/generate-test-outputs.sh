#!/bin/bash
set -euo pipefail

# Generate Test Outputs for Documentation Verification
# This script captures actual diffai outputs to demonstrate functionality

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUTPUT_DIR="$PROJECT_ROOT/demo-outputs"

echo "🎬 Generating diffai demonstration outputs..."
echo "📁 Output directory: $OUTPUT_DIR"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build diffai if needed
if [ ! -f "$PROJECT_ROOT/target/release/diffai" ]; then
    echo "🔨 Building diffai..."
    cd "$PROJECT_ROOT"
    cargo build --release
fi

DIFFAI_BIN="$PROJECT_ROOT/target/release/diffai"

# Check if test fixtures exist
FIXTURE_DIR="$PROJECT_ROOT/tests/fixtures"
if [ ! -d "$FIXTURE_DIR" ]; then
    echo "❌ Test fixtures not found at $FIXTURE_DIR"
    echo "   Please run tests first to generate fixtures"
    exit 1
fi

echo "✅ Using diffai binary: $DIFFAI_BIN"
echo "✅ Using fixtures from: $FIXTURE_DIR"

# Function to run diffai and capture output
run_demo() {
    local name="$1"
    local description="$2"
    shift 2
    local args=("$@")
    
    echo ""
    echo "🔍 Demo: $name"
    echo "📝 Description: $description"
    echo "💻 Command: diffai ${args[*]}"
    echo "─────────────────────────────────────────────"
    
    local output_file="$OUTPUT_DIR/${name}.txt"
    
    # Write header to output file
    {
        echo "# diffai Demo Output: $name"
        echo "# Description: $description"
        echo "# Command: diffai ${args[*]}"
        echo "# Generated: $(date)"
        echo "# Version: $(cd "$PROJECT_ROOT" && cargo metadata --no-deps --format-version 1 | jq -r '.packages[0].version')"
        echo ""
        echo "## Command Output:"
        echo ""
    } > "$output_file"
    
    # Run diffai and capture output
    if "$DIFFAI_BIN" "${args[@]}" >> "$output_file" 2>&1; then
        echo "✅ Success - Output saved to: ${output_file##*/}"
    else
        echo "⚠️  Command failed but output captured - Check: ${output_file##*/}"
    fi
    
    # Show preview of output
    echo ""
    echo "📋 Output preview (first 10 lines):"
    head -n 15 "$output_file" | tail -n 10
    echo "   ... (full output in $output_file)"
}

# Demo 1: Basic JSON comparison
if [ -f "$FIXTURE_DIR/simple1.json" ] && [ -f "$FIXTURE_DIR/simple2.json" ]; then
    run_demo "basic-json-diff" \
        "Basic JSON file comparison showing added/removed/modified fields" \
        "$FIXTURE_DIR/simple1.json" "$FIXTURE_DIR/simple2.json"
fi

# Demo 2: ML model comparison
if [ -f "$FIXTURE_DIR/ml_models/simple_base.safetensors" ] && [ -f "$FIXTURE_DIR/ml_models/simple_modified.safetensors" ]; then
    run_demo "ml-model-basic" \
        "Basic ML model comparison between SafeTensors files" \
        "$FIXTURE_DIR/ml_models/simple_base.safetensors" "$FIXTURE_DIR/ml_models/simple_modified.safetensors"
        
    run_demo "ml-model-stats" \
        "ML model comparison with statistical analysis" \
        "$FIXTURE_DIR/ml_models/simple_base.safetensors" "$FIXTURE_DIR/ml_models/simple_modified.safetensors" \
        "--stats"
        
    run_demo "ml-model-architecture" \
        "ML model architecture comparison" \
        "$FIXTURE_DIR/ml_models/simple_base.safetensors" "$FIXTURE_DIR/ml_models/simple_modified.safetensors" \
        "--architecture-comparison"
fi

# Demo 3: Verbose output
if [ -f "$FIXTURE_DIR/simple1.json" ] && [ -f "$FIXTURE_DIR/simple2.json" ]; then
    run_demo "verbose-mode" \
        "Verbose mode showing detailed analysis process" \
        "$FIXTURE_DIR/simple1.json" "$FIXTURE_DIR/simple2.json" \
        "--verbose"
fi

# Demo 4: Different output formats
if [ -f "$FIXTURE_DIR/simple1.json" ] && [ -f "$FIXTURE_DIR/simple2.json" ]; then
    run_demo "output-json" \
        "JSON output format for programmatic consumption" \
        "$FIXTURE_DIR/simple1.json" "$FIXTURE_DIR/simple2.json" \
        "--output" "json"
        
    run_demo "output-yaml" \
        "YAML output format for human-readable structured data" \
        "$FIXTURE_DIR/simple1.json" "$FIXTURE_DIR/simple2.json" \
        "--output" "yaml"
fi

# Demo 5: Help output
run_demo "help-output" \
    "Complete CLI help showing all available options" \
    "--help"

# Demo 6: Version info
run_demo "version-info" \
    "Version information" \
    "--version"

echo ""
echo "🎉 Demo generation complete!"
echo "📁 All outputs saved to: $OUTPUT_DIR"
echo ""
echo "📋 Generated demos:"
ls -la "$OUTPUT_DIR"/*.txt | while read -r line; do
    filename=$(echo "$line" | awk '{print $NF}')
    size=$(echo "$line" | awk '{print $5}')
    echo "   • $(basename "$filename") ($size bytes)"
done

echo ""
echo "💡 You can now review these files to see exactly how diffai works!"
echo "   Each file shows the command used and the actual output produced."