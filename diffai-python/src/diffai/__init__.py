"""
diffai - AI/ML specialized diff tool for deep tensor comparison and analysis

This package provides a Python wrapper around the diffai Rust binary,
following the same pattern as ruff for optimal performance and reliability.
"""

# No backward compatibility imports

import json
import subprocess
import sys
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

# Version is now managed dynamically from pyproject.toml
# This prevents hardcoded version mismatches during releases
try:
    from importlib.metadata import version
    __version__ = version("diffai-python")
except ImportError:
    # Fallback for Python < 3.8
    try:
        import pkg_resources
        __version__ = pkg_resources.get_distribution("diffai-python").version
    except Exception:
        __version__ = "unknown"

class OutputFormat(Enum):
    """Supported output formats for diffai results."""
    CLI = "cli"
    JSON = "json"
    YAML = "yaml"

@dataclass
class DiffOptions:
    """Configuration options for diffai analysis."""
    
    # Basic options
    input_format: Optional[str] = None
    output_format: Optional[OutputFormat] = None
    recursive: bool = False
    verbose: bool = False
    path: Optional[str] = None
    ignore_keys_regex: Optional[str] = None
    epsilon: Optional[float] = None
    array_id_key: Optional[str] = None
    
    # ML analysis options
    show_layer_impact: bool = False
    quantization_analysis: bool = False
    sort_by_change_magnitude: bool = False
    stats: bool = False
    learning_progress: bool = False
    convergence_analysis: bool = False
    anomaly_detection: bool = False
    gradient_analysis: bool = False
    memory_analysis: bool = False
    inference_speed_estimate: bool = False
    regression_test: bool = False
    alert_on_degradation: bool = False
    review_friendly: bool = False
    change_summary: bool = False
    deployment_readiness: bool = False
    architecture_comparison: bool = False
    param_efficiency_analysis: bool = False
    hyperparameter_impact: bool = False
    learning_rate_analysis: bool = False
    performance_impact_estimate: bool = False
    generate_report: bool = False
    markdown_output: bool = False
    include_charts: bool = False
    embedding_analysis: bool = False
    similarity_matrix: bool = False
    clustering_change: bool = False
    attention_analysis: bool = False
    head_importance: bool = False
    attention_pattern_diff: bool = False
    hyperparameter_comparison: bool = False
    learning_curve_analysis: bool = False
    statistical_significance: bool = False
    
    def to_args(self) -> List[str]:
        """Convert options to command line arguments."""
        args = []
        
        # Basic options
        if self.input_format:
            args.extend(["--format", self.input_format])
        if self.output_format:
            args.extend(["--output", self.output_format.value])
        if self.recursive:
            args.append("--recursive")
        if self.verbose:
            args.append("--verbose")
        if self.path:
            args.extend(["--path", self.path])
        if self.ignore_keys_regex:
            args.extend(["--ignore-keys-regex", self.ignore_keys_regex])
        if self.epsilon is not None:
            args.extend(["--epsilon", str(self.epsilon)])
        if self.array_id_key:
            args.extend(["--array-id-key", self.array_id_key])
            
        # ML analysis options
        if self.show_layer_impact:
            args.append("--show-layer-impact")
        if self.quantization_analysis:
            args.append("--quantization-analysis")
        if self.sort_by_change_magnitude:
            args.append("--sort-by-change-magnitude")
        if self.stats:
            args.append("--stats")
        if self.learning_progress:
            args.append("--learning-progress")
        if self.convergence_analysis:
            args.append("--convergence-analysis")
        if self.anomaly_detection:
            args.append("--anomaly-detection")
        if self.gradient_analysis:
            args.append("--gradient-analysis")
        if self.memory_analysis:
            args.append("--memory-analysis")
        if self.inference_speed_estimate:
            args.append("--inference-speed-estimate")
        if self.regression_test:
            args.append("--regression-test")
        if self.alert_on_degradation:
            args.append("--alert-on-degradation")
        if self.review_friendly:
            args.append("--review-friendly")
        if self.change_summary:
            args.append("--change-summary")
        if self.deployment_readiness:
            args.append("--deployment-readiness")
        if self.architecture_comparison:
            args.append("--architecture-comparison")
        if self.param_efficiency_analysis:
            args.append("--param-efficiency-analysis")
        if self.hyperparameter_impact:
            args.append("--hyperparameter-impact")
        if self.learning_rate_analysis:
            args.append("--learning-rate-analysis")
        if self.performance_impact_estimate:
            args.append("--performance-impact-estimate")
        if self.generate_report:
            args.append("--generate-report")
        if self.markdown_output:
            args.append("--markdown-output")
        if self.include_charts:
            args.append("--include-charts")
        if self.embedding_analysis:
            args.append("--embedding-analysis")
        if self.similarity_matrix:
            args.append("--similarity-matrix")
        if self.clustering_change:
            args.append("--clustering-change")
        if self.attention_analysis:
            args.append("--attention-analysis")
        if self.head_importance:
            args.append("--head-importance")
        if self.attention_pattern_diff:
            args.append("--attention-pattern-diff")
        if self.hyperparameter_comparison:
            args.append("--hyperparameter-comparison")
        if self.learning_curve_analysis:
            args.append("--learning-curve-analysis")
        if self.statistical_significance:
            args.append("--statistical-significance")
            
        return args

class DiffaiError(Exception):
    """Base exception for diffai-related errors."""
    pass

class DiffResult:
    """Result from diffai analysis."""
    
    def __init__(self, raw_output: str, format_type: str = "cli", return_code: int = 0):
        self.raw_output = raw_output
        self.format_type = format_type
        self.return_code = return_code
        self._parsed_data = None
        
    @property
    def data(self) -> Any:
        """Get parsed data (JSON objects for JSON output, raw string otherwise)."""
        if self._parsed_data is None:
            if self.format_type == "json" and self.raw_output.strip():
                try:
                    self._parsed_data = json.loads(self.raw_output)
                except json.JSONDecodeError:
                    self._parsed_data = self.raw_output
            else:
                self._parsed_data = self.raw_output
        return self._parsed_data
    
    @property
    def is_json(self) -> bool:
        """True if result is in JSON format."""
        return self.format_type == "json" and isinstance(self.data, (dict, list))
    
    def __str__(self) -> str:
        return self.raw_output

def _find_diffai_binary() -> str:
    """Find the diffai binary, checking bundled location first."""
    # Check if bundled with package
    package_dir = Path(__file__).parent.parent.parent
    bundled_binary = package_dir / "diffai"
    
    if bundled_binary.exists() and bundled_binary.is_file():
        return str(bundled_binary)
    
    # Fallback to system PATH
    system_binary = shutil.which("diffai")
    if system_binary:
        return system_binary
    
    raise DiffaiError(
        "diffai binary not found. Please ensure diffai is installed or available in PATH."
    )

def diff(
    input1: str,
    input2: str,
    options: Optional[Union[DiffOptions, Dict[str, Any]]] = None,
    **kwargs
) -> DiffResult:
    """
    Compare two files using diffai.
    
    Args:
        input1: Path to first input file
        input2: Path to second input file
        options: DiffOptions object or dict of options
        **kwargs: Additional options as keyword arguments
        
    Returns:
        DiffResult object containing comparison results
    """
    # Handle different option formats
    if options is None:
        options = DiffOptions(**kwargs)
    elif isinstance(options, dict):
        combined_options = {**options, **kwargs}
        options = DiffOptions(**combined_options)
    elif kwargs:
        # Merge kwargs into existing DiffOptions
        option_dict = {
            field.name: getattr(options, field.name) 
            for field in options.__dataclass_fields__.values()
        }
        combined_options = {**option_dict, **kwargs}
        options = DiffOptions(**combined_options)
    
    try:
        binary_path = _find_diffai_binary()
        cmd = [binary_path] + options.to_args() + [input1, input2]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode != 0 and result.stderr:
            raise DiffaiError(f"diffai failed: {result.stderr}")
        
        format_type = options.output_format.value if options.output_format else "cli"
        return DiffResult(result.stdout, format_type, result.returncode)
        
    except FileNotFoundError:
        raise DiffaiError("diffai binary not found")
    except Exception as e:
        raise DiffaiError(f"Diff failed: {e}")

def main():
    """CLI entry point for the diffai command."""
    try:
        binary_path = _find_diffai_binary()
        # Forward all arguments to the binary
        result = subprocess.run([binary_path] + sys.argv[1:])
        sys.exit(result.returncode)
    except DiffaiError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

# Export main API
__all__ = [
    "diff",
    "DiffOptions",
    "DiffResult",
    "OutputFormat",
    "DiffaiError",
    "__version__",
    "main",
]