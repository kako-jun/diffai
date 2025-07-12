"""
Main diffai module providing Python interface to the diffai CLI tool.

This module provides a high-level Python API for the diffai command-line tool,
with support for type-safe configuration, structured results, and comprehensive
error handling.
"""

import json
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import importlib.metadata

try:
    __version__ = importlib.metadata.version("diffai-python")
except importlib.metadata.PackageNotFoundError:
    # Fallback for development
    __version__ = "0.2.9"


class OutputFormat(Enum):
    """Supported output formats for diffai results - exact mapping from CLI."""
    CLI = "cli"        # Default colored CLI output
    JSON = "json"      # Machine-readable JSON
    YAML = "yaml"      # Human-readable YAML
    UNIFIED = "unified"  # Unified diff format


@dataclass
class DiffOptions:
    """
    Configuration options for diffai analysis.
    
    Provides a type-safe way to configure all diffai analysis options
    including ML-specific analysis functions and output formatting.
    
    Based on actual CLI options from diffai-cli/src/main.rs
    """
    
    # Output and basic configuration
    output_format: Optional[OutputFormat] = None
    input_format: Optional[str] = None  # json, yaml, toml, ini, xml, csv, safetensors, pytorch, numpy, npz, matlab
    recursive: bool = False
    path: Optional[str] = None  # Filter by specific path
    ignore_keys_regex: Optional[str] = None  # Ignore keys matching regex
    epsilon: Optional[float] = None  # Tolerance for float comparisons
    array_id_key: Optional[str] = None  # Key for identifying array elements
    
    # Core analysis options
    stats: bool = False
    verbose: bool = False  # NEW: Added verbose mode
    show_layer_impact: bool = False
    quantization_analysis: bool = False
    sort_by_change_magnitude: bool = False
    
    # Enhanced ML analysis (Phase 3 implemented)
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
    risk_assessment: bool = False
    architecture_comparison: bool = False
    param_efficiency_analysis: bool = False
    hyperparameter_impact: bool = False
    learning_rate_analysis: bool = False
    deployment_readiness: bool = False
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
    
    # Phase 2 options
    hyperparameter_comparison: bool = False
    learning_curve_analysis: bool = False
    statistical_significance: bool = False
    
    # Additional options
    extra_args: List[str] = field(default_factory=list)
    
    def to_cli_args(self) -> List[str]:
        """Convert options to CLI arguments based on actual CLI structure."""
        args = []
        
        # Input format
        if self.input_format:
            args.extend(["--format", self.input_format])
        
        # Output format
        if self.output_format:
            args.extend(["--output", self.output_format.value])
        
        # String options
        if self.path:
            args.extend(["--path", self.path])
        if self.ignore_keys_regex:
            args.extend(["--ignore-keys-regex", self.ignore_keys_regex])
        if self.epsilon is not None:
            args.extend(["--epsilon", str(self.epsilon)])
        if self.array_id_key:
            args.extend(["--array-id-key", self.array_id_key])
        
        # Boolean flags - exact mapping from CLI
        flag_mapping = {
            "recursive": "--recursive",
            "stats": "--stats",
            "verbose": "--verbose",  # NEW: Added verbose mode
            "show_layer_impact": "--show-layer-impact",
            "quantization_analysis": "--quantization-analysis",
            "sort_by_change_magnitude": "--sort-by-change-magnitude",
            "learning_progress": "--learning-progress",
            "convergence_analysis": "--convergence-analysis",
            "anomaly_detection": "--anomaly-detection",
            "gradient_analysis": "--gradient-analysis",
            "memory_analysis": "--memory-analysis",
            "inference_speed_estimate": "--inference-speed-estimate",
            "regression_test": "--regression-test",
            "alert_on_degradation": "--alert-on-degradation",
            "review_friendly": "--review-friendly",
            "change_summary": "--change-summary",
            "risk_assessment": "--risk-assessment",
            "architecture_comparison": "--architecture-comparison",
            "param_efficiency_analysis": "--param-efficiency-analysis",
            "hyperparameter_impact": "--hyperparameter-impact",
            "learning_rate_analysis": "--learning-rate-analysis",
            "deployment_readiness": "--deployment-readiness",
            "performance_impact_estimate": "--performance-impact-estimate",
            "generate_report": "--generate-report",
            "markdown_output": "--markdown-output",
            "include_charts": "--include-charts",
            "embedding_analysis": "--embedding-analysis",
            "similarity_matrix": "--similarity-matrix",
            "clustering_change": "--clustering-change",
            "attention_analysis": "--attention-analysis",
            "head_importance": "--head-importance",
            "attention_pattern_diff": "--attention-pattern-diff",
            "hyperparameter_comparison": "--hyperparameter-comparison",
            "learning_curve_analysis": "--learning-curve-analysis",
            "statistical_significance": "--statistical-significance",
        }
        
        for option, flag in flag_mapping.items():
            if getattr(self, option):
                args.append(flag)
        
        # Extra arguments
        args.extend(self.extra_args)
        
        return args


class DiffaiError(Exception):
    """Base exception for diffai-related errors."""
    
    def __init__(self, message: str, exit_code: Optional[int] = None, stderr: Optional[str] = None):
        super().__init__(message)
        self.exit_code = exit_code
        self.stderr = stderr


class BinaryNotFoundError(DiffaiError):
    """Raised when the diffai binary cannot be found."""
    pass


class InvalidInputError(DiffaiError):
    """Raised when input files or arguments are invalid."""
    pass


class DiffResult:
    """
    Structured result from diffai analysis.
    
    Provides convenient access to diff results with automatic JSON parsing
    for structured data and raw text access for other formats.
    """
    
    def __init__(self, raw_output: str, exit_code: int = 0, format_type: str = "diffai"):
        self.raw_output = raw_output
        self.exit_code = exit_code
        self.format_type = format_type
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
    
    @property
    def changes(self) -> List[Dict[str, Any]]:
        """Get list of changes (for JSON output)."""
        if self.is_json and isinstance(self.data, list):
            return self.data
        return []
    
    @property
    def summary(self) -> Dict[str, Any]:
        """Get summary information (for JSON output)."""
        if self.is_json and isinstance(self.data, dict):
            return self.data
        return {}
    
    def __str__(self) -> str:
        """String representation of the result."""
        return self.raw_output
    
    def __repr__(self) -> str:
        """Detailed representation of the result."""
        return f"DiffResult(format={self.format_type}, exit_code={self.exit_code}, length={len(self.raw_output)})"


def _get_diffai_binary_path() -> str:
    """
    Get the path to the diffai binary.
    
    Checks for local installation first, then falls back to system PATH.
    """
    # Check for local installation (installed via pip)
    package_dir = Path(__file__).parent.parent.parent
    binary_name = "diffai.exe" if platform.system() == "Windows" else "diffai"
    local_binary_path = package_dir / "bin" / binary_name
    
    if local_binary_path.exists():
        return str(local_binary_path)
    
    # Check package-local bin directory
    package_bin = Path(__file__).parent.parent / "bin" / binary_name
    if package_bin.exists():
        return str(package_bin)
    
    # Fall back to system PATH
    return "diffai"


def verify_installation() -> Dict[str, Any]:
    """
    Verify that diffai is properly installed and accessible.
    
    Returns:
        Dict containing installation status and version information.
        
    Raises:
        BinaryNotFoundError: If diffai binary cannot be found or executed.
    """
    try:
        binary_path = _get_diffai_binary_path()
        result = subprocess.run(
            [binary_path, "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            version_output = result.stdout.strip()
            info: Dict[str, str] = {
                "binary_path": binary_path,
                "version": version_output,
                "status": "ok"
            }
            return info
        else:
            raise BinaryNotFoundError(
                f"diffai binary found at {binary_path} but failed to execute: {result.stderr}"
            )
            
    except subprocess.TimeoutExpired:
        raise BinaryNotFoundError("diffai binary execution timed out")
    except FileNotFoundError:
        raise BinaryNotFoundError(
            "diffai binary not found. Please install diffai or ensure it's in your PATH. "
            "See: https://github.com/kako-jun/diffai/releases"
        )
    except Exception as e:
        raise BinaryNotFoundError(f"Failed to verify diffai installation: {e}")


def run_diffai(args: List[str], timeout: Optional[int] = None) -> DiffResult:
    """
    Execute diffai with specified arguments.
    
    Args:
        args: Command-line arguments to pass to diffai
        timeout: Maximum execution time in seconds
        
    Returns:
        DiffResult object containing execution results
        
    Raises:
        DiffaiError: If execution fails
        BinaryNotFoundError: If diffai binary cannot be found
    """
    try:
        binary_path = _get_diffai_binary_path()
        cmd = [binary_path] + args
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        # Determine output format
        format_type = "json" if "--output" in args and "json" in args else "diffai"
        
        if result.returncode == 0:
            return DiffResult(result.stdout, result.returncode, format_type)
        else:
            # Handle common error cases
            if "No such file" in result.stderr or "not found" in result.stderr:
                raise InvalidInputError(
                    f"Input file not found: {result.stderr}",
                    result.returncode,
                    result.stderr
                )
            else:
                raise DiffaiError(
                    f"diffai execution failed: {result.stderr}",
                    result.returncode,
                    result.stderr
                )
                
    except subprocess.TimeoutExpired:
        raise DiffaiError(f"diffai execution timed out after {timeout} seconds")
    except FileNotFoundError:
        raise BinaryNotFoundError(
            "diffai binary not found. Please install diffai or ensure it's in your PATH."
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
        
    Example:
        >>> result = diff("model1.safetensors", "model2.safetensors", stats=True)
        >>> print(result)
        
        >>> options = DiffOptions(stats=True, architecture_comparison=True)
        >>> result = diff("model1.safetensors", "model2.safetensors", options)
    """
    # Handle different option formats
    if options is None:
        options = DiffOptions(**kwargs)
    elif isinstance(options, dict):
        combined_options: Dict[str, Any] = {**options, **kwargs}
        options = DiffOptions(**combined_options)
    elif kwargs:
        # Merge kwargs into existing DiffOptions
        option_dict = {
            field.name: getattr(options, field.name) 
            for field in options.__dataclass_fields__.values()
        }
        combined_options = {**option_dict, **kwargs}
        merged_opts: Dict[str, Union[str, bool, int, float, None]] = combined_options
        options = DiffOptions(**merged_opts)
    
    # Build command arguments
    args = [input1, input2]
    args.extend(options.to_cli_args())
    
    return run_diffai(args)


def diff_string(
    content1: str, 
    content2: str, 
    format_type: Union[str, None] = None,
    **kwargs
) -> DiffResult:
    """
    Compare two strings using diffai (creates temporary files).
    
    Args:
        content1: First string content
        content2: Second string content
        **kwargs: Options passed to diff()
        
    Returns:
        DiffResult object containing comparison results
        
    Note:
        This function creates temporary files for string comparison.
        Use diff() directly for file-based comparisons.
    """
    import tempfile
    import os
    
    # Create temporary files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f1, \
         tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f2:
        
        f1.write(content1)
        f2.write(content2)
        f1_path = f1.name
        f2_path = f2.name
    
    try:
        return diff(f1_path, f2_path, **kwargs)
    finally:
        # Clean up temporary files
        try:
            os.unlink(f1_path)
            os.unlink(f2_path)
        except OSError:
            pass  # Ignore cleanup errors