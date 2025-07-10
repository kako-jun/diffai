"""
Backward compatibility layer for diffai Python package.

This module provides legacy function names and interfaces for backward
compatibility with older versions of the diffai Python package.
"""

from typing import Any, Dict, List, Optional, Union
from .diffai import diff, run_diffai, verify_installation, DiffResult, DiffOptions


def diffai_diff(
    input1: str,
    input2: str,
    stats: bool = False,
    output_format: Optional[str] = None,
    **kwargs
) -> str:
    """
    Legacy function for running diffai comparisons.
    
    Args:
        input1: Path to first input file
        input2: Path to second input file
        stats: Whether to include statistics
        output_format: Output format ("json", "yaml", or None for default)
        **kwargs: Additional options
        
    Returns:
        Raw output string from diffai
        
    Note:
        This function is deprecated. Use diffai.diff() instead for better
        type safety and structured results.
    """
    from .diffai import OutputFormat
    
    options = DiffOptions(
        stats=stats,
        output_format=OutputFormat(output_format) if output_format else None,
        **kwargs
    )
    
    result = diff(input1, input2, options)
    return result.raw_output


def diffai_diff_files(
    file1: str,
    file2: str,
    format_type: str = "diffai",
    **options
) -> Union[str, List[Dict[str, Any]]]:
    """
    Legacy function for file-based diffai comparisons.
    
    Args:
        file1: Path to first file
        file2: Path to second file
        format_type: Output format ("json", "yaml", or "diffai")
        **options: Additional diffai options
        
    Returns:
        Parsed results (list/dict for JSON, string for others)
        
    Note:
        This function is deprecated. Use diffai.diff() instead.
    """
    from .diffai import OutputFormat
    
    diff_options = DiffOptions(
        output_format=OutputFormat(format_type) if format_type != "diffai" else None,
        **options
    )
    
    result = diff(file1, file2, diff_options)
    
    if format_type == "json" and result.is_json:
        return result.data
    else:
        return result.raw_output


def check_diffai_binary() -> bool:
    """
    Legacy function to check if diffai binary is available.
    
    Returns:
        True if diffai binary is accessible, False otherwise
        
    Note:
        This function is deprecated. Use diffai.verify_installation() instead
        for more detailed installation information.
    """
    try:
        verify_installation()
        return True
    except Exception:
        return False


# Legacy aliases for common operations
def get_stats(input1: str, input2: str) -> str:
    """Get statistics for two files (legacy function)."""
    return diffai_diff(input1, input2, stats=True)


def get_json_diff(input1: str, input2: str, **kwargs) -> Union[List, Dict]:
    """Get JSON-formatted diff results (legacy function)."""
    result = diffai_diff_files(input1, input2, format_type="json", **kwargs)
    return result if isinstance(result, (list, dict)) else []


def compare_models(model1: str, model2: str, **analysis_options) -> str:
    """Compare ML models with specified analysis options (legacy function)."""
    return diffai_diff(model1, model2, stats=True, **analysis_options)


# Version compatibility
def get_version() -> str:
    """Get diffai version information."""
    try:
        info = verify_installation()
        return info.get("version", "unknown")
    except Exception:
        return "unknown"


# Legacy configuration object
class LegacyDiffConfig:
    """
    Legacy configuration class for backward compatibility.
    
    Note:
        This class is deprecated. Use DiffOptions instead.
    """
    
    def __init__(self, **kwargs):
        self.options = kwargs
    
    def to_options(self) -> DiffOptions:
        """Convert to modern DiffOptions object."""
        return DiffOptions(**self.options)
    
    def add_option(self, key: str, value: Any) -> None:
        """Add an option to the configuration."""
        self.options[key] = value
    
    def get_option(self, key: str, default: Any = None) -> Any:
        """Get an option value."""
        return self.options.get(key, default)


# Export legacy symbols for compatibility
__all__ = [
    "diffai_diff",
    "diffai_diff_files", 
    "check_diffai_binary",
    "get_stats",
    "get_json_diff",
    "compare_models",
    "get_version",
    "LegacyDiffConfig",
]