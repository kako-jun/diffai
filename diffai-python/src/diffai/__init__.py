"""
diffai - AI/ML specialized diff tool for deep tensor comparison and analysis

This package provides Python bindings for diffai, a powerful command-line tool
specialized for comparing AI/ML model files, scientific data, and structured data.

Quick start:
    >>> import diffai
    >>> result = diffai.diff("model1.safetensors", "model2.safetensors", stats=True)
    >>> print(result)

Advanced usage:
    >>> options = diffai.DiffOptions(
    ...     stats=True,
    ...     architecture_comparison=True,
    ...     output_format="json"
    ... )
    >>> result = diffai.diff("model1.safetensors", "model2.safetensors", options)
"""

from .diffai import (
    # Main API functions
    diff,
    diff_string,
    run_diffai,
    verify_installation,
    
    # Configuration and result classes
    DiffOptions,
    DiffResult,
    OutputFormat,
    
    # Exceptions
    DiffaiError,
    BinaryNotFoundError,
    InvalidInputError,
    
    # Version info
    __version__,
)

# No backward compatibility imports - clean modern API only

__all__ = [
    # Main API
    "diff",
    "diff_string", 
    "run_diffai",
    "verify_installation",
    
    # Configuration
    "DiffOptions",
    "DiffResult",
    "OutputFormat",
    
    # Exceptions
    "DiffaiError",
    "BinaryNotFoundError", 
    "InvalidInputError",
    
    # Metadata
    "__version__",
]