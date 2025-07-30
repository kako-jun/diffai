"""
diffai_python - AI/ML specialized diff tool Python bindings

This module provides Python bindings for the diffai Rust library,
specializing in comparing AI/ML model files like PyTorch, Safetensors,
NumPy arrays, and MATLAB files with advanced tensor analysis.
"""

from .diffai_python import diff_py, __version__

# Export diff_py as the main function (following diffx pattern)
# Users can call diffai_python.diff_py() directly
def diff(old, new, **kwargs):
    """
    AI/ML specialized diff function with tensor analysis capabilities.
    
    Compare two data structures with automatic ML analysis for supported formats.
    
    Args:
        old: The original data structure
        new: The modified data structure  
        **kwargs: Optional diff options including:
            - ml_analysis_enabled: Enable ML-specific analysis
            - tensor_comparison_mode: "shape", "data", or "both"
            - learning_rate_tracking: Track learning rate changes
            - weight_threshold: Threshold for significant weight changes
            - epsilon: Numerical comparison tolerance
            - And many more ML-specific options...
    
    Returns:
        List of difference dictionaries with AI/ML specific analysis
    
    Example:
        >>> import diffai_python
        >>> old = {"learning_rate": 0.001, "accuracy": 0.85}
        >>> new = {"learning_rate": 0.01, "accuracy": 0.92}
        >>> results = diffai_python.diff(old, new, learning_rate_tracking=True)
        >>> print(results)
    """
    return diff_py(old, new, **kwargs)

# Keep both interfaces available for compatibility
__all__ = ['diff', 'diff_py', '__version__']