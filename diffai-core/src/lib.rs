// diffai-core: AI/ML specific diff library
//
// This library provides diff functionality specifically designed for AI/ML file formats
// such as PyTorch, SafeTensors, NumPy, and MATLAB.

// Module declarations
mod types;
mod diff;
mod ml_analysis;
mod parsers;
mod output;

// Re-export diffx-core utilities
pub use diffx_core::{
    value_type_name,
    estimate_memory_usage,
    would_exceed_memory_limit,
};

// Re-export types
pub use types::{
    DiffResult,
    TensorStats,
    OutputFormat,
    DiffOptions,
    FileFormat,
    DiffFormat,
};

// Re-export main diff functions
pub use diff::{
    diff,
    diff_paths,
    extract_tensor_data,
    extract_tensor_shape,
};

// Re-export parsers
pub use parsers::{
    parse_pytorch_model,
    parse_safetensors_model,
    parse_numpy_file,
    parse_matlab_file,
    detect_format_from_path,
    parse_file_by_format,
};

// Re-export output formatting functions
pub use output::{
    format_output,
    format_diff_results,
};
