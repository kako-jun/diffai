// diffai-core: AI/ML specific diff library
//
// This library provides diff functionality specifically designed for AI/ML file formats
// such as PyTorch, SafeTensors, NumPy, and MATLAB.

// Allow dead_code for ML analysis features that are placeholders for future functionality
#![allow(dead_code)]
// Allow these lints for complex ML analysis code patterns
#![allow(clippy::collapsible_if)]
#![allow(clippy::collapsible_match)]
#![allow(clippy::manual_clamp)]

// Module declarations
mod diff;
mod ml_analysis;
mod output;
mod parsers;
mod types;

// Re-export diffx-core utilities
pub use diffx_core::{estimate_memory_usage, value_type_name, would_exceed_memory_limit};

// Re-export types
pub use types::{DiffFormat, DiffOptions, DiffResult, FileFormat, OutputFormat, TensorStats};

// Re-export main diff functions
pub use diff::{diff, diff_paths, extract_tensor_data, extract_tensor_shape};

// Re-export parsers
pub use parsers::{
    detect_format_from_path, parse_file_by_format, parse_matlab_file, parse_numpy_file,
    parse_pytorch_model, parse_safetensors_model,
};

// Re-export output formatting functions
pub use output::{format_diff_results, format_output};
