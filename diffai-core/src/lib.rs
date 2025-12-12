// diffai-core: AI/ML specific diff library
//
// This library provides diff functionality specifically designed for AI/ML file formats
// such as PyTorch, SafeTensors, NumPy, and MATLAB.

#![allow(dead_code)]
#![allow(clippy::collapsible_if)]
#![allow(clippy::collapsible_match)]
#![allow(clippy::manual_clamp)]

mod diff;
mod ml_analysis;
mod output;
mod parsers;
mod types;

// Re-export diffx-core utilities
pub use diffx_core::value_type_name;

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
