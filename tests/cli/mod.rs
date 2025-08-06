// Restructured CLI tests following diffx best practices

// Basic functionality tests
pub mod basic_commands;          // Core commands (help, version, basic usage)
pub mod basic_verbose_mode;      // Verbose mode functionality

// Format support tests  
pub mod formats_support;         // All supported formats (PyTorch, Safetensors, NumPy, MATLAB)

// ML Analysis features tests
pub mod features_ml_analysis;    // All 11 ML analysis features
pub mod comprehensive_ml_analysis_tests;  // Comprehensive ML analysis testing

// Output format tests
pub mod output_formats;          // JSON, YAML, default format output

// Option tests
pub mod no_color_option;         // No color option specific tests
pub mod options;                 // General options testing

// Error handling tests
pub mod errors_error_handling;   // Error cases and graceful failure

// Legacy tests (to be updated)
pub mod help_output;             // Legacy help test
pub mod version_info;            // Legacy version test
