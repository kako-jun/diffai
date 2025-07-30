// Core test modules for diffai-core library
// These tests validate the unified API and ML analysis functionality

mod fixtures;

// Core functionality tests
mod core_unified_api_tests;
mod core_ml_analysis_tests;
mod core_format_tests_simple;

// Re-export fixtures for use in other test modules
pub use fixtures::*;