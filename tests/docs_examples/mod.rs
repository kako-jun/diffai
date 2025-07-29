// Documentation examples tests - following diffx pattern
// Each test module corresponds to a specific documentation file

// High priority - essential documentation tests
pub mod cli_help;                           // CLI help output validation
pub mod index_examples;                     // index.md examples
pub mod getting_started_examples;           // user-guide/getting-started.md
pub mod basic_usage_examples;              // user-guide/basic-usage.md  
pub mod cli_reference_examples;            // reference/cli-reference.md
pub mod api_examples;                      // reference/api-reference.md
pub mod ml_model_comparison_examples;      // user-guide/ml-model-comparison.md

// Medium priority - important functionality tests
pub mod ml_workflows_examples;             // user-guide/ml-workflows.md
pub mod scientific_data_examples;          // user-guide/scientific-data.md
pub mod formats_examples;                  // reference/formats.md
pub mod output_formats_examples;           // reference/output-formats.md

// Lower priority - supplementary tests
// pub mod ml_analysis_examples;           // reference/ml-analysis.md (to be implemented)
// pub mod unified_api_examples;           // bindings/unified-api.md (to be implemented)