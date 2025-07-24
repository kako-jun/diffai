use anyhow::Result;
use clap::{Parser, ValueEnum};
use diffai_core::{
    diff_paths, DiffOptions, DiffaiSpecificOptions, OutputFormat, format_output
};
use regex::Regex;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "diffai")]
#[command(about = "A unified AI/ML diff tool for structured data and ML models")]
#[command(version)]
struct Args {
    /// The first input file
    #[arg(value_name = "FILE1")]
    input1: PathBuf,

    /// The second input file  
    #[arg(value_name = "FILE2")]
    input2: PathBuf,

    /// Input file format (auto-detected if not specified)
    #[arg(short, long, value_enum)]
    format: Option<Format>,

    /// Output format
    #[arg(short, long)]
    output: Option<String>,

    /// Filter by path (only show differences in paths containing this string)
    #[arg(long)]
    path: Option<String>,

    /// Ignore keys matching this regex pattern
    #[arg(long)]
    ignore_keys_regex: Option<String>,

    /// Numerical comparison tolerance (for floating point numbers)
    #[arg(long)]
    epsilon: Option<f64>,

    /// Array comparison by ID key (compare arrays by this field instead of index)
    #[arg(long)]
    array_id_key: Option<String>,

    /// Suppress normal output; return only exit status
    #[arg(short, long)]  
    quiet: bool,

    /// Report only whether files differ, not the differences
    #[arg(long)]
    brief: bool,

    /// Show verbose processing information
    #[arg(short, long)]
    verbose: bool,

    /// Disable colored output
    #[arg(long)]
    no_color: bool,

    /// Enable memory optimization for large files
    #[arg(long)]
    memory_optimization: bool,

    /// Batch size for memory optimization
    #[arg(long)]
    batch_size: Option<usize>,

    /// Show unchanged values as well
    #[arg(long)]
    show_unchanged: bool,

    /// Show type information in output
    #[arg(long)]
    show_types: bool,

    // AI/ML specific options
    /// Enable ML analysis features
    #[arg(long)]
    ml_analysis: bool,

    /// Tensor comparison mode: shape, data, or both
    #[arg(long)]
    tensor_mode: Option<String>,

    /// Model format: pytorch, safetensors, numpy, matlab
    #[arg(long)]
    model_format: Option<String>,

    /// Enable scientific precision mode
    #[arg(long)]
    scientific_precision: bool,

    /// Weight change significance threshold
    #[arg(long)]
    weight_threshold: Option<f64>,

    /// Enable activation function analysis
    #[arg(long)]
    activation_analysis: bool,

    /// Track learning rate changes
    #[arg(long)]
    learning_rate_tracking: bool,

    /// Compare optimizer settings
    #[arg(long)]
    optimizer_comparison: bool,

    /// Track loss function changes
    #[arg(long)]
    loss_tracking: bool,

    /// Track accuracy metrics
    #[arg(long)]
    accuracy_tracking: bool,

    /// Check model version changes
    #[arg(long)]
    model_version_check: bool,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum Format {
    Json,
    Yaml,
    Csv,
    Toml,
    Ini,
    Xml,
    Pytorch,
    Safetensors,
    Numpy,
    Matlab,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Build options from CLI arguments
    let options = build_diff_options(&args)?;

    // Perform diff using paths (automatic file/directory detection)
    let results = diff_paths(
        &args.input1.to_string_lossy(),
        &args.input2.to_string_lossy(),
        Some(&options)
    )?;

    // Handle quiet mode
    if args.quiet {
        std::process::exit(if results.is_empty() { 0 } else { 1 });
    }

    // Handle brief mode
    if args.brief {
        if results.is_empty() {
            if args.verbose {
                println!("Files {} and {} are identical", 
                    args.input1.display(), args.input2.display());
            }
        } else {
            println!("Files {} and {} differ", 
                args.input1.display(), args.input2.display());
        }
        std::process::exit(if results.is_empty() { 0 } else { 1 });
    }

    // Format and output results
    let output_format = if let Some(format_str) = &args.output {
        OutputFormat::from_str(format_str)?
    } else {
        OutputFormat::Diffai
    };
    let formatted_output = format_output(&results, output_format)?;

    if !formatted_output.trim().is_empty() {
        println!("{}", formatted_output);
    } else if args.verbose {
        println!("No differences found");
    }

    // Exit with appropriate code (0 = no differences, 1 = differences found)
    std::process::exit(if results.is_empty() { 0 } else { 1 });
}

// File format detection and parsing functions are now handled by diffai-core

fn build_diff_options(args: &Args) -> Result<DiffOptions> {
    let ignore_keys_regex = if let Some(pattern) = &args.ignore_keys_regex {
        Some(Regex::new(pattern)?)
    } else {
        None
    };

    let diffai_options = Some(DiffaiSpecificOptions {
        ml_analysis_enabled: Some(args.ml_analysis),
        tensor_comparison_mode: args.tensor_mode.clone(),
        model_format: args.model_format.clone(),
        scientific_precision: Some(args.scientific_precision),
        weight_threshold: args.weight_threshold,
        activation_analysis: Some(args.activation_analysis),
        learning_rate_tracking: Some(args.learning_rate_tracking),
        optimizer_comparison: Some(args.optimizer_comparison),
        loss_tracking: Some(args.loss_tracking),
        accuracy_tracking: Some(args.accuracy_tracking),
        model_version_check: Some(args.model_version_check),
    });

    let output_format = if let Some(format_str) = &args.output {
        Some(OutputFormat::from_str(format_str)?)
    } else {
        None
    };

    Ok(DiffOptions {
        epsilon: args.epsilon,
        array_id_key: args.array_id_key.clone(),
        ignore_keys_regex,
        path_filter: args.path.clone(),
        output_format,
        show_unchanged: Some(args.show_unchanged),
        show_types: Some(args.show_types),
        use_memory_optimization: Some(args.memory_optimization),
        batch_size: args.batch_size,
        diffai_options,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_basic_diff() {
        let old = json!({"a": 1, "b": 2});
        let new = json!({"a": 1, "b": 3});
        
        let results = diff(&old, &new, None).unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_with_epsilon() {
        let old = json!({"value": 1.0});
        let new = json!({"value": 1.001});
        
        let options = DiffOptions {
            epsilon: Some(0.01),
            ..Default::default()
        };
        
        let results = diff(&old, &new, Some(&options)).unwrap();
        assert_eq!(results.len(), 0); // Should be within epsilon tolerance
    }

    #[test]
    fn test_ml_specific_options() {
        let old = json!({"learning_rate": 0.01, "accuracy": 0.85});
        let new = json!({"learning_rate": 0.02, "accuracy": 0.87});
        
        let diffai_options = DiffaiSpecificOptions {
            learning_rate_tracking: Some(true),
            accuracy_tracking: Some(true),
            ..Default::default()
        };
        
        let options = DiffOptions {
            diffai_options: Some(diffai_options),
            ..Default::default()
        };
        
        let results = diff(&old, &new, Some(&options)).unwrap();
        assert_eq!(results.len(), 2);
    }
}