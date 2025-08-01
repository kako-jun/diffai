use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use diffai_core::{
    diff, diff_paths, format_output, DiffOptions, DiffResult, OutputFormat,
};
use regex::Regex;
use serde_json::Value;
use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(name = "diffai")]
#[command(about = "An AI/ML diff tool for structured data and ML models")]
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
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
enum Format {
    Pytorch,
    Safetensors,
    Numpy,
    Matlab,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Check for stdin usage
    let input1_is_stdin = args.input1.to_str() == Some("-");
    let input2_is_stdin = args.input2.to_str() == Some("-");

    if input1_is_stdin || input2_is_stdin {
        // Handle stdin cases
        return handle_stdin_input(&args, input1_is_stdin, input2_is_stdin);
    }

    // Build options from CLI arguments
    let options = build_diff_options(&args)?;

    // Perform diff using paths (automatic file/directory detection)
    let results = diff_paths(
        &args.input1.to_string_lossy(),
        &args.input2.to_string_lossy(),
        Some(&options),
    )?;

    // Handle quiet mode
    if args.quiet {
        std::process::exit(if results.is_empty() { 0 } else { 1 });
    }

    // Handle brief mode
    if args.brief {
        if results.is_empty() {
            if args.verbose {
                println!(
                    "Files {} and {} are identical",
                    args.input1.display(),
                    args.input2.display()
                );
            }
        } else {
            println!(
                "Files {} and {} differ",
                args.input1.display(),
                args.input2.display()
            );
        }
        std::process::exit(if results.is_empty() { 0 } else { 1 });
    }

    // Format and output results
    let output_format = if let Some(format_str) = &args.output {
        OutputFormat::parse_format(format_str)?
    } else {
        OutputFormat::Diffai
    };
    let formatted_output = format_output(&results, output_format)?;

    if !formatted_output.trim().is_empty() {
        println!("{formatted_output}");
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

    
    let output_format = if let Some(format_str) = &args.output {
        Some(OutputFormat::parse_format(format_str)?)
    } else {
        None
    };

    Ok(DiffOptions {
        epsilon: args.epsilon,
        array_id_key: args.array_id_key.clone(),
        ignore_keys_regex,
        path_filter: args.path.clone(),
        output_format,
        // lawkitパターン：オプションは削除、最適化は常に有効
    })
}

fn read_input(file_path: &PathBuf) -> Result<String> {
    if file_path.to_str() == Some("-") {
        let mut buffer = String::new();
        io::stdin()
            .read_to_string(&mut buffer)
            .context("Failed to read from stdin")?;
        Ok(buffer)
    } else {
        fs::read_to_string(file_path)
            .with_context(|| format!("Failed to read file: {}", file_path.display()))
    }
}

fn infer_format_from_path(path: &Path) -> Option<Format> {
    if path.to_str() == Some("-") {
        // Cannot infer format from stdin, user must specify --format
        None
    } else {
        path.extension()
            .and_then(|ext| ext.to_str())
            .and_then(|ext_str| match ext_str.to_lowercase().as_str() {
                "pt" | "pth" => Some(Format::Pytorch),
                "safetensors" => Some(Format::Safetensors),
                "npy" | "npz" => Some(Format::Numpy),
                "mat" => Some(Format::Matlab),
                _ => None,
            })
    }
}

fn parse_content(_content: &str, format: Format) -> Result<Value> {
    // AI/ML files are binary formats and cannot be read from stdin
    Err(anyhow::anyhow!(
        "Format {:?} not supported for stdin input. AI/ML files are binary formats and must be read from files. diffai only supports: .pt, .pth, .safetensors, .npy, .npz, .mat",
        format
    ))
}

fn handle_stdin_input(args: &Args, input1_is_stdin: bool, input2_is_stdin: bool) -> Result<()> {
    if input1_is_stdin && input2_is_stdin {
        // Case 2 & 3: Both inputs from stdin - read two data sets from stdin
        return handle_both_stdin(args);
    }

    // Case 1: One stdin, one file
    let content1 = read_input(&args.input1)?;
    let content2 = read_input(&args.input2)?;

    // Determine input format
    let input_format = if let Some(fmt) = args.format {
        fmt
    } else {
        infer_format_from_path(&args.input1)
            .or_else(|| infer_format_from_path(&args.input2))
            .context("Could not infer format from file extensions. Please specify --format.")?
    };

    // Parse content
    let v1: Value = parse_content(&content1, input_format)?;
    let v2: Value = parse_content(&content2, input_format)?;

    // Build options and perform diff
    let options = build_diff_options_for_values(args)?;
    let differences = diff(&v1, &v2, Some(&options))?;

    // Handle output
    handle_output_and_exit(&differences, args)
}

fn handle_both_stdin(_args: &Args) -> Result<()> {
    Err(anyhow::anyhow!(
        "diffai does not support reading from stdin. AI/ML files are binary formats and must be read from files. Supported formats: .pt, .pth, .safetensors, .npy, .npz, .mat"
    ))
}

fn build_diff_options_for_values(args: &Args) -> Result<DiffOptions> {
    let ignore_keys_regex = if let Some(pattern) = &args.ignore_keys_regex {
        Some(Regex::new(pattern)?)
    } else {
        None
    };

    let output_format = if let Some(format_str) = &args.output {
        Some(OutputFormat::parse_format(format_str)?)
    } else {
        None
    };

    Ok(DiffOptions {
        epsilon: args.epsilon,
        array_id_key: args.array_id_key.clone(),
        ignore_keys_regex,
        path_filter: args.path.clone(),
        output_format,
        // lawkitパターン：オプションは削除、最適化は常に有効
    })
}

fn handle_output_and_exit(differences: &[DiffResult], args: &Args) -> Result<()> {
    // Handle quiet mode
    if args.quiet {
        std::process::exit(if differences.is_empty() { 0 } else { 1 });
    }

    // Handle brief mode
    if args.brief {
        if differences.is_empty() {
            if args.verbose {
                println!("Inputs are identical");
            }
        } else {
            println!("Inputs differ");
        }
        std::process::exit(if differences.is_empty() { 0 } else { 1 });
    }

    // Format and output results
    let output_format = if let Some(format_str) = &args.output {
        OutputFormat::parse_format(format_str)?
    } else {
        OutputFormat::Diffai
    };
    let formatted_output = format_output(differences, output_format)?;

    if !formatted_output.trim().is_empty() {
        println!("{formatted_output}");
    } else if args.verbose {
        println!("No differences found");
    }

    // Exit with appropriate code (0 = no differences, 1 = differences found)
    std::process::exit(if differences.is_empty() { 0 } else { 1 });
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
    fn test_ml_automatic_analysis() {
        let old = json!({"learning_rate": 0.01, "accuracy": 0.85});
        let new = json!({"learning_rate": 0.02, "accuracy": 0.87});

        // lawkitパターン：ML分析は自動実行、個別オプションは不要
        let options = DiffOptions {
            ..Default::default()
        };

        let results = diff(&old, &new, Some(&options)).unwrap();
        // 自動ML分析でlearning_rateとaccuracyの変更を検出
        assert_eq!(results.len(), 2);
    }
}
