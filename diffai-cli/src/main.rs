use anyhow::{Result, Context};
use clap::{Parser, ValueEnum};
use diffai_core::{
    diff_paths, DiffOptions, DiffaiSpecificOptions, OutputFormat, format_output, DiffResult, diff, parse_csv, parse_ini, parse_xml
};
use regex::Regex;
use serde_json::Value;
use std::path::PathBuf;
use std::io::{self, Read};
use std::fs;

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

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
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

fn infer_format_from_path(path: &PathBuf) -> Option<Format> {
    if path.to_str() == Some("-") {
        // Cannot infer format from stdin, user must specify --format
        None
    } else {
        path.extension()
            .and_then(|ext| ext.to_str())
            .and_then(|ext_str| match ext_str.to_lowercase().as_str() {
                "json" => Some(Format::Json),
                "yaml" | "yml" => Some(Format::Yaml),
                "toml" => Some(Format::Toml),
                "ini" => Some(Format::Ini),
                "xml" => Some(Format::Xml),
                "csv" => Some(Format::Csv),
                "pytorch" => Some(Format::Pytorch),
                "safetensors" => Some(Format::Safetensors),
                "numpy" => Some(Format::Numpy),
                "matlab" => Some(Format::Matlab),
                _ => None,
            })
    }
}

fn parse_content(content: &str, format: Format) -> Result<Value> {
    match format {
        Format::Json => serde_json::from_str(content).context("Failed to parse JSON"),
        Format::Yaml => serde_yml::from_str(content).context("Failed to parse YAML"),
        Format::Toml => toml::from_str(content).context("Failed to parse TOML"),
        Format::Ini => parse_ini(content).context("Failed to parse INI"),
        Format::Xml => parse_xml(content).context("Failed to parse XML"),
        Format::Csv => parse_csv(content).context("Failed to parse CSV"),
        // Note: AI/ML formats would need special handling, but for stdin they would typically be JSON
        _ => Err(anyhow::anyhow!("Format {:?} not supported for stdin input", format)),
    }
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

fn handle_both_stdin(args: &Args) -> Result<()> {
    // Read entire stdin
    let mut buffer = String::new();
    io::stdin().read_to_string(&mut buffer).context("Failed to read from stdin")?;
    
    // Try to parse as two separate JSON/YAML objects
    if let Some(fmt) = args.format {
        match fmt {
            Format::Json => handle_both_stdin_json(&buffer, args),
            Format::Yaml => handle_both_stdin_yaml(&buffer, args),
            _ => Err(anyhow::anyhow!("Two stdin inputs only supported for JSON and YAML formats")),
        }
    } else {
        // Try JSON first, then YAML
        handle_both_stdin_json(&buffer, args)
            .or_else(|_| handle_both_stdin_yaml(&buffer, args))
    }
}

fn handle_both_stdin_json(buffer: &str, args: &Args) -> Result<()> {
    // Try to parse as JSON Lines (two separate JSON objects)
    let lines: Vec<&str> = buffer.trim().lines().collect();
    
    if lines.len() >= 2 {
        // Try to parse first and last non-empty lines as JSON
        let first_json = lines.iter().find(|line| !line.trim().is_empty())
            .ok_or_else(|| anyhow::anyhow!("No JSON content found in stdin"))?;
        let second_json = lines.iter().rev().find(|line| !line.trim().is_empty())
            .ok_or_else(|| anyhow::anyhow!("Only one JSON object found in stdin"))?;
        
        if first_json != second_json {
            let v1: Value = serde_json::from_str(first_json)?;
            let v2: Value = serde_json::from_str(second_json)?;
            
            let options = build_diff_options_for_values(args)?;
            let differences = diff(&v1, &v2, Some(&options))?;
            
            return handle_output_and_exit(&differences, args);
        }
    }
    
    // Try to parse as two concatenated JSON objects
    let trimmed = buffer.trim();
    if let Some(end_of_first) = find_json_object_end(trimmed) {
        let first_part = &trimmed[..end_of_first];
        let second_part = trimmed[end_of_first..].trim();
        
        if !second_part.is_empty() {
            let v1: Value = serde_json::from_str(first_part)?;
            let v2: Value = serde_json::from_str(second_part)?;
            
            let options = build_diff_options_for_values(args)?;
            let differences = diff(&v1, &v2, Some(&options))?;
            
            return handle_output_and_exit(&differences, args);
        }
    }
    
    Err(anyhow::anyhow!("Could not parse two JSON objects from stdin"))
}

fn handle_both_stdin_yaml(buffer: &str, args: &Args) -> Result<()> {
    // Try to parse as two YAML documents separated by ---
    let documents: Vec<&str> = buffer.split("---").collect();
    
    if documents.len() >= 2 {
        let doc1 = documents[0].trim();
        let doc2 = documents[1].trim();
        
        if !doc1.is_empty() && !doc2.is_empty() {
            let v1: Value = serde_yml::from_str(doc1)?;
            let v2: Value = serde_yml::from_str(doc2)?;
            
            let options = build_diff_options_for_values(args)?;
            let differences = diff(&v1, &v2, Some(&options))?;
            
            return handle_output_and_exit(&differences, args);
        }
    }
    
    Err(anyhow::anyhow!("Could not parse two YAML documents from stdin (expected '---' separator)"))
}

fn find_json_object_end(json_str: &str) -> Option<usize> {
    let mut brace_count = 0;
    let mut in_string = false;
    let mut escape_next = false;
    
    for (i, ch) in json_str.char_indices() {
        if escape_next {
            escape_next = false;
            continue;
        }
        
        match ch {
            '"' if !escape_next => in_string = !in_string,
            '\\' if in_string => escape_next = true,
            '{' if !in_string => brace_count += 1,
            '}' if !in_string => {
                brace_count -= 1;
                if brace_count == 0 {
                    return Some(i + 1);
                }
            }
            _ => {}
        }
    }
    
    None
}

fn build_diff_options_for_values(args: &Args) -> Result<DiffOptions> {
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
        OutputFormat::from_str(format_str)?
    } else {
        OutputFormat::Diffai
    };
    let formatted_output = format_output(differences, output_format)?;

    if !formatted_output.trim().is_empty() {
        println!("{}", formatted_output);
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