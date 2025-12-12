use anyhow::{Context, Result};
use diffai_core::{diff, DiffOptions, OutputFormat};
use regex::Regex;
use serde_json::Value;

use crate::cli::Args;
use crate::formatters::handle_output_and_exit;
use crate::input::{infer_format_from_path, parse_content, read_input};

pub fn build_diff_options(args: &Args) -> Result<DiffOptions> {
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

pub fn handle_stdin_input(args: &Args, input1_is_stdin: bool, input2_is_stdin: bool) -> Result<()> {
    if input1_is_stdin && input2_is_stdin {
        // Case 2 & 3: Both inputs from stdin - read two data sets from stdin
        return handle_both_stdin(args);
    }

    let input1 = args.input1.as_ref().expect("input1 is required");
    let input2 = args.input2.as_ref().expect("input2 is required");

    // Case 1: One stdin, one file
    let content1 = read_input(input1)?;
    let content2 = read_input(input2)?;

    // Determine input format
    let input_format = if let Some(fmt) = args.format {
        fmt
    } else {
        infer_format_from_path(input1)
            .or_else(|| infer_format_from_path(input2))
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

#[cfg(test)]
mod tests {
    use super::*;
    use diffai_core::{diff, DiffResult};
    use serde_json::json;

    #[test]
    fn test_basic_diff() {
        let old = json!({"a": 1, "b": 2});
        let new = json!({"a": 1, "b": 3});

        let results = diff(&old, &new, None).unwrap();
        // Filter for base diff results only (ML analysis may add more)
        let base_diffs: Vec<_> = results
            .iter()
            .filter(|r| matches!(r, DiffResult::Modified(path, _, _) if path == "b"))
            .collect();
        assert!(
            !base_diffs.is_empty(),
            "Should have Modified result for 'b'"
        );
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
        let base_diffs: Vec<_> = results
            .iter()
            .filter(|r| matches!(r, DiffResult::Modified(path, _, _) if path == "value"))
            .collect();
        assert!(base_diffs.is_empty(), "Should be within epsilon tolerance");
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
        assert!(results.len() >= 2, "Should detect at least 2 changes");
    }
}
