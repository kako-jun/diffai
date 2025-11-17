use anyhow::Result;
use diffai_core::{format_output, DiffResult, OutputFormat};
use std::path::PathBuf;

use crate::cli::Args;

pub fn handle_output_and_exit(differences: &[DiffResult], args: &Args) -> Result<()> {
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

pub fn handle_file_output_and_exit(
    differences: &[DiffResult],
    args: &Args,
    file1: &PathBuf,
    file2: &PathBuf,
) -> Result<()> {
    // Handle quiet mode
    if args.quiet {
        std::process::exit(if differences.is_empty() { 0 } else { 1 });
    }

    // Handle brief mode
    if args.brief {
        if differences.is_empty() {
            if args.verbose {
                println!("Files {} and {} are identical", file1.display(), file2.display());
            }
        } else {
            println!("Files {} and {} differ", file1.display(), file2.display());
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
