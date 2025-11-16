mod cli;
mod commands;
mod formatters;
mod input;

use anyhow::Result;
use clap::Parser;
use diffai_core::diff_paths;

use cli::Args;
use commands::{build_diff_options, handle_stdin_input};
use formatters::handle_file_output_and_exit;

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

    // Handle output and exit
    handle_file_output_and_exit(&results, &args, &args.input1, &args.input2)
}
