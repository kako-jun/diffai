mod cli;
mod commands;
mod formatters;
mod input;

use anyhow::Result;
use clap::{CommandFactory, Parser};
use clap_complete::generate;
use diffai_core::diff_paths;
use std::io;

use cli::Args;
use commands::{build_diff_options, handle_stdin_input};
use formatters::handle_file_output_and_exit;

fn main() -> Result<()> {
    let args = Args::parse();

    // Handle shell completions
    if let Some(shell) = args.completions {
        let mut cmd = Args::command();
        let name = cmd.get_name().to_string();
        generate(shell, &mut cmd, name, &mut io::stdout());
        return Ok(());
    }

    // input1 and input2 are required unless completions is present
    let input1 = args.input1.as_ref().expect("input1 is required");
    let input2 = args.input2.as_ref().expect("input2 is required");

    // Check for stdin usage
    let input1_is_stdin = input1.to_str() == Some("-");
    let input2_is_stdin = input2.to_str() == Some("-");

    if input1_is_stdin || input2_is_stdin {
        // Handle stdin cases
        return handle_stdin_input(&args, input1_is_stdin, input2_is_stdin);
    }

    // Build options from CLI arguments
    let options = build_diff_options(&args)?;

    // Perform diff using paths (automatic file/directory detection)
    let results = diff_paths(
        &input1.to_string_lossy(),
        &input2.to_string_lossy(),
        Some(&options),
    )?;

    // Handle output and exit
    handle_file_output_and_exit(&results, &args, input1, input2)
}
