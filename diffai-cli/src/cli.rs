use clap::{Parser, ValueEnum};
use clap_complete::Shell;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "diffai")]
#[command(about = "An AI/ML diff tool for structured data and ML models")]
#[command(version)]
pub struct Args {
    /// The first input file
    #[arg(value_name = "FILE1", required_unless_present = "completions")]
    pub input1: Option<PathBuf>,

    /// The second input file
    #[arg(value_name = "FILE2", required_unless_present = "completions")]
    pub input2: Option<PathBuf>,

    /// Generate shell completions for the specified shell
    #[arg(long, value_enum, value_name = "SHELL")]
    pub completions: Option<Shell>,

    /// Input file format (auto-detected if not specified)
    #[arg(short, long, value_enum)]
    pub format: Option<Format>,

    /// Output format
    #[arg(short, long)]
    pub output: Option<String>,

    /// Filter by path (only show differences in paths containing this string)
    #[arg(long)]
    pub path: Option<String>,

    /// Ignore keys matching this regex pattern
    #[arg(long)]
    pub ignore_keys_regex: Option<String>,

    /// Numerical comparison tolerance (for floating point numbers)
    #[arg(long)]
    pub epsilon: Option<f64>,

    /// Array comparison by ID key (compare arrays by this field instead of index)
    #[arg(long)]
    pub array_id_key: Option<String>,

    /// Suppress normal output; return only exit status
    #[arg(short, long)]
    pub quiet: bool,

    /// Report only whether files differ, not the differences
    #[arg(long)]
    pub brief: bool,

    /// Show verbose processing information
    #[arg(short, long)]
    pub verbose: bool,

    /// Disable colored output
    #[arg(long)]
    pub no_color: bool,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
pub enum Format {
    Pytorch,
    Safetensors,
    Numpy,
    Matlab,
}
