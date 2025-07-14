use std::env;
use std::process::{Command, ExitCode};

fn main() -> ExitCode {
    // Get command line arguments (skip the first one which is the program name)
    let args: Vec<String> = env::args().skip(1).collect();
    
    // Execute diffai CLI with the provided arguments
    match Command::new("diffai").args(&args).status() {
        Ok(status) => {
            if status.success() {
                ExitCode::SUCCESS
            } else {
                ExitCode::from(status.code().unwrap_or(1) as u8)
            }
        }
        Err(_) => {
            eprintln!("Error: Failed to execute diffai CLI. Make sure diffai is installed.");
            ExitCode::FAILURE
        }
    }
}