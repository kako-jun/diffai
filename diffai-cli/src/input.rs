use anyhow::{Context, Result};
use serde_json::Value;
use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};

use crate::cli::Format;

pub fn read_input(file_path: &PathBuf) -> Result<String> {
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

pub fn infer_format_from_path(path: &Path) -> Option<Format> {
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

pub fn parse_content(_content: &str, format: Format) -> Result<Value> {
    // AI/ML files are binary formats and cannot be read from stdin
    Err(anyhow::anyhow!(
        "Format {:?} not supported for stdin input. AI/ML files are binary formats and must be read from files. diffai only supports: .pt, .pth, .safetensors, .npy, .npz, .mat",
        format
    ))
}
