mod pytorch;
mod safetensors;
mod numpy;
mod matlab;

pub use pytorch::parse_pytorch_model;
pub use safetensors::parse_safetensors_model;
pub use numpy::parse_numpy_file;
pub use matlab::parse_matlab_file;

use anyhow::{anyhow, Result};
use serde_json::Value;
use std::path::Path;

use crate::types::FileFormat;

pub fn detect_format_from_path(path: &Path) -> Result<FileFormat> {
    match path.extension().and_then(|ext| ext.to_str()) {
        Some("pt") | Some("pth") => Ok(FileFormat::PyTorch),
        Some("safetensors") => Ok(FileFormat::Safetensors),
        Some("npy") | Some("npz") => Ok(FileFormat::NumPy),
        Some("mat") => Ok(FileFormat::Matlab),
        _ => {
            let ext = path
                .extension()
                .and_then(|ext| ext.to_str())
                .unwrap_or("unknown");
            Err(anyhow!(
                "Unsupported file format: '{}'. diffai only supports AI/ML file formats: .pt, .pth, .safetensors, .npy, .npz, .mat. For general structured data formats, please use diffx.",
                ext
            ))
        }
    }
}

pub fn parse_file_by_format(path: &Path, format: FileFormat) -> Result<Value> {
    match format {
        FileFormat::PyTorch => parse_pytorch_model(path),
        FileFormat::Safetensors => parse_safetensors_model(path),
        FileFormat::NumPy => parse_numpy_file(path),
        FileFormat::Matlab => parse_matlab_file(path),
    }
}
