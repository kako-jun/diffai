#!/usr/bin/env python3
"""
Binary installer for diffai Python package.

This module handles downloading and installing platform-specific diffai binaries
from GitHub releases. It supports automatic platform detection and graceful
error handling for environments where binary installation fails.
"""

import argparse
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import importlib.metadata

try:
    PACKAGE_VERSION = importlib.metadata.version("diffai-python")
except importlib.metadata.PackageNotFoundError:
    # Fallback for development
    PACKAGE_VERSION = "0.2.8"

GITHUB_REPO = "kako-jun/diffai"
RELEASES_URL = f"https://github.com/{GITHUB_REPO}/releases/download/v{PACKAGE_VERSION}"


def get_platform_info() -> Dict[str, str]:
    """
    Detect current platform and return appropriate binary information.
    
    Returns:
        Dictionary containing platform-specific binary information
        
    Raises:
        ValueError: If platform is not supported
    """
    system = platform.system()
    machine = platform.machine()
    
    # Normalize architecture names
    if machine in ('x86_64', 'AMD64'):
        arch = 'x86_64'
    elif machine in ('arm64', 'aarch64'):
        arch = 'aarch64'
    else:
        raise ValueError(f"Unsupported architecture: {machine}")
    
    if system == "Windows":
        return {
            "platform": "windows",
            "arch": arch,
            "extension": "zip",
            "binary_name": "diffai.exe",
            "archive_name": f"diffai-windows-{arch}.zip"
        }
    elif system == "Darwin":
        return {
            "platform": "macos", 
            "arch": arch,
            "extension": "tar.gz",
            "binary_name": "diffai",
            "archive_name": f"diffai-macos-{arch}.tar.gz"
        }
    elif system == "Linux":
        return {
            "platform": "linux",
            "arch": arch, 
            "extension": "tar.gz",
            "binary_name": "diffai",
            "archive_name": f"diffai-linux-{arch}.tar.gz"
        }
    else:
        raise ValueError(f"Unsupported platform: {system}")


def download_file(url: str, destination: Path, chunk_size: int = 8192) -> None:
    """
    Download a file from URL to destination path.
    
    Args:
        url: URL to download from
        destination: Local path to save file
        chunk_size: Size of chunks to download at a time
        
    Raises:
        urllib.error.URLError: If download fails
    """
    print(f"Downloading from: {url}")
    
    try:
        with urllib.request.urlopen(url) as response:
            if response.status != 200:
                raise urllib.error.URLError(f"HTTP {response.status}: {response.reason}")
            
            total_size = response.headers.get('content-length')
            if total_size:
                total_size = int(total_size)
                print(f"File size: {total_size:,} bytes")
            
            downloaded = 0
            with open(destination, 'wb') as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size:
                        percent = (downloaded / total_size) * 100
                        print(f"\rProgress: {percent:.1f}% ({downloaded:,}/{total_size:,} bytes)", end='')
            
            if total_size:
                print()  # New line after progress
                
    except Exception as e:
        if destination.exists():
            destination.unlink()
        raise urllib.error.URLError(f"Download failed: {e}")


def extract_archive(archive_path: Path, extract_to: Path, platform_info: Dict[str, str]) -> Path:
    """
    Extract downloaded archive and return path to binary.
    
    Args:
        archive_path: Path to downloaded archive
        extract_to: Directory to extract to
        platform_info: Platform information dictionary
        
    Returns:
        Path to extracted binary
        
    Raises:
        Exception: If extraction fails
    """
    print(f"Extracting {archive_path} to {extract_to}")
    
    try:
        if platform_info["extension"] == "zip":
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        else:  # tar.gz
            with tarfile.open(archive_path, 'r:gz') as tar_ref:
                tar_ref.extractall(extract_to)
        
        # Find the binary in extracted files
        binary_name = platform_info["binary_name"]
        
        # Check common locations
        possible_paths = [
            extract_to / binary_name,
            extract_to / "diffai" / binary_name,
            extract_to / f"diffai-{platform_info['platform']}-{platform_info['arch']}" / binary_name,
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        
        # If not found in expected locations, search recursively
        for root, dirs, files in os.walk(extract_to):
            if binary_name in files:
                return Path(root) / binary_name
        
        raise FileNotFoundError(f"Binary '{binary_name}' not found in extracted archive")
        
    except Exception as e:
        raise Exception(f"Failed to extract archive: {e}")


def verify_binary(binary_path: Path) -> bool:
    """
    Verify that the downloaded binary is functional.
    
    Args:
        binary_path: Path to binary to verify
        
    Returns:
        True if binary is functional, False otherwise
    """
    try:
        # Make executable on Unix systems
        if platform.system() != "Windows":
            os.chmod(binary_path, 0o755)
        
        # Test binary execution
        result = subprocess.run(
            [str(binary_path), "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        return result.returncode == 0 and "diffai" in result.stdout.lower()
        
    except Exception:
        return False


def install_binary(target_dir: Optional[Path] = None, force: bool = False) -> bool:
    """
    Download and install diffai binary.
    
    Args:
        target_dir: Directory to install binary (default: package bin directory)
        force: Whether to overwrite existing binary
        
    Returns:
        True if installation succeeded, False otherwise
    """
    try:
        platform_info = get_platform_info()
        print(f"Detected platform: {platform_info['platform']} {platform_info['arch']}")
        
        # Determine target directory
        if target_dir is None:
            # Install to package bin directory
            package_dir = Path(__file__).parent.parent.parent
            target_dir = package_dir / "bin"
        
        target_dir.mkdir(parents=True, exist_ok=True)
        binary_path = target_dir / platform_info["binary_name"]
        
        # Check if binary already exists
        if binary_path.exists() and not force:
            if verify_binary(binary_path):
                print(f"diffai binary already exists and is functional: {binary_path}")
                return True
            else:
                print(f"Existing binary appears corrupted, replacing: {binary_path}")
        
        # Download and install
        download_url = f"{RELEASES_URL}/{platform_info['archive_name']}"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            archive_path = temp_path / platform_info["archive_name"]
            
            # Download archive
            download_file(download_url, archive_path)
            
            # Extract archive
            extracted_binary = extract_archive(archive_path, temp_path, platform_info)
            
            # Verify extracted binary
            if not verify_binary(extracted_binary):
                raise Exception("Downloaded binary failed verification")
            
            # Move to target location
            shutil.move(str(extracted_binary), str(binary_path))
            
            print(f"Successfully installed diffai binary to: {binary_path}")
            return True
            
    except Exception as e:
        print(f"Failed to install diffai binary: {e}", file=sys.stderr)
        print(f"Manual installation available at: {RELEASES_URL}", file=sys.stderr)
        return False


def main() -> int:
    """
    Main entry point for diffai-download-binary command.
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    parser = argparse.ArgumentParser(
        description="Download and install diffai binary",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  diffai-download-binary                    # Install to default location
  diffai-download-binary --force           # Force reinstall 
  diffai-download-binary --target ./bin    # Install to specific directory
  diffai-download-binary --verify          # Verify existing installation
        """
    )
    
    parser.add_argument(
        "--target",
        type=Path,
        help="Target directory for installation (default: package bin directory)"
    )
    
    parser.add_argument(
        "--force", 
        action="store_true",
        help="Force reinstallation even if binary exists"
    )
    
    parser.add_argument(
        "--verify",
        action="store_true", 
        help="Only verify existing installation"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version=f"diffai-installer {PACKAGE_VERSION}"
    )
    
    args = parser.parse_args()
    
    if args.verify:
        # Verify existing installation
        target_dir = args.target or (Path(__file__).parent.parent.parent / "bin")
        platform_info = get_platform_info()
        binary_path = target_dir / platform_info["binary_name"]
        
        if binary_path.exists() and verify_binary(binary_path):
            print(f"✅ diffai binary is properly installed: {binary_path}")
            return 0
        else:
            print(f"❌ diffai binary not found or not functional: {binary_path}")
            return 1
    else:
        # Install binary
        success = install_binary(args.target, args.force)
        return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())