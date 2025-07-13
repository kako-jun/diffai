"""
Binary installer for diffai Python package.

This module handles downloading and installing the diffai binary
when the package is installed via pip.
"""

import os
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path


def get_platform_info():
    """Get platform-specific information for binary download."""
    system = platform.system().lower()
    machine = platform.machine().lower()
    
    if system == "linux":
        if machine in ("x86_64", "amd64"):
            return "linux-x86_64", "diffai"
        else:
            raise RuntimeError(f"Unsupported Linux architecture: {machine}")
    elif system == "darwin":
        if machine == "arm64":
            return "macos-aarch64", "diffai"
        elif machine in ("x86_64", "amd64"):
            return "macos-x86_64", "diffai"
        else:
            raise RuntimeError(f"Unsupported macOS architecture: {machine}")
    elif system == "windows":
        if machine in ("x86_64", "amd64"):
            return "windows-x86_64", "diffai.exe"
        else:
            raise RuntimeError(f"Unsupported Windows architecture: {machine}")
    else:
        raise RuntimeError(f"Unsupported operating system: {system}")


def get_latest_release_info():
    """Get information about the latest GitHub release."""
    try:
        import json
        
        url = "https://api.github.com/repos/kako-jun/diffai/releases/latest"
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode())
            
        return data["tag_name"], data["assets"]
    except Exception as e:
        raise RuntimeError(f"Failed to get release information: {e}")


def download_binary(version=None):
    """Download the diffai binary for the current platform."""
    platform_name, binary_name = get_platform_info()
    
    if version is None:
        version, assets = get_latest_release_info()
    else:
        # For specific version, construct asset URL manually
        assets = None
    
    # Construct download URL
    if assets:
        # Find the correct asset
        asset_name = f"diffai-{platform_name}.tar.gz"
        if platform_name.startswith("windows"):
            asset_name = f"diffai-{platform_name}.zip"
            
        asset_url = None
        for asset in assets:
            if asset["name"] == asset_name:
                asset_url = asset["browser_download_url"]
                break
                
        if not asset_url:
            raise RuntimeError(f"Binary not found for platform: {platform_name}")
    else:
        # Fallback URL construction
        base_url = "https://github.com/kako-jun/diffai/releases/download"
        if platform_name.startswith("windows"):
            asset_name = f"diffai-{platform_name}.zip"
        else:
            asset_name = f"diffai-{platform_name}.tar.gz"
        asset_url = f"{base_url}/{version}/{asset_name}"
    
    # Download to temporary location
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        archive_path = temp_path / asset_name
        
        print(f"Downloading {asset_url}...")
        urllib.request.urlretrieve(asset_url, archive_path)
        
        # Extract archive
        if asset_name.endswith(".tar.gz"):
            subprocess.run(["tar", "-xzf", archive_path, "-C", temp_path], check=True)
        elif asset_name.endswith(".zip"):
            import zipfile
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(temp_path)
        
        # Find the binary
        binary_path = temp_path / binary_name
        if not binary_path.exists():
            raise RuntimeError(f"Binary not found in archive: {binary_name}")
        
        return binary_path


def install_binary():
    """Install the diffai binary to the package directory."""
    try:
        # Get package directory
        package_dir = Path(__file__).parent.parent.parent
        binary_dir = package_dir / "bin"
        binary_dir.mkdir(exist_ok=True)
        
        # Download binary
        temp_binary = download_binary()
        
        # Copy to package directory
        platform_name, binary_name = get_platform_info()
        target_path = binary_dir / binary_name
        
        shutil.copy2(temp_binary, target_path)
        
        # Make executable on Unix-like systems
        if not platform_name.startswith("windows"):
            os.chmod(target_path, 0o755)
        
        print(f"Successfully installed diffai binary to {target_path}")
        return target_path
        
    except Exception as e:
        print(f"Warning: Failed to install diffai binary: {e}")
        print("The Python API will still work if diffai is available in PATH")
        return None


if __name__ == "__main__":
    install_binary()