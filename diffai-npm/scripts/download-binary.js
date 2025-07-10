#!/usr/bin/env node

const https = require('https');
const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

/**
 * Post-install script to download platform-specific diffai binary
 * Automatically detects platform and downloads appropriate binary from GitHub releases
 */

const PACKAGE_VERSION = require('../package.json').version;
const GITHUB_REPO = 'diffai-team/diffai';
const RELEASES_URL = `https://github.com/${GITHUB_REPO}/releases/download/v${PACKAGE_VERSION}`;

// Platform and architecture detection
function getPlatformInfo() {
  const platform = process.platform;
  const arch = process.arch;
  
  let platformName, archName, extension, extractCmd;
  
  switch (platform) {
    case 'win32':
      platformName = 'windows';
      archName = 'x86_64';
      extension = 'zip';
      extractCmd = (archive, destination) => 
        `powershell -command "Expand-Archive -Path '${archive}' -DestinationPath '${destination}' -Force"`;
      break;
    case 'darwin':
      platformName = 'macos';
      archName = arch === 'arm64' ? 'aarch64' : 'x86_64';
      extension = 'tar.gz';
      extractCmd = (archive, destination) => 
        `tar -xzf "${archive}" -C "${destination}"`;
      break;
    case 'linux':
      platformName = 'linux';
      archName = arch === 'arm64' ? 'aarch64' : 'x86_64';
      extension = 'tar.gz';
      extractCmd = (archive, destination) => 
        `tar -xzf "${archive}" -C "${destination}"`;
      break;
    default:
      throw new Error(`Unsupported platform: ${platform}`);
  }
  
  return {
    platformName,
    archName,
    extension,
    extractCmd,
    fileName: `diffai-${platformName}-${archName}.${extension}`,
    binaryName: platform === 'win32' ? 'diffai.exe' : 'diffai'
  };
}

// Download file from URL
function downloadFile(url, destination) {
  return new Promise((resolve, reject) => {
    const file = fs.createWriteStream(destination);
    
    https.get(url, (response) => {
      if (response.statusCode === 302 || response.statusCode === 301) {
        // Follow redirect
        return downloadFile(response.headers.location, destination)
          .then(resolve)
          .catch(reject);
      }
      
      if (response.statusCode !== 200) {
        reject(new Error(`Download failed: HTTP ${response.statusCode}`));
        return;
      }
      
      response.pipe(file);
      
      file.on('finish', () => {
        file.close();
        resolve();
      });
      
      file.on('error', (err) => {
        fs.unlink(destination, () => {}); // Clean up on error
        reject(err);
      });
    }).on('error', (err) => {
      reject(err);
    });
  });
}

async function downloadAndExtractBinary() {
  try {
    const platformInfo = getPlatformInfo();
    const binDir = path.join(__dirname, '..', 'bin');
    const tempDir = path.join(__dirname, '..', 'temp');
    
    // Create directories
    if (!fs.existsSync(binDir)) {
      fs.mkdirSync(binDir, { recursive: true });
    }
    if (!fs.existsSync(tempDir)) {
      fs.mkdirSync(tempDir, { recursive: true });
    }
    
    const binaryPath = path.join(binDir, platformInfo.binaryName);
    
    // Skip download if binary already exists
    if (fs.existsSync(binaryPath)) {
      console.log('diffai binary already exists, skipping download.');
      return;
    }
    
    const downloadUrl = `${RELEASES_URL}/${platformInfo.fileName}`;
    const archivePath = path.join(tempDir, platformInfo.fileName);
    
    console.log(`Downloading diffai binary from: ${downloadUrl}`);
    console.log(`Platform: ${platformInfo.platformName}, Architecture: ${platformInfo.archName}`);
    
    // Download the archive
    await downloadFile(downloadUrl, archivePath);
    console.log('Download completed, extracting...');
    
    // Extract the archive
    try {
      execSync(platformInfo.extractCmd(archivePath, tempDir), { stdio: 'pipe' });
    } catch (error) {
      throw new Error(`Failed to extract archive: ${error.message}`);
    }
    
    // Find and move the binary
    const extractedBinaryPath = path.join(tempDir, platformInfo.binaryName);
    if (!fs.existsSync(extractedBinaryPath)) {
      throw new Error(`Binary not found in extracted archive: ${extractedBinaryPath}`);
    }
    
    // Move binary to bin directory
    fs.renameSync(extractedBinaryPath, binaryPath);
    
    // Make binary executable on Unix systems
    if (process.platform !== 'win32') {
      fs.chmodSync(binaryPath, '755');
    }
    
    // Clean up temporary files
    fs.rmSync(tempDir, { recursive: true, force: true });
    
    console.log('diffai binary installation completed successfully!');
    
  } catch (error) {
    console.warn('Failed to download diffai binary:', error.message);
    console.warn('You can install diffai manually from: https://github.com/diffai-team/diffai/releases');
    console.warn('The package will still work if diffai is available in your system PATH.');
    
    // Create empty bin directory to avoid errors
    const binDir = path.join(__dirname, '..', 'bin');
    if (!fs.existsSync(binDir)) {
      fs.mkdirSync(binDir, { recursive: true });
    }
    
    // Don't fail the npm install process
    process.exit(0);
  }
}

// Only run if this script is called directly (not required as module)
if (require.main === module) {
  downloadAndExtractBinary();
}