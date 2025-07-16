#!/usr/bin/env node

const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

// Determine the platform-specific binary name and directory
function getPlatformInfo() {
  const platform = process.platform;
  const arch = process.arch;
  
  if (platform === 'win32') {
    return { subdir: 'win32-x64', binaryName: 'diffai.exe' };
  } else if (platform === 'darwin') {
    if (arch === 'arm64') {
      return { subdir: 'darwin-arm64', binaryName: 'diffai' };
    } else {
      return { subdir: 'darwin-x64', binaryName: 'diffai' };
    }
  } else if (platform === 'linux') {
    if (arch === 'arm64') {
      return { subdir: 'linux-arm64', binaryName: 'diffai' };
    } else {
      return { subdir: 'linux-x64', binaryName: 'diffai' };
    }
  } else {
    throw new Error(`Unsupported platform: ${platform}-${arch}`);
  }
}

// Get platform-specific binary path
const platformInfo = getPlatformInfo();
const binaryPath = path.join(__dirname, 'bin', platformInfo.subdir, platformInfo.binaryName);

// Check if the binary exists
if (!fs.existsSync(binaryPath)) {
  console.error(`Error: Binary not found at ${binaryPath}`);
  console.error(`Platform: ${process.platform}-${process.arch}`);
  console.error('Expected platform-specific binary not found.');
  console.error('This might indicate a packaging issue. Please report this at:');
  console.error('https://github.com/kako-jun/diffai/issues');
  process.exit(1);
}

// Spawn the diffai process with arguments
const child = spawn(binaryPath, process.argv.slice(2), {
  stdio: 'inherit',
});

child.on('close', (code) => {
  process.exit(code);
});

child.on('error', (err) => {
  console.error(`Failed to start diffai: ${err.message}`);
  process.exit(1);
});