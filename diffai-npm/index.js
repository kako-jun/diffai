#!/usr/bin/env node

const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

/**
 * Main entry point for diffai npm package
 * Spawns the native Rust binary and proxies all arguments
 */

// Determine the binary name based on platform
let binaryName = 'diffai';
if (process.platform === 'win32') {
  binaryName = 'diffai.exe';
}

const binaryPath = path.join(__dirname, 'bin', binaryName);

// Check if binary exists
if (!fs.existsSync(binaryPath)) {
  console.error('diffai binary not found. Please run "npm install diffai" to download the binary.');
  console.error('Manual installation: https://github.com/diffai-team/diffai/releases');
  process.exit(1);
}

// Spawn the binary with all arguments passed through
const child = spawn(binaryPath, process.argv.slice(2), {
  stdio: 'inherit',
});

// Handle process events
child.on('error', (error) => {
  console.error('Failed to start diffai:', error.message);
  process.exit(1);
});

child.on('close', (code) => {
  process.exit(code);
});

// Forward signals to child process
process.on('SIGINT', () => {
  child.kill('SIGINT');
});

process.on('SIGTERM', () => {
  child.kill('SIGTERM');
});