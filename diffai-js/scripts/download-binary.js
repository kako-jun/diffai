/**
 * Legacy download script for backward compatibility.
 * 
 * This script is kept for compatibility with older build systems.
 * New implementations should use download-all-binaries.js instead.
 */

const fs = require('fs');
const path = require('path');
const https = require('https');
const { downloadAllBinaries } = require('./download-all-binaries');

function getPlatformInfo() {
    const platform = process.platform;
    const arch = process.arch;
    
    if (platform === 'linux' && arch === 'x64') {
        return { platform: 'linux', arch: 'x86_64' };
    } else if (platform === 'darwin') {
        if (arch === 'x64') {
            return { platform: 'macos', arch: 'x86_64' };
        } else if (arch === 'arm64') {
            return { platform: 'macos', arch: 'aarch64' };
        }
    } else if (platform === 'win32' && arch === 'x64') {
        return { platform: 'windows', arch: 'x86_64' };
    }
    
    throw new Error(`Unsupported platform: ${platform}-${arch}`);
}

function downloadFile(url, dest) {
    return new Promise((resolve, reject) => {
        console.log(`Downloading ${path.basename(dest)}...`);
        const file = fs.createWriteStream(dest);
        
        https.get(url, (response) => {
            if (response.statusCode === 302 || response.statusCode === 301) {
                // Follow redirect
                downloadFile(response.headers.location, dest).then(resolve).catch(reject);
                return;
            }
            
            if (response.statusCode !== 200) {
                reject(new Error(`HTTP ${response.statusCode}: ${response.statusMessage}`));
                return;
            }
            
            response.pipe(file);
            
            file.on('finish', () => {
                file.close();
                resolve();
            });
            
            file.on('error', (err) => {
                fs.unlink(dest, () => {}); // Delete partial file
                reject(err);
            });
        }).on('error', reject);
    });
}

/**
 * Download binary for the current platform (legacy interface)
 * Downloads from https://github.com/kako-jun/diffai/releases
 */
async function downloadBinary() {
    console.log('Using legacy download-binary.js - consider upgrading to download-all-binaries.js');
    
    try {
        await downloadAllBinaries();
        console.log('✅ Binary download completed via legacy interface');
    } catch (error) {
        console.error('❌ Binary download failed:', error.message);
        process.exit(1);
    }
}

// Run if called directly
if (require.main === module) {
    downloadBinary().catch(error => {
        console.error('❌ Download failed:', error);
        process.exit(1);
    });
}

module.exports = { downloadBinary };