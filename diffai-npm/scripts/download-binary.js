/**
 * Legacy download script for backward compatibility.
 * 
 * This script is kept for compatibility with older build systems.
 * New implementations should use download-all-binaries.js instead.
 */

const { downloadAllBinaries } = require('./download-all-binaries');

/**
 * Download binary for the current platform (legacy interface)
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