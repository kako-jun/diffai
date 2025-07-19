/**
 * Binary direct execution tests for diffai npm package
 */

const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');
const os = require('os');

const TIMEOUT_MS = 30000;

/**
 * Get expected binary name based on platform
 */
function getExpectedBinaryName() {
    const platform = os.platform();
    const arch = os.arch();
    
    let platformName = platform;
    if (platform === 'darwin') platformName = 'macos';
    if (platform === 'win32') platformName = 'windows';
    
    let archName = arch;
    if (arch === 'x64') archName = 'x86_64';
    if (arch === 'arm64') archName = 'aarch64';
    
    const extension = platform === 'win32' ? '.exe' : '';
    return `diffai-${platformName}-${archName}${extension}`;
}

/**
 * Helper function to run binary directly
 */
function runBinary(binaryPath, args, options = {}) {
    return new Promise((resolve, reject) => {
        const timeout = options.timeout || TIMEOUT_MS;
        const child = spawn(binaryPath, args, {
            stdio: ['pipe', 'pipe', 'pipe'],
            timeout: timeout
        });

        let stdout = '';
        let stderr = '';

        child.stdout.on('data', (data) => {
            stdout += data.toString();
        });

        child.stderr.on('data', (data) => {
            stderr += data.toString();
        });

        child.on('close', (code) => {
            resolve({ code, stdout, stderr });
        });

        child.on('error', (error) => {
            reject(error);
        });

        setTimeout(() => {
            child.kill();
            reject(new Error(`Binary execution timed out after ${timeout}ms`));
        }, timeout);
    });
}

/**
 * Test binary existence
 */
async function testBinaryExists() {
    console.log('Testing binary existence...');
    
    const binDir = path.join(__dirname, '../bin');
    const expectedBinary = getExpectedBinaryName();
    const binaryPath = path.join(binDir, expectedBinary);
    
    console.log(`Expected binary: ${expectedBinary}`);
    console.log(`Binary path: ${binaryPath}`);
    
    if (fs.existsSync(binaryPath)) {
        console.log('✓ Binary exists');
        return { exists: true, path: binaryPath };
    } else {
        // Try to find any diffai binary in bin directory
        if (fs.existsSync(binDir)) {
            const files = fs.readdirSync(binDir);
            const diffaiBinaries = files.filter(file => file.startsWith('diffai'));
            
            if (diffaiBinaries.length > 0) {
                console.log(`ℹ Found other binaries: ${diffaiBinaries.join(', ')}`);
                const altBinary = path.join(binDir, diffaiBinaries[0]);
                console.log(`✓ Using alternative binary: ${diffaiBinaries[0]}`);
                return { exists: true, path: altBinary };
            }
        }
        
        console.log('✗ Binary not found');
        return { exists: false, path: null };
    }
}

/**
 * Test binary direct execution
 */
async function testBinaryExecution(binaryPath) {
    console.log('Testing binary direct execution...');
    
    try {
        // Test version
        const versionResult = await runBinary(binaryPath, ['--version']);
        if (versionResult.code === 0 && versionResult.stdout.includes('diffai')) {
            console.log('✓ Binary version command works');
        } else {
            console.log('✗ Binary version command failed');
            return false;
        }
        
        // Test help
        const helpResult = await runBinary(binaryPath, ['--help']);
        if (helpResult.code === 0 && helpResult.stdout.includes('diffai')) {
            console.log('✓ Binary help command works');
        } else {
            console.log('✗ Binary help command failed');
            return false;
        }
        
        return true;
    } catch (error) {
        console.log(`✗ Binary execution error: ${error.message}`);
        return false;
    }
}

/**
 * Test binary path resolution
 */
async function testPathResolution(binaryPath) {
    console.log('Testing binary path resolution...');
    
    try {
        // Test absolute path
        const absoluteResult = await runBinary(path.resolve(binaryPath), ['--version']);
        if (absoluteResult.code === 0) {
            console.log('✓ Absolute path resolution works');
        } else {
            console.log('✗ Absolute path resolution failed');
            return false;
        }
        
        // Test relative path (if we're in the right directory)
        const relativeDir = path.dirname(binaryPath);
        const binaryName = path.basename(binaryPath);
        const relativePath = path.join('.', binaryName);
        
        // Change to binary directory temporarily
        const originalCwd = process.cwd();
        process.chdir(relativeDir);
        
        try {
            const relativeResult = await runBinary(relativePath, ['--version']);
            if (relativeResult.code === 0) {
                console.log('✓ Relative path resolution works');
            } else {
                console.log('✗ Relative path resolution failed');
                return false;
            }
        } finally {
            process.chdir(originalCwd);
        }
        
        return true;
    } catch (error) {
        console.log(`✗ Path resolution error: ${error.message}`);
        return false;
    }
}

/**
 * Main test runner for binary tests
 */
async function runTests() {
    console.log('Running binary direct execution tests...');
    console.log('='.repeat(50));

    let passed = 0;
    let total = 0;

    // Test 1: Binary existence
    total++;
    const binaryCheck = await testBinaryExists();
    if (binaryCheck.exists) {
        passed++;
    } else {
        console.log('⚠️  Skipping binary execution tests (binary not found)');
        console.log('='.repeat(50));
        console.log(`Binary Tests: ${passed}/${total} passed`);
        return { passed, total };
    }

    console.log('');

    // Test 2: Binary execution
    total++;
    if (await testBinaryExecution(binaryCheck.path)) {
        passed++;
    }

    console.log('');

    // Test 3: Path resolution
    total++;
    if (await testPathResolution(binaryCheck.path)) {
        passed++;
    }

    console.log('');
    console.log('='.repeat(50));
    console.log(`Binary Tests: ${passed}/${total} passed`);

    return { passed, total };
}

module.exports = { runTests };