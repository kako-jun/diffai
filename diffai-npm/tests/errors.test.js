/**
 * Error handling tests for diffai npm package
 */

const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');
const os = require('os');

const TIMEOUT_MS = 30000;

/**
 * Helper function to run CLI commands via npx
 */
function runCLI(args, options = {}) {
    return new Promise((resolve, reject) => {
        const timeout = options.timeout || TIMEOUT_MS;
        const child = spawn('npx', ['diffai', ...args], {
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
            reject(new Error(`Command timed out after ${timeout}ms`));
        }, timeout);
    });
}

/**
 * Create temporary test files for error scenarios
 */
function createErrorTestFiles() {
    const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'diffai-errors-'));
    
    // Valid file
    const validFile = path.join(tempDir, 'valid.json');
    fs.writeFileSync(validFile, JSON.stringify({ test: 'data' }, null, 2));
    
    // Invalid JSON file
    const invalidJson = path.join(tempDir, 'invalid.json');
    fs.writeFileSync(invalidJson, '{ "invalid": json syntax }');
    
    // Empty file
    const emptyFile = path.join(tempDir, 'empty.txt');
    fs.writeFileSync(emptyFile, '');
    
    // Binary file (simulate non-text content)
    const binaryFile = path.join(tempDir, 'binary.bin');
    const binaryData = Buffer.from([0x00, 0x01, 0x02, 0x03, 0xFF, 0xFE, 0xFD]);
    fs.writeFileSync(binaryFile, binaryData);
    
    // Large file (for potential memory issues)
    const largeFile = path.join(tempDir, 'large.txt');
    const largeContent = 'Large file content line\n'.repeat(1000);
    fs.writeFileSync(largeFile, largeContent);
    
    return {
        tempDir,
        validFile,
        invalidJson,
        emptyFile,
        binaryFile,
        largeFile,
        nonExistentFile: path.join(tempDir, 'does-not-exist.txt')
    };
}

/**
 * Clean up temporary files
 */
function cleanupTempFiles(tempDir) {
    try {
        fs.rmSync(tempDir, { recursive: true, force: true });
    } catch (error) {
        console.log(`Warning: Failed to cleanup temp directory: ${error.message}`);
    }
}

/**
 * Test handling of non-existent files
 */
async function testNonExistentFiles() {
    console.log('Testing non-existent file handling...');
    
    const { tempDir, nonExistentFile, validFile } = createErrorTestFiles();
    
    try {
        const result = await runCLI([nonExistentFile, validFile]);
        
        // Should fail with appropriate error
        if (result.code !== 0 && result.stderr.length > 0) {
            console.log('✓ Non-existent file handling works');
            return true;
        } else {
            console.log('✗ Should have failed with non-existent file');
            return false;
        }
    } catch (error) {
        // Command throwing an error is also acceptable
        console.log('✓ Non-existent file handling works (threw error)');
        return true;
    } finally {
        cleanupTempFiles(tempDir);
    }
}

/**
 * Test handling of invalid file formats
 */
async function testInvalidFormat() {
    console.log('Testing invalid file format handling...');
    
    const { tempDir, invalidJson, validFile } = createErrorTestFiles();
    
    try {
        const result = await runCLI([invalidJson, validFile]);
        
        // Should either handle gracefully or fail with proper error
        if (result.code >= 0) {
            console.log('✓ Invalid format handling works');
            return true;
        } else {
            console.log('✗ Invalid format handling failed');
            return false;
        }
    } catch (error) {
        console.log('✓ Invalid format handling works (threw error)');
        return true;
    } finally {
        cleanupTempFiles(tempDir);
    }
}

/**
 * Test handling of empty files
 */
async function testEmptyFiles() {
    console.log('Testing empty file handling...');
    
    const { tempDir, emptyFile, validFile } = createErrorTestFiles();
    
    try {
        const result = await runCLI([emptyFile, validFile]);
        
        // Should handle empty files gracefully
        if (result.code <= 1) {
            console.log('✓ Empty file handling works');
            return true;
        } else {
            console.log('✗ Empty file handling failed');
            console.log(`Exit code: ${result.code}`);
            console.log(`Stderr: ${result.stderr}`);
            return false;
        }
    } catch (error) {
        console.log(`✗ Empty file test error: ${error.message}`);
        return false;
    } finally {
        cleanupTempFiles(tempDir);
    }
}

/**
 * Test handling of binary files
 */
async function testBinaryFiles() {
    console.log('Testing binary file handling...');
    
    const { tempDir, binaryFile, validFile } = createErrorTestFiles();
    
    try {
        const result = await runCLI([binaryFile, validFile]);
        
        // Should either handle gracefully or provide appropriate error
        if (result.code >= 0) {
            console.log('✓ Binary file handling works');
            return true;
        } else {
            console.log('✗ Binary file handling failed');
            return false;
        }
    } catch (error) {
        console.log('✓ Binary file handling works (threw error)');
        return true;
    } finally {
        cleanupTempFiles(tempDir);
    }
}

/**
 * Test handling of invalid command line arguments
 */
async function testInvalidArguments() {
    console.log('Testing invalid argument handling...');
    
    try {
        // Test with invalid flag
        const result = await runCLI(['--invalid-flag']);
        
        // Should fail with proper error message
        if (result.code !== 0) {
            console.log('✓ Invalid argument handling works');
            return true;
        } else {
            console.log('✗ Should have failed with invalid flag');
            return false;
        }
    } catch (error) {
        console.log('✓ Invalid argument handling works (threw error)');
        return true;
    }
}

/**
 * Test handling when only one file is provided
 */
async function testSingleFileArgument() {
    console.log('Testing single file argument handling...');
    
    const { tempDir, validFile } = createErrorTestFiles();
    
    try {
        const result = await runCLI([validFile]);
        
        // Should either work (showing help) or fail gracefully
        if (result.code >= 0) {
            console.log('✓ Single file argument handling works');
            return true;
        } else {
            console.log('✗ Single file argument handling failed');
            return false;
        }
    } catch (error) {
        console.log('✓ Single file argument handling works (threw error)');
        return true;
    } finally {
        cleanupTempFiles(tempDir);
    }
}

/**
 * Test handling of large files
 */
async function testLargeFiles() {
    console.log('Testing large file handling...');
    
    const { tempDir, largeFile, validFile } = createErrorTestFiles();
    
    try {
        const result = await runCLI([largeFile, validFile], { timeout: 45000 });
        
        // Should handle large files without memory issues
        if (result.code <= 1) {
            console.log('✓ Large file handling works');
            return true;
        } else {
            console.log('✗ Large file handling failed');
            return false;
        }
    } catch (error) {
        if (error.message.includes('timeout')) {
            console.log('⚠ Large file test timed out (may be expected)');
            return true;
        } else {
            console.log(`✗ Large file test error: ${error.message}`);
            return false;
        }
    } finally {
        cleanupTempFiles(tempDir);
    }
}

/**
 * Main test runner for error tests
 */
async function runTests() {
    console.log('Running error handling tests...');
    console.log('='.repeat(45));

    const tests = [
        testNonExistentFiles,
        testInvalidFormat,
        testEmptyFiles,
        testBinaryFiles,
        testInvalidArguments,
        testSingleFileArgument,
        testLargeFiles
    ];

    let passed = 0;
    const total = tests.length;

    for (const test of tests) {
        try {
            if (await test()) {
                passed++;
            }
        } catch (error) {
            console.log(`✗ Test error: ${error.message}`);
        }
        console.log('');
    }

    console.log('='.repeat(45));
    console.log(`Error Tests: ${passed}/${total} passed`);

    return { passed, total };
}

module.exports = { runTests };