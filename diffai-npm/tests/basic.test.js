/**
 * Basic functionality tests for diffai npm package
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
 * Create temporary test files
 */
function createTempTestFiles() {
    const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'diffai-test-'));
    
    // Create two simple text files for basic diff testing
    const file1 = path.join(tempDir, 'file1.txt');
    const file2 = path.join(tempDir, 'file2.txt');
    
    fs.writeFileSync(file1, 'Hello\nWorld\nOriginal\n');
    fs.writeFileSync(file2, 'Hello\nWorld\nModified\n');
    
    return { tempDir, file1, file2 };
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
 * Test basic file comparison
 */
async function testBasicDiff() {
    console.log('Testing basic file comparison...');
    
    const { tempDir, file1, file2 } = createTempTestFiles();
    
    try {
        const result = await runCLI([file1, file2]);
        
        // Basic diff should succeed (exit code 0 or 1 are both valid for diff tools)
        if (result.code <= 1) {
            console.log('✓ Basic diff works');
            return true;
        } else {
            console.log('✗ Basic diff failed');
            console.log(`Exit code: ${result.code}`);
            console.log(`Stderr: ${result.stderr}`);
            return false;
        }
    } catch (error) {
        console.log(`✗ Basic diff error: ${error.message}`);
        return false;
    } finally {
        cleanupTempFiles(tempDir);
    }
}

/**
 * Test JSON output format
 */
async function testJsonOutput() {
    console.log('Testing JSON output format...');
    
    const { tempDir, file1, file2 } = createTempTestFiles();
    
    try {
        const result = await runCLI(['--output-format', 'json', file1, file2]);
        
        if (result.code <= 1) {
            // Try to parse JSON output
            try {
                JSON.parse(result.stdout);
                console.log('✓ JSON output format works');
                return true;
            } catch (parseError) {
                console.log('✗ JSON output is not valid JSON');
                return false;
            }
        } else {
            console.log('✗ JSON output format failed');
            return false;
        }
    } catch (error) {
        console.log(`✗ JSON output test error: ${error.message}`);
        return false;
    } finally {
        cleanupTempFiles(tempDir);
    }
}

/**
 * Test YAML output format
 */
async function testYamlOutput() {
    console.log('Testing YAML output format...');
    
    const { tempDir, file1, file2 } = createTempTestFiles();
    
    try {
        const result = await runCLI(['--output-format', 'yaml', file1, file2]);
        
        if (result.code <= 1 && result.stdout.trim().length > 0) {
            console.log('✓ YAML output format works');
            return true;
        } else {
            console.log('✗ YAML output format failed');
            return false;
        }
    } catch (error) {
        console.log(`✗ YAML output test error: ${error.message}`);
        return false;
    } finally {
        cleanupTempFiles(tempDir);
    }
}

/**
 * Test verbosity flag
 */
async function testVerbosity() {
    console.log('Testing verbosity flag...');
    
    const { tempDir, file1, file2 } = createTempTestFiles();
    
    try {
        const result = await runCLI(['-v', file1, file2]);
        
        if (result.code <= 1) {
            console.log('✓ Verbosity flag works');
            return true;
        } else {
            console.log('✗ Verbosity flag failed');
            return false;
        }
    } catch (error) {
        console.log(`✗ Verbosity test error: ${error.message}`);
        return false;
    } finally {
        cleanupTempFiles(tempDir);
    }
}

/**
 * Test quiet flag
 */
async function testQuietMode() {
    console.log('Testing quiet mode...');
    
    const { tempDir, file1, file2 } = createTempTestFiles();
    
    try {
        const result = await runCLI(['--quiet', file1, file2]);
        
        if (result.code <= 1) {
            console.log('✓ Quiet mode works');
            return true;
        } else {
            console.log('✗ Quiet mode failed');
            return false;
        }
    } catch (error) {
        console.log(`✗ Quiet mode test error: ${error.message}`);
        return false;
    } finally {
        cleanupTempFiles(tempDir);
    }
}

/**
 * Main test runner for basic tests
 */
async function runTests() {
    console.log('Running basic functionality tests...');
    console.log('='.repeat(45));

    const tests = [
        testBasicDiff,
        testJsonOutput,
        testYamlOutput,
        testVerbosity,
        testQuietMode
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
    console.log(`Basic Tests: ${passed}/${total} passed`);

    return { passed, total };
}

module.exports = { runTests };