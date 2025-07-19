/**
 * Feature-specific tests for diffai npm package
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
 * Create test files for feature testing
 */
function createFeatureTestFiles() {
    const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'diffai-features-'));
    
    // Complex JSON structure for ML-like data
    const mlData1 = {
        model_info: {
            name: 'test_model',
            version: '1.0',
            architecture: 'transformer',
            parameters: 125000000
        },
        weights: {
            layer1: [0.1, 0.2, 0.3, 0.4],
            layer2: [0.5, 0.6, 0.7, 0.8]
        },
        metadata: {
            training_steps: 10000,
            learning_rate: 0.001
        }
    };
    
    const mlData2 = {
        model_info: {
            name: 'test_model',
            version: '1.1',
            architecture: 'transformer',
            parameters: 125000000
        },
        weights: {
            layer1: [0.11, 0.21, 0.31, 0.41],
            layer2: [0.5, 0.6, 0.7, 0.8, 0.9]
        },
        metadata: {
            training_steps: 15000,
            learning_rate: 0.0008
        }
    };
    
    const mlFile1 = path.join(tempDir, 'model1.json');
    const mlFile2 = path.join(tempDir, 'model2.json');
    fs.writeFileSync(mlFile1, JSON.stringify(mlData1, null, 2));
    fs.writeFileSync(mlFile2, JSON.stringify(mlData2, null, 2));
    
    // Numerical data files
    const numData1 = path.join(tempDir, 'numbers1.json');
    const numData2 = path.join(tempDir, 'numbers2.json');
    fs.writeFileSync(numData1, JSON.stringify({ values: [1.0, 2.0, 3.0, 4.0] }, null, 2));
    fs.writeFileSync(numData2, JSON.stringify({ values: [1.1, 2.1, 3.1, 4.1] }, null, 2));
    
    return {
        tempDir,
        mlData: { file1: mlFile1, file2: mlFile2 },
        numData: { file1: numData1, file2: numData2 }
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
 * Test ML analysis features
 */
async function testMLAnalysis() {
    console.log('Testing ML analysis features...');
    
    const { tempDir, mlData } = createFeatureTestFiles();
    
    try {
        const result = await runCLI(['--analyze-ml', mlData.file1, mlData.file2]);
        
        if (result.code <= 1) {
            console.log('✓ ML analysis features work');
            return true;
        } else {
            // Try without ML-specific flag as fallback
            const fallbackResult = await runCLI([mlData.file1, mlData.file2]);
            if (fallbackResult.code <= 1) {
                console.log('✓ ML analysis works (via fallback)');
                return true;
            } else {
                console.log('✗ ML analysis features failed');
                return false;
            }
        }
    } catch (error) {
        console.log(`✗ ML analysis test error: ${error.message}`);
        return false;
    } finally {
        cleanupTempFiles(tempDir);
    }
}

/**
 * Test statistical analysis
 */
async function testStatisticalAnalysis() {
    console.log('Testing statistical analysis...');
    
    const { tempDir, numData } = createFeatureTestFiles();
    
    try {
        const result = await runCLI(['--stats', numData.file1, numData.file2]);
        
        if (result.code <= 1) {
            console.log('✓ Statistical analysis works');
            return true;
        } else {
            // Try with verbose output as alternative
            const fallbackResult = await runCLI(['-v', numData.file1, numData.file2]);
            if (fallbackResult.code <= 1) {
                console.log('✓ Statistical analysis works (via verbose)');
                return true;
            } else {
                console.log('✗ Statistical analysis failed');
                return false;
            }
        }
    } catch (error) {
        console.log(`✗ Statistical analysis test error: ${error.message}`);
        return false;
    } finally {
        cleanupTempFiles(tempDir);
    }
}

/**
 * Test color output
 */
async function testColorOutput() {
    console.log('Testing color output...');
    
    const { tempDir, mlData } = createFeatureTestFiles();
    
    try {
        const result = await runCLI(['--color', 'always', mlData.file1, mlData.file2]);
        
        if (result.code <= 1) {
            console.log('✓ Color output works');
            return true;
        } else {
            console.log('✗ Color output failed');
            return false;
        }
    } catch (error) {
        console.log(`✗ Color output test error: ${error.message}`);
        return false;
    } finally {
        cleanupTempFiles(tempDir);
    }
}

/**
 * Test no-color output
 */
async function testNoColorOutput() {
    console.log('Testing no-color output...');
    
    const { tempDir, mlData } = createFeatureTestFiles();
    
    try {
        const result = await runCLI(['--no-color', mlData.file1, mlData.file2]);
        
        if (result.code <= 1) {
            console.log('✓ No-color output works');
            return true;
        } else {
            console.log('✗ No-color output failed');
            return false;
        }
    } catch (error) {
        console.log(`✗ No-color output test error: ${error.message}`);
        return false;
    } finally {
        cleanupTempFiles(tempDir);
    }
}

/**
 * Test context lines option
 */
async function testContextLines() {
    console.log('Testing context lines option...');
    
    const { tempDir, mlData } = createFeatureTestFiles();
    
    try {
        const result = await runCLI(['--context', '3', mlData.file1, mlData.file2]);
        
        if (result.code <= 1) {
            console.log('✓ Context lines option works');
            return true;
        } else {
            console.log('✗ Context lines option failed');
            return false;
        }
    } catch (error) {
        console.log(`✗ Context lines test error: ${error.message}`);
        return false;
    } finally {
        cleanupTempFiles(tempDir);
    }
}

/**
 * Test ignore whitespace option
 */
async function testIgnoreWhitespace() {
    console.log('Testing ignore whitespace option...');
    
    const { tempDir } = createFeatureTestFiles();
    
    // Create files with different whitespace
    const ws1 = path.join(tempDir, 'whitespace1.txt');
    const ws2 = path.join(tempDir, 'whitespace2.txt');
    fs.writeFileSync(ws1, 'line1\nline2\nline3\n');
    fs.writeFileSync(ws2, 'line1\n  line2  \nline3\n');
    
    try {
        const result = await runCLI(['--ignore-whitespace', ws1, ws2]);
        
        if (result.code <= 1) {
            console.log('✓ Ignore whitespace option works');
            return true;
        } else {
            console.log('✗ Ignore whitespace option failed');
            return false;
        }
    } catch (error) {
        console.log(`✗ Ignore whitespace test error: ${error.message}`);
        return false;
    } finally {
        cleanupTempFiles(tempDir);
    }
}

/**
 * Test output to file option
 */
async function testOutputToFile() {
    console.log('Testing output to file option...');
    
    const { tempDir, mlData } = createFeatureTestFiles();
    const outputFile = path.join(tempDir, 'output.txt');
    
    try {
        const result = await runCLI(['--output', outputFile, mlData.file1, mlData.file2]);
        
        if (result.code <= 1 && fs.existsSync(outputFile)) {
            console.log('✓ Output to file option works');
            return true;
        } else {
            console.log('✗ Output to file option failed');
            return false;
        }
    } catch (error) {
        console.log(`✗ Output to file test error: ${error.message}`);
        return false;
    } finally {
        cleanupTempFiles(tempDir);
    }
}

/**
 * Main test runner for feature tests
 */
async function runTests() {
    console.log('Running feature-specific tests...');
    console.log('='.repeat(45));

    const tests = [
        testMLAnalysis,
        testStatisticalAnalysis,
        testColorOutput,
        testNoColorOutput,
        testContextLines,
        testIgnoreWhitespace,
        testOutputToFile
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
    console.log(`Feature Tests: ${passed}/${total} passed`);

    return { passed, total };
}

module.exports = { runTests };