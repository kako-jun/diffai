/**
 * File format support tests for diffai npm package
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
 * Create test files for different formats
 */
function createFormatTestFiles() {
    const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'diffai-formats-'));
    
    // JSON files
    const json1 = path.join(tempDir, 'data1.json');
    const json2 = path.join(tempDir, 'data2.json');
    fs.writeFileSync(json1, JSON.stringify({ name: 'test', version: '1.0', data: [1, 2, 3] }, null, 2));
    fs.writeFileSync(json2, JSON.stringify({ name: 'test', version: '1.1', data: [1, 2, 3, 4] }, null, 2));
    
    // YAML files
    const yaml1 = path.join(tempDir, 'config1.yaml');
    const yaml2 = path.join(tempDir, 'config2.yaml');
    fs.writeFileSync(yaml1, 'name: test\nversion: 1.0\ndata:\n  - 1\n  - 2\n  - 3\n');
    fs.writeFileSync(yaml2, 'name: test\nversion: 1.1\ndata:\n  - 1\n  - 2\n  - 3\n  - 4\n');
    
    // CSV files
    const csv1 = path.join(tempDir, 'data1.csv');
    const csv2 = path.join(tempDir, 'data2.csv');
    fs.writeFileSync(csv1, 'name,age,city\nJohn,25,NYC\nJane,30,LA\n');
    fs.writeFileSync(csv2, 'name,age,city\nJohn,26,NYC\nJane,30,LA\nBob,35,SF\n');
    
    // Text files
    const txt1 = path.join(tempDir, 'readme1.txt');
    const txt2 = path.join(tempDir, 'readme2.txt');
    fs.writeFileSync(txt1, 'This is a test file.\nLine 2\nLine 3\n');
    fs.writeFileSync(txt2, 'This is a test file.\nLine 2 modified\nLine 3\nLine 4\n');
    
    return {
        tempDir,
        json: { file1: json1, file2: json2 },
        yaml: { file1: yaml1, file2: yaml2 },
        csv: { file1: csv1, file2: csv2 },
        txt: { file1: txt1, file2: txt2 }
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
 * Test JSON format support
 */
async function testJsonFormat() {
    console.log('Testing JSON format support...');
    
    const { tempDir, json } = createFormatTestFiles();
    
    try {
        const result = await runCLI([json.file1, json.file2]);
        
        if (result.code <= 1) {
            console.log('✓ JSON format support works');
            return true;
        } else {
            console.log('✗ JSON format support failed');
            console.log(`Exit code: ${result.code}`);
            console.log(`Stderr: ${result.stderr}`);
            return false;
        }
    } catch (error) {
        console.log(`✗ JSON format test error: ${error.message}`);
        return false;
    } finally {
        cleanupTempFiles(tempDir);
    }
}

/**
 * Test YAML format support
 */
async function testYamlFormat() {
    console.log('Testing YAML format support...');
    
    const { tempDir, yaml } = createFormatTestFiles();
    
    try {
        const result = await runCLI([yaml.file1, yaml.file2]);
        
        if (result.code <= 1) {
            console.log('✓ YAML format support works');
            return true;
        } else {
            console.log('✗ YAML format support failed');
            console.log(`Exit code: ${result.code}`);
            console.log(`Stderr: ${result.stderr}`);
            return false;
        }
    } catch (error) {
        console.log(`✗ YAML format test error: ${error.message}`);
        return false;
    } finally {
        cleanupTempFiles(tempDir);
    }
}

/**
 * Test CSV format support
 */
async function testCsvFormat() {
    console.log('Testing CSV format support...');
    
    const { tempDir, csv } = createFormatTestFiles();
    
    try {
        const result = await runCLI([csv.file1, csv.file2]);
        
        if (result.code <= 1) {
            console.log('✓ CSV format support works');
            return true;
        } else {
            console.log('✗ CSV format support failed');
            console.log(`Exit code: ${result.code}`);
            console.log(`Stderr: ${result.stderr}`);
            return false;
        }
    } catch (error) {
        console.log(`✗ CSV format test error: ${error.message}`);
        return false;
    } finally {
        cleanupTempFiles(tempDir);
    }
}

/**
 * Test plain text format support
 */
async function testTextFormat() {
    console.log('Testing plain text format support...');
    
    const { tempDir, txt } = createFormatTestFiles();
    
    try {
        const result = await runCLI([txt.file1, txt.file2]);
        
        if (result.code <= 1) {
            console.log('✓ Plain text format support works');
            return true;
        } else {
            console.log('✗ Plain text format support failed');
            console.log(`Exit code: ${result.code}`);
            console.log(`Stderr: ${result.stderr}`);
            return false;
        }
    } catch (error) {
        console.log(`✗ Plain text format test error: ${error.message}`);
        return false;
    } finally {
        cleanupTempFiles(tempDir);
    }
}

/**
 * Test mixed format comparison (should handle gracefully)
 */
async function testMixedFormats() {
    console.log('Testing mixed format handling...');
    
    const { tempDir, json, txt } = createFormatTestFiles();
    
    try {
        const result = await runCLI([json.file1, txt.file1]);
        
        // Should either work or fail gracefully
        if (result.code <= 2) {
            console.log('✓ Mixed format handling works');
            return true;
        } else {
            console.log('✗ Mixed format handling failed');
            return false;
        }
    } catch (error) {
        console.log(`✗ Mixed format test error: ${error.message}`);
        return false;
    } finally {
        cleanupTempFiles(tempDir);
    }
}

/**
 * Main test runner for format tests
 */
async function runTests() {
    console.log('Running file format support tests...');
    console.log('='.repeat(50));

    const tests = [
        testJsonFormat,
        testYamlFormat,
        testCsvFormat,
        testTextFormat,
        testMixedFormats
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

    console.log('='.repeat(50));
    console.log(`Format Tests: ${passed}/${total} passed`);

    return { passed, total };
}

module.exports = { runTests };