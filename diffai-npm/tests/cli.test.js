/**
 * CLI basic functionality tests for diffai npm package
 */

const { spawn } = require('child_process');
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
 * Test CLI version command
 */
async function testVersion() {
    console.log('Testing CLI version...');
    try {
        const result = await runCLI(['--version']);
        if (result.code === 0 && result.stdout.includes('diffai')) {
            console.log('✓ Version command works');
            return true;
        } else {
            console.log('✗ Version command failed');
            return false;
        }
    } catch (error) {
        console.log(`✗ Version command error: ${error.message}`);
        return false;
    }
}

/**
 * Test CLI help command
 */
async function testHelp() {
    console.log('Testing CLI help...');
    try {
        const result = await runCLI(['--help']);
        if (result.code === 0 && result.stdout.includes('diffai')) {
            console.log('✓ Help command works');
            return true;
        } else {
            console.log('✗ Help command failed');
            return false;
        }
    } catch (error) {
        console.log(`✗ Help command error: ${error.message}`);
        return false;
    }
}

/**
 * Test CLI without arguments (should show help or usage)
 */
async function testNoArgs() {
    console.log('Testing CLI without arguments...');
    try {
        const result = await runCLI([]);
        // Should either show help (exit 0) or show usage info (exit non-zero)
        if (result.stdout.includes('diffai') || result.stderr.includes('diffai')) {
            console.log('✓ No arguments handling works');
            return true;
        } else {
            console.log('✗ No arguments handling failed');
            return false;
        }
    } catch (error) {
        console.log(`✗ No arguments test error: ${error.message}`);
        return false;
    }
}

/**
 * Main test runner for CLI tests
 */
async function runTests() {
    console.log('Running CLI functionality tests...');
    console.log('='.repeat(40));

    const tests = [
        testVersion,
        testHelp,
        testNoArgs
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

    console.log('='.repeat(40));
    console.log(`CLI Tests: ${passed}/${total} passed`);

    return { passed, total };
}

module.exports = { runTests };