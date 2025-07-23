#!/usr/bin/env node
/**
 * Unified test runner for diffai npm package
 * Runs all tests in the tests/ directory
 */

const fs = require('fs');
const path = require('path');

const TESTS_DIR = path.join(__dirname, 'tests');

async function runAllTests() {
    console.log('🧪 Running all diffai npm package tests...');
    console.log('='.repeat(60));

    // Check if tests directory exists
    if (!fs.existsSync(TESTS_DIR)) {
        console.log('❌ Tests directory not found:', TESTS_DIR);
        process.exit(1);
    }

    // Find all test files
    const testFiles = fs.readdirSync(TESTS_DIR)
        .filter(file => file.endsWith('.test.js'))
        .map(file => path.join(TESTS_DIR, file));

    if (testFiles.length === 0) {
        console.log('⚠️  No test files found in tests/ directory');
        process.exit(1);
    }

    console.log(`Found ${testFiles.length} test file(s):`);
    testFiles.forEach(file => console.log(`  - ${path.basename(file)}`));
    console.log('');

    let totalPassed = 0;
    let totalTests = 0;
    let allPassed = true;

    // Run each test file
    for (const testFile of testFiles) {
        console.log(`\n📋 Running ${path.basename(testFile)}...`);
        console.log('-'.repeat(40));

        try {
            // Import and run the test
            const testModule = require(testFile);
            if (typeof testModule.runTests === 'function') {
                const result = await testModule.runTests();
                // If runTests returns a result object, use it
                if (typeof result === 'object' && result.passed !== undefined) {
                    totalPassed += result.passed;
                    totalTests += result.total;
                    if (result.passed < result.total) {
                        allPassed = false;
                    }
                }
            } else {
                console.log('⚠️  Test file does not export runTests function');
                allPassed = false;
            }
        } catch (error) {
            console.log(`❌ Error running test file: ${error.message}`);
            allPassed = false;
        }
    }

    // Final summary
    console.log('\n' + '='.repeat(60));
    console.log('📊 FINAL RESULTS');
    console.log('='.repeat(60));
    
    if (totalTests > 0) {
        console.log(`Total tests: ${totalTests}`);
        console.log(`Passed: ${totalPassed}`);
        console.log(`Failed: ${totalTests - totalPassed}`);
    }

    if (allPassed) {
        console.log('🎉 All test suites passed!');
        process.exit(0);
    } else {
        console.log('❌ Some test suites failed');
        process.exit(1);
    }
}

// Handle uncaught errors
process.on('uncaughtException', (error) => {
    console.error('❌ Uncaught exception:', error);
    process.exit(1);
});

process.on('unhandledRejection', (reason, promise) => {
    console.error('❌ Unhandled rejection at:', promise, 'reason:', reason);
    process.exit(1);
});

// Run tests
runAllTests().catch(error => {
    console.error('❌ Test runner error:', error);
    process.exit(1);
});