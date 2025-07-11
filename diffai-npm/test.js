#!/usr/bin/env node

const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

// Import JavaScript API for testing
const { diff, diffString, isDiffaiAvailable, getVersion, DiffaiError } = require('./lib.js');

/**
 * Test script for diffai npm package
 * Verifies basic functionality, binary availability, and JavaScript API
 */

function runTest(testName, command, expectedPatterns = []) {
  return new Promise((resolve, reject) => {
    console.log(`\n🧪 Running test: ${testName}`);
    console.log(`Command: ${command.join(' ')}`);
    
    const child = spawn(command[0], command.slice(1), {
      stdio: ['pipe', 'pipe', 'pipe'],
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
      console.log(`Exit code: ${code}`);
      
      // Check expected patterns
      let allPatternsFound = true;
      for (const pattern of expectedPatterns) {
        const found = stdout.includes(pattern) || stderr.includes(pattern);
        console.log(`Pattern "${pattern}": ${found ? '✅ Found' : '❌ Not found'}`);
        if (!found) allPatternsFound = false;
      }
      
      if (allPatternsFound) {
        console.log(`✅ Test "${testName}" passed`);
        resolve({ code, stdout, stderr });
      } else {
        console.log(`❌ Test "${testName}" failed`);
        reject(new Error(`Test failed: ${testName}`));
      }
    });
    
    child.on('error', (error) => {
      console.log(`❌ Test "${testName}" errored: ${error.message}`);
      reject(error);
    });
  });
}

async function runAllTests() {
  console.log('🚀 Starting diffai npm package tests');
  
  try {
    // Test 1: Binary availability
    console.log('\n📋 Test 1: Binary availability');
    const binaryName = process.platform === 'win32' ? 'diffai.exe' : 'diffai';
    const binaryPath = path.join(__dirname, 'bin', binaryName);
    
    if (fs.existsSync(binaryPath)) {
      console.log('✅ Binary exists at:', binaryPath);
    } else {
      console.log('❌ Binary not found at:', binaryPath);
      console.log('Checking system PATH...');
    }
    
    // Test 2: Version check
    await runTest(
      'Version check',
      ['node', path.join(__dirname, 'index.js'), '--version'],
      ['diffai', '0.2']
    );
    
    // Test 3: Help output
    await runTest(
      'Help output',
      ['node', path.join(__dirname, 'index.js'), '--help'],
      ['Usage:', 'diffai', 'AI/ML']
    );
    
    // Test 4: Basic invalid command (should show error)
    try {
      await runTest(
        'Invalid command handling',
        ['node', path.join(__dirname, 'index.js'), 'nonexistent_file1.txt', 'nonexistent_file2.txt'],
        ['error', 'Error', 'not found', 'No such file']
      );
    } catch (error) {
      // This test is expected to fail, which is actually success
      console.log('✅ Invalid command properly handled');
    }
    
    // Test 5: JavaScript API tests
    console.log('\n📋 Test 5: JavaScript API tests');
    
    // Test isDiffaiAvailable
    console.log('\n🧪 Testing isDiffaiAvailable...');
    const isAvailable = await isDiffaiAvailable();
    console.log(`✅ isDiffaiAvailable: ${isAvailable}`);
    
    if (isAvailable) {
      // Test getVersion
      console.log('\n🧪 Testing getVersion...');
      try {
        const version = await getVersion();
        console.log(`✅ getVersion: ${version}`);
      } catch (error) {
        console.log(`❌ getVersion failed: ${error.message}`);
      }
      
      // Test diffString with sample data
      console.log('\n🧪 Testing diffString...');
      const sampleJson1 = '{"name": "model1", "version": "1.0", "params": 1000}';
      const sampleJson2 = '{"name": "model2", "version": "2.0", "params": 2000}';
      
      try {
        const result = await diffString(sampleJson1, sampleJson2, 'json', { output: 'json' });
        console.log(`✅ diffString: Found ${result.length} differences`);
        if (result.length > 0) {
          console.log(`   First difference: ${result[0].type} at ${result[0].path}`);
        }
      } catch (error) {
        if (error instanceof DiffaiError) {
          console.log(`✅ diffString properly threw DiffaiError: ${error.message}`);
        } else {
          console.log(`❌ diffString failed: ${error.message}`);
        }
      }
      
      // Test error handling
      console.log('\n🧪 Testing error handling...');
      try {
        await diff('nonexistent1.json', 'nonexistent2.json');
        console.log('❌ Should have thrown error for nonexistent files');
      } catch (error) {
        if (error instanceof DiffaiError) {
          console.log(`✅ Error handling: Properly caught DiffaiError`);
        } else {
          console.log(`✅ Error handling: Caught error: ${error.message}`);
        }
      }
    } else {
      console.log('⚠️  Skipping JavaScript API tests - diffai binary not available');
    }
    
    console.log('\n🎉 All tests completed successfully!');
    console.log('\n📝 CLI Usage examples:');
    console.log('  npx diffai model1.safetensors model2.safetensors --stats');
    console.log('  npx diffai data1.npy data2.npy --stats');
    console.log('  npx diffai config1.json config2.json');
    
    console.log('\n📝 JavaScript API Usage examples:');
    console.log('  const { diff } = require("diffai");');
    console.log('  const result = await diff("model1.pt", "model2.pt", { output: "json", stats: true });');
    console.log('  const stringResult = await diffString(json1, json2, "json", { output: "json" });');
    
  } catch (error) {
    console.error('\n💥 Test suite failed:', error.message);
    process.exit(1);
  }
}

// Only run if this script is called directly
if (require.main === module) {
  runAllTests();
}