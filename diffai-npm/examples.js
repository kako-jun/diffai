#!/usr/bin/env node

/**
 * diffai npm package examples
 * Demonstrates various usage patterns for AI/ML model comparison
 * Includes both CLI and JavaScript API usage examples
 */

const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

// Import the JavaScript API
const { diff, diffString, inspect, isDiffaiAvailable, getVersion, DiffaiError } = require('./lib.js');

// Helper function to run diffai commands (CLI)
function runDiffai(args) {
  return new Promise((resolve, reject) => {
    const child = spawn('diffai', args, { stdio: 'inherit' });
    
    child.on('close', (code) => {
      if (code === 0) {
        resolve();
      } else {
        resolve(); // Don't reject on diff found (code 1)
      }
    });
    
    child.on('error', (err) => {
      reject(err);
    });
  });
}

async function runExamples() {
  console.log('=== diffai npm package examples ===\n');
  
  // Check if diffai is available using JavaScript API
  try {
    console.log('1. Checking diffai installation using JavaScript API...');
    const available = await isDiffaiAvailable();
    if (available) {
      const version = await getVersion();
      console.log(`✓ diffai is installed and working: ${version}\n`);
    } else {
      throw new Error('diffai not available');
    }
  } catch (error) {
    console.error('✗ diffai is not available. Please install first: npm install -g diffai');
    process.exit(1);
  }
  
  // Basic help
  console.log('2. Basic help and features:');
  await runDiffai(['--help']);
  console.log('');
  
  // Create sample files for demonstration
  const sampleDir = path.join(__dirname, 'sample_data');
  if (!fs.existsSync(sampleDir)) {
    fs.mkdirSync(sampleDir);
  }
  
  // Sample JSON configs
  const config1 = {
    model: {
      name: "bert-base",
      layers: 12,
      hidden_size: 768,
      attention_heads: 12,
      parameters: 110000000
    },
    training: {
      learning_rate: 2e-5,
      batch_size: 32,
      epochs: 3,
      optimizer: "AdamW"
    }
  };
  
  const config2 = {
    model: {
      name: "bert-large",
      layers: 24,
      hidden_size: 1024,
      attention_heads: 16,
      parameters: 340000000
    },
    training: {
      learning_rate: 1e-5,
      batch_size: 16,
      epochs: 5,
      optimizer: "AdamW"
    }
  };
  
  fs.writeFileSync(path.join(sampleDir, 'bert_base_config.json'), JSON.stringify(config1, null, 2));
  fs.writeFileSync(path.join(sampleDir, 'bert_large_config.json'), JSON.stringify(config2, null, 2));
  
  console.log('3. Comparing model configurations:');
  await runDiffai([
    path.join(sampleDir, 'bert_base_config.json'),
    path.join(sampleDir, 'bert_large_config.json')
  ]);
  console.log('');
  
  console.log('4. JSON output for MLOps integration:');
  await runDiffai([
    path.join(sampleDir, 'bert_base_config.json'),
    path.join(sampleDir, 'bert_large_config.json'),
    '--output', 'json'
  ]);
  console.log('');
  
  console.log('5. Filtering specific configuration sections:');
  await runDiffai([
    path.join(sampleDir, 'bert_base_config.json'),
    path.join(sampleDir, 'bert_large_config.json'),
    '--path', 'training'
  ]);
  console.log('');
  
  // Sample YAML for ML pipeline
  const pipeline1 = `
name: training_pipeline_v1
steps:
  - name: data_preprocessing
    image: python:3.9
    script: preprocess.py
  - name: model_training
    image: pytorch/pytorch:latest
    script: train.py
    resources:
      gpu: 1
      memory: 8Gi
  - name: model_evaluation
    image: python:3.9
    script: evaluate.py
parameters:
  learning_rate: 0.001
  batch_size: 64
  epochs: 10
`;
  
  const pipeline2 = `
name: training_pipeline_v2
steps:
  - name: data_preprocessing
    image: python:3.9
    script: preprocess.py
  - name: model_training
    image: pytorch/pytorch:1.12
    script: train.py
    resources:
      gpu: 2
      memory: 16Gi
  - name: model_evaluation
    image: python:3.9
    script: evaluate.py
  - name: model_deployment
    image: tensorflow/serving:latest
    script: deploy.py
parameters:
  learning_rate: 0.0005
  batch_size: 128
  epochs: 20
`;
  
  fs.writeFileSync(path.join(sampleDir, 'pipeline_v1.yaml'), pipeline1);
  fs.writeFileSync(path.join(sampleDir, 'pipeline_v2.yaml'), pipeline2);
  
  console.log('6. Comparing ML pipeline configurations:');
  await runDiffai([
    path.join(sampleDir, 'pipeline_v1.yaml'),
    path.join(sampleDir, 'pipeline_v2.yaml')
  ]);
  console.log('');
  
  console.log('7. Unified diff output:');
  await runDiffai([
    path.join(sampleDir, 'pipeline_v1.yaml'),
    path.join(sampleDir, 'pipeline_v2.yaml'),
    '--output', 'unified'
  ]);
  console.log('');
  
  // JavaScript API Examples
  console.log('8. JavaScript API Examples:');
  console.log('---');
  
  try {
    // Basic comparison using JavaScript API
    console.log('8.1 Basic comparison using JavaScript API:');
    const basicResult = await diff(
      path.join(sampleDir, 'bert_base_config.json'),
      path.join(sampleDir, 'bert_large_config.json'),
      { output: 'json' }
    );
    console.log('Number of differences found:', basicResult.length);
    console.log('First difference:', basicResult[0]);
    console.log('');
    
    // String comparison
    console.log('8.2 String comparison using JavaScript API:');
    const json1 = JSON.stringify(config1, null, 2);
    const json2 = JSON.stringify(config2, null, 2);
    const stringResult = await diffString(json1, json2, 'json', { output: 'json' });
    console.log('String comparison result:', stringResult.length, 'differences');
    console.log('');
    
    // Advanced options
    console.log('8.3 Advanced comparison with ML analysis options:');
    const advancedResult = await diff(
      path.join(sampleDir, 'bert_base_config.json'),
      path.join(sampleDir, 'bert_large_config.json'),
      {
        output: 'json',
        stats: true,
        quiet: false
      }
    );
    console.log('Advanced analysis completed with', advancedResult.length, 'results');
    console.log('');
    
    // Error handling example
    console.log('8.4 Error handling example:');
    try {
      await diff('nonexistent1.json', 'nonexistent2.json');
    } catch (error) {
      if (error instanceof DiffaiError) {
        console.log('Caught DiffaiError:', error.message);
        console.log('Exit code:', error.exitCode);
      }
    }
    console.log('');
    
  } catch (error) {
    console.error('JavaScript API example failed:', error.message);
  }
  
  // Clean up sample files
  fs.rmSync(sampleDir, { recursive: true, force: true });
  
  console.log('Examples completed successfully!');
  console.log('\nJavaScript API Usage:');
  console.log('const { diff, diffString, inspect } = require("diffai");');
  console.log('const result = await diff("model1.safetensors", "model2.safetensors", { output: "json", stats: true });');
  console.log('\nFor more examples and ML model comparison features, see:');
  console.log('- https://github.com/kako-jun/diffai/blob/main/README.md');
  console.log('- https://github.com/kako-jun/diffai/blob/main/docs/');
}

if (require.main === module) {
  runExamples().catch(console.error);
}

module.exports = { runExamples };