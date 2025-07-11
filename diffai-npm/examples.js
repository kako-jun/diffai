#!/usr/bin/env node

/**
 * diffai npm package examples
 * Demonstrates various usage patterns for AI/ML model comparison
 */

const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

// Helper function to run diffai commands
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
  
  // Check if diffai is available
  try {
    console.log('1. Checking diffai installation...');
    await runDiffai(['--version']);
    console.log('✓ diffai is installed and working\n');
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
  
  // Clean up sample files
  fs.rmSync(sampleDir, { recursive: true, force: true });
  
  console.log('Examples completed successfully!');
  console.log('\nFor more examples and ML model comparison features, see:');
  console.log('- https://github.com/kako-jun/diffai/blob/main/README.md');
  console.log('- https://github.com/kako-jun/diffai/blob/main/docs/');
}

if (require.main === module) {
  runExamples().catch(console.error);
}

module.exports = { runExamples };