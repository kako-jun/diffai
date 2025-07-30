#!/usr/bin/env node
/**
 * Simple test for diffai-js Node.js integration
 */

try {
    const diffai = require('./index.js');
    console.log('✅ Successfully imported diffai-js');
    console.log(`Available functions:`, Object.keys(diffai));
    
    // Test basic diff functionality
    const oldData = { name: "test", value: 123 };
    const newData = { name: "test", value: 456 };
    
    const results = diffai.diff(oldData, newData);
    console.log(`✅ Diff results:`, results);
    
    // Test AI/ML specific functionality
    const oldModel = {
        learning_rate: 0.001,
        accuracy: 0.85,
        model_type: "pytorch"
    };
    
    const newModel = {
        learning_rate: 0.01,
        accuracy: 0.92,
        model_type: "pytorch"
    };
    
    const mlResults = diffai.diff(oldModel, newModel, {
        learningRateTracking: true,
        accuracyTracking: true
    });
    console.log(`✅ ML analysis results:`, mlResults);
    
    console.log('🎉 diffai-js Node.js integration working correctly!');
    
} catch (error) {
    console.error(`❌ Test failed:`, error.message);
    process.exit(1);
}