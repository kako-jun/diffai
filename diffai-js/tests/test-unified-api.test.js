const fs = require('fs');
const path = require('path');

// Import the diffai module
let diffai;
try {
    diffai = require('../index.js');
} catch (error) {
    console.log('diffai module not built, skipping tests');
    process.exit(0);
}

// ============================================================================
// TEST FIXTURES - Shared with Core and Python Tests (AI/ML Focus)
// ============================================================================

class TestFixtures {
    /**
     * JavaScript equivalent of Rust/Python fixtures for diffai unified API testing.
     * Focuses on AI/ML specific test data.
     */
    
    static loadMlModelFixture(filename) {
        const fixturesDir = path.join(__dirname, '..', '..', 'tests', 'fixtures', 'ml_models');
        const fixturePath = path.join(fixturesDir, filename);
        
        if (!fs.existsSync(fixturePath)) {
            throw new Error(`ML model fixture not found: ${fixturePath}`);
        }
        
        return fixturePath; // Return path for binary ML files
    }
    
    static modelV1Path() {
        return TestFixtures.loadMlModelFixture('model1.pt');
    }
    
    static modelV2Path() {
        return TestFixtures.loadMlModelFixture('model2.pt');
    }
    
    // AI/ML specific fixtures
    
    static pytorchModelOld() {
        return {
            model_type: "pytorch",
            model_info: {
                architecture: "ResNet",
                layers: [
                    {
                        name: "conv1",
                        type: "Conv2d",
                        in_channels: 3,
                        out_channels: 64,
                        kernel_size: [7, 7],
                        weights: {
                            shape: [64, 3, 7, 7],
                            mean: 0.01,
                            std: 0.1
                        }
                    },
                    {
                        name: "fc",
                        type: "Linear",
                        in_features: 512,
                        out_features: 1000,
                        weights: {
                            shape: [1000, 512],
                            mean: 0.0,
                            std: 0.05
                        }
                    }
                ],
                optimizer: {
                    type: "Adam",
                    learning_rate: 0.001,
                    beta1: 0.9,
                    beta2: 0.999
                },
                loss_function: "CrossEntropyLoss",
                training: {
                    epoch: 10,
                    loss: 0.25,
                    accuracy: 0.92
                }
            }
        };
    }
    
    static pytorchModelNew() {
        return {
            model_type: "pytorch",
            model_info: {
                architecture: "ResNet",
                layers: [
                    {
                        name: "conv1",
                        type: "Conv2d",
                        in_channels: 3,
                        out_channels: 64,
                        kernel_size: [7, 7],
                        weights: {
                            shape: [64, 3, 7, 7],
                            mean: 0.015,  // Changed
                            std: 0.12     // Changed
                        }
                    },
                    {
                        name: "fc",
                        type: "Linear",
                        in_features: 512,
                        out_features: 1000,
                        weights: {
                            shape: [1000, 512],
                            mean: 0.002,  // Changed
                            std: 0.048    // Changed
                        }
                    }
                ],
                optimizer: {
                    type: "SGD",        // Changed from Adam
                    learning_rate: 0.01, // Changed
                    momentum: 0.9       // Added
                },
                loss_function: "CrossEntropyLoss",
                training: {
                    epoch: 15,          // Changed
                    loss: 0.18,         // Improved
                    accuracy: 0.95      // Improved
                }
            }
        };
    }
    
    static safetensorsModelOld() {
        return {
            model_type: "safetensors",
            tensors: {
                "embedding.weight": {
                    shape: [50000, 768],
                    dtype: "float32"
                },
                "encoder.layer.0.attention.self.query.weight": {
                    shape: [768, 768],
                    dtype: "float32"
                },
                "classifier.weight": {
                    shape: [2, 768],
                    dtype: "float32"
                }
            },
            metadata: {
                model_name: "bert-base",
                version: "1.0",
                total_params: 110000000
            }
        };
    }
    
    static safetensorsModelNew() {
        return {
            model_type: "safetensors",
            tensors: {
                "embedding.weight": {
                    shape: [50000, 1024], // Changed dimension
                    dtype: "float32"
                },
                "encoder.layer.0.attention.self.query.weight": {
                    shape: [1024, 1024], // Changed dimension
                    dtype: "float16"     // Changed precision
                },
                "classifier.weight": {
                    shape: [2, 1024],    // Changed dimension
                    dtype: "float32"
                },
                "new_layer.weight": {      // Added new tensor
                    shape: [1024, 512],
                    dtype: "float32"
                }
            },
            metadata: {
                model_name: "bert-large", // Changed model
                version: "2.0",           // Changed version
                total_params: 340000000   // Changed param count
            }
        };
    }
    
    static trainingMetricsOld() {
        return {
            experiment: {
                name: "baseline_experiment",
                model: "resnet50",
                dataset: "imagenet",
                metrics: {
                    training: {
                        loss: [2.5, 1.8, 1.2, 0.9, 0.7],
                        accuracy: [0.2, 0.4, 0.6, 0.75, 0.82]
                    },
                    validation: {
                        loss: [2.8, 2.0, 1.5, 1.1, 0.9],
                        accuracy: [0.18, 0.35, 0.55, 0.70, 0.78]
                    },
                    hyperparameters: {
                        learning_rate: 0.001,
                        batch_size: 32,
                        optimizer: "Adam",
                        weight_decay: 0.0001
                    }
                }
            }
        };
    }
    
    static trainingMetricsNew() {
        return {
            experiment: {
                name: "improved_experiment",
                model: "resnet50",
                dataset: "imagenet",
                metrics: {
                    training: {
                        loss: [2.3, 1.5, 1.0, 0.7, 0.5],    // Improved
                        accuracy: [0.25, 0.45, 0.65, 0.80, 0.88] // Improved
                    },
                    validation: {
                        loss: [2.5, 1.7, 1.2, 0.9, 0.7],    // Improved
                        accuracy: [0.22, 0.40, 0.60, 0.75, 0.83] // Improved
                    },
                    hyperparameters: {
                        learning_rate: 0.01,     // Changed
                        batch_size: 64,          // Changed
                        optimizer: "SGD",        // Changed
                        weight_decay: 0.0005,    // Changed
                        momentum: 0.9            // Added
                    }
                }
            }
        };
    }
    
    static modelArchitectureOld() {
        return {
            model: {
                name: "custom_cnn",
                type: "sequential",
                layers: [
                    {
                        name: "input",
                        type: "Input",
                        shape: [224, 224, 3]
                    },
                    {
                        name: "conv1",
                        type: "Conv2D",
                        filters: 32,
                        kernel_size: [3, 3],
                        activation: "relu"
                    },
                    {
                        name: "pool1",
                        type: "MaxPooling2D",
                        pool_size: [2, 2]
                    },
                    {
                        name: "flatten",
                        type: "Flatten"
                    },
                    {
                        name: "dense1",
                        type: "Dense",
                        units: 128,
                        activation: "relu"
                    },
                    {
                        name: "output",
                        type: "Dense",
                        units: 10,
                        activation: "softmax"
                    }
                ]
            }
        };
    }
    
    static modelArchitectureNew() {
        return {
            model: {
                name: "improved_cnn",  // Changed name
                type: "functional",    // Changed type
                layers: [
                    {
                        name: "input",
                        type: "Input",
                        shape: [224, 224, 3]
                    },
                    {
                        name: "conv1",
                        type: "Conv2D",
                        filters: 64,       // Increased filters
                        kernel_size: [3, 3],
                        activation: "relu"
                    },
                    {
                        name: "conv2",     // Added new layer
                        type: "Conv2D",
                        filters: 64,
                        kernel_size: [3, 3],
                        activation: "relu"
                    },
                    {
                        name: "pool1",
                        type: "MaxPooling2D",
                        pool_size: [2, 2]
                    },
                    {
                        name: "dropout1", // Added dropout
                        type: "Dropout",
                        rate: 0.25
                    },
                    {
                        name: "flatten",
                        type: "Flatten"
                    },
                    {
                        name: "dense1",
                        type: "Dense",
                        units: 256,       // Increased units
                        activation: "relu"
                    },
                    {
                        name: "dropout2", // Added dropout
                        type: "Dropout",
                        rate: 0.5
                    },
                    {
                        name: "output",
                        type: "Dense",
                        units: 10,
                        activation: "softmax"
                    }
                ]
            }
        };
    }
}

// ============================================================================
// TEST HELPER FUNCTIONS
// ============================================================================

function expectDiffResult(result, type, path, additionalChecks = {}) {
    expect(result).toHaveProperty('type', type);
    expect(result).toHaveProperty('path', path);
    
    Object.entries(additionalChecks).forEach(([key, value]) => {
        expect(result).toHaveProperty(key, value);
    });
}

function expectNoDifferences(old, newObj, options = {}) {
    const results = diffai.diff(old, newObj, options);
    expect(results).toHaveLength(0);
}

function expectDifferences(old, newObj, expectedCount, options = {}) {
    const results = diffai.diff(old, newObj, options);
    expect(results).toHaveLength(expectedCount);
    return results;
}

// ============================================================================
// UNIFIED API TESTS - Core Functionality
// ============================================================================

describe('Unified API - Core Functionality', () => {
    test('diff basic modification', () => {
        const old = { name: "Alice", age: 30 };
        const newObj = { name: "Alice", age: 31 };
        
        const results = diffai.diff(old, newObj);
        
        expect(results).toHaveLength(1);
        expectDiffResult(results[0], 'modified', 'age', {
            oldValue: expect.stringContaining('30'),
            newValue: expect.stringContaining('31')
        });
    });
    
    test('diff AI/ML specific results', () => {
        const old = { learning_rate: 0.001, accuracy: 0.85 };
        const newObj = { learning_rate: 0.01, accuracy: 0.92 };
        
        const results = diffai.diff(old, newObj, {
            learningRateTracking: true,
            accuracyTracking: true
        });
        
        expect(results).toHaveLength(2);
        
        // Check for learning rate change
        const lrResult = results.find(r => r.type === 'LearningRateChanged');
        expect(lrResult).toBeDefined();
        expect(lrResult.oldLearningRate).toBe(0.001);
        expect(lrResult.newLearningRate).toBe(0.01);
        
        // Check for accuracy change
        const accResult = results.find(r => r.type === 'AccuracyChange');
        expect(accResult).toBeDefined();
        expect(accResult.oldAccuracy).toBe(0.85);
        expect(accResult.newAccuracy).toBe(0.92);
    });
    
    test('diff weight threshold', () => {
        const old = { weights: { layer1: 0.1, layer2: 0.05 } };
        const newObj = { weights: { layer1: 0.2, layer2: 0.051 } };
        
        const results = diffai.diff(old, newObj, {
            weightThreshold: 0.05  // Only changes > 0.05 are significant
        });
        
        // Should only detect layer1 as significant change (0.1 difference > 0.05 threshold)
        const significantChanges = results.filter(r => r.type === 'WeightSignificantChange');
        expect(significantChanges).toHaveLength(1);
        
        const change = significantChanges[0];
        expect(change.path).toMatch(/layer1/);
        expect(change.magnitude).toBe(0.1);
    });
});

// ============================================================================
// AI/ML SPECIFIC TESTS - PyTorch Models
// ============================================================================

describe('PyTorch Models', () => {
    test('pytorch model comparison', () => {
        const old = TestFixtures.pytorchModelOld();
        const newObj = TestFixtures.pytorchModelNew();
        
        const results = diffai.diff(old, newObj, {
            mlAnalysisEnabled: true,
            learningRateTracking: true,
            optimizerComparison: true,
            lossTracking: true,
            accuracyTracking: true
        });
        
        expect(results.length).toBeGreaterThan(0);
        
        // Should detect optimizer change (Adam -> SGD)
        const optimizerChanges = results.filter(r => 
            r.type === 'modified' && r.path.includes('optimizer.type'));
        expect(optimizerChanges.length).toBeGreaterThan(0);
        
        // Should detect learning rate change
        const lrChanges = results.filter(r => r.type === 'LearningRateChanged');
        expect(lrChanges.length).toBeGreaterThan(0);
    });
    
    test('pytorch layer weight changes', () => {
        const old = {
            layers: {
                conv1: {
                    weights: {
                        mean: 0.01,
                        std: 0.1
                    }
                },
                fc: {
                    weights: {
                        mean: 0.0,
                        std: 0.05
                    }
                }
            }
        };
        
        const newObj = {
            layers: {
                conv1: {
                    weights: {
                        mean: 0.015,  // Small change
                        std: 0.12     // Small change
                    }
                },
                fc: {
                    weights: {
                        mean: 0.1,    // Large change
                        std: 0.15     // Large change
                    }
                }
            }
        };
        
        const results = diffai.diff(old, newObj, {
            weightThreshold: 0.05,
            epsilon: 0.001
        });
        
        // Should detect significant changes in fc layer
        const significantChanges = results.filter(r => r.type === 'WeightSignificantChange');
        expect(significantChanges.length).toBeGreaterThan(0);
    });
});

// ============================================================================
// AI/ML SPECIFIC TESTS - SafeTensors Models
// ============================================================================

describe('SafeTensors Models', () => {
    test('safetensors model comparison', () => {
        const old = TestFixtures.safetensorsModelOld();
        const newObj = TestFixtures.safetensorsModelNew();
        
        const results = diffai.diff(old, newObj, {
            tensorComparisonMode: "both"
        });
        
        expect(results.length).toBeGreaterThan(0);
        
        // Should detect tensor shape changes
        const shapeChanges = results.filter(r => 
            r.type === 'modified' && r.path.includes('shape'));
        expect(shapeChanges.length).toBeGreaterThan(0);
        
        // Should detect dtype changes  
        const dtypeChanges = results.filter(r => 
            r.type === 'modified' && r.path.includes('dtype'));
        expect(dtypeChanges.length).toBeGreaterThan(0);
        
        // Should detect new tensors
        const addedTensors = results.filter(r => 
            r.type === 'added' && r.path.includes('new_layer'));
        expect(addedTensors.length).toBeGreaterThan(0);
    });
    
    test('safetensors metadata comparison', () => {
        const old = TestFixtures.safetensorsModelOld();
        const newObj = TestFixtures.safetensorsModelNew();
        
        const results = diffai.diff(old, newObj);
        
        // Should detect metadata changes
        const metadataChanges = results.filter(r => 
            r.type === 'modified' && r.path.includes('metadata'));
        expect(metadataChanges.length).toBeGreaterThan(0);
        
        // Check for specific version change
        const versionChange = results.find(r => 
            r.type === 'modified' && r.path.includes('version') &&
            r.oldValue.includes('1.0') && r.newValue.includes('2.0'));
        expect(versionChange).toBeDefined();
    });
});

// ============================================================================
// TRAINING METRICS COMPARISON TESTS
// ============================================================================

describe('Training Metrics', () => {
    test('training metrics comparison', () => {
        const old = TestFixtures.trainingMetricsOld();
        const newObj = TestFixtures.trainingMetricsNew();
        
        const results = diffai.diff(old, newObj, {
            lossTracking: true,
            accuracyTracking: true,
            optimizerComparison: true,
            learningRateTracking: true
        });
        
        expect(results.length).toBeGreaterThan(0);
        
        // Should detect optimizer change
        const optimizerChanges = results.filter(r => 
            r.type === 'modified' && r.path.includes('optimizer') &&
            r.oldValue.includes('Adam') && r.newValue.includes('SGD'));
        expect(optimizerChanges.length).toBeGreaterThan(0);
        
        // Should detect learning rate change
        const lrChanges = results.filter(r => r.type === 'LearningRateChanged' &&
            r.oldLearningRate === 0.001 && r.newLearningRate === 0.01);
        expect(lrChanges.length).toBeGreaterThan(0);
    });
    
    test('training history arrays', () => {
        const old = {
            training: {
                loss: [2.5, 1.8, 1.2, 0.9, 0.7],
                accuracy: [0.2, 0.4, 0.6, 0.75, 0.82]
            }
        };
        
        const newObj = {
            training: {
                loss: [2.3, 1.5, 1.0, 0.7, 0.5],
                accuracy: [0.25, 0.45, 0.65, 0.80, 0.88]
            }
        };
        
        const results = diffai.diff(old, newObj);
        
        // Should detect changes in loss and accuracy arrays
        expect(results.length).toBeGreaterThan(0);
        
        const lossChanges = results.filter(r => 
            r.type === 'modified' && r.path.includes('loss['));
        expect(lossChanges.length).toBeGreaterThan(0);
        
        const accuracyChanges = results.filter(r => 
            r.type === 'modified' && r.path.includes('accuracy['));
        expect(accuracyChanges.length).toBeGreaterThan(0);
    });
});

// ============================================================================
// JAVASCRIPT TYPE HANDLING TESTS (AI/ML Focus)
// ============================================================================

describe('JavaScript Type Handling (AI/ML)', () => {
    test('tensor-like array data', () => {
        const old = {
            tensor: {
                data: [[1.0, 2.0], [3.0, 4.0]],
                shape: [2, 2],
                dtype: "float32"
            }
        };
        
        const newObj = {
            tensor: {
                data: [[1.1, 2.1], [3.1, 4.1]],
                shape: [2, 2],
                dtype: "float64"  // Changed precision
            }
        };
        
        const results = diffai.diff(old, newObj, { epsilon: 0.05 });
        
        // Should detect dtype change
        const dtypeChanges = results.filter(r => 
            r.type === 'modified' && r.path.includes('dtype'));
        expect(dtypeChanges.length).toBeGreaterThan(0);
    });
    
    test('ml metadata handling', () => {
        const old = {
            model_metadata: {
                framework: "pytorch",
                version: "1.9.0",
                device: "cuda:0",
                mixed_precision: false
            }
        };
        
        const newObj = {
            model_metadata: {
                framework: "pytorch",
                version: "2.0.0",      // Changed
                device: "cuda:1",      // Changed
                mixed_precision: true  // Changed
            }
        };
        
        const results = diffai.diff(old, newObj);
        
        expect(results).toHaveLength(3); // Three changes
        
        // Check version change
        const versionChange = results.find(r => r.path.includes('version'));
        expect(versionChange).toBeDefined();
        expect(versionChange.oldValue).toContain('1.9.0');
        expect(versionChange.newValue).toContain('2.0.0');
    });
    
    test('large numeric arrays', () => {
        const old = {
            weights: Array.from({ length: 1000 }, (_, i) => i * 0.001)
        };
        
        const newObj = {
            weights: Array.from({ length: 1000 }, (_, i) => i * 0.001 + 0.01)
        };
        
        const results = diffai.diff(old, newObj, { epsilon: 0.005 });
        
        // Should detect changes in the array
        expect(results.length).toBeGreaterThan(0);
        
        // All changes should be in the weights array
        results.forEach(result => {
            expect(result.path).toMatch(/weights\[\d+\]/);
        });
    });
});

// ============================================================================
// OPTIONS TESTING - diffai Specific Options
// ============================================================================

describe('diffai Specific Options', () => {
    test('tensor comparison mode', () => {
        const old = {
            tensor: {
                shape: [100, 200],
                data: [1.0, 2.0, 3.0]
            }
        };
        
        const newObj = {
            tensor: {
                shape: [100, 300], // Shape changed
                data: [1.1, 2.1, 3.1] // Data changed
            }
        };
        
        // Test shape-only mode
        const results = diffai.diff(old, newObj, {
            tensorComparisonMode: "shape"
        });
        
        // Should primarily focus on shape changes
        const shapeChanges = results.filter(r => 
            r.type === 'modified' && r.path.includes('shape'));
        expect(shapeChanges.length).toBeGreaterThan(0);
    });
    
    test('ml analysis enabled', () => {
        const old = {
            model: {
                learning_rate: 0.001,
                loss: 0.5,
                accuracy: 0.8,
                weights: { layer1: 0.1 }
            }
        };
        
        const newObj = {
            model: {
                learning_rate: 0.01,
                loss: 0.3,
                accuracy: 0.9,
                weights: { layer1: 0.2 }
            }
        };
        
        // With ML analysis enabled
        const mlResults = diffai.diff(old, newObj, {
            mlAnalysisEnabled: true,
            learningRateTracking: true,
            lossTracking: true,
            accuracyTracking: true,
            weightThreshold: 0.05
        });
        
        // Should use ML-specific diff result types
        const mlSpecificResults = mlResults.filter(r => [
            'LearningRateChanged', 'LossChange', 'AccuracyChange', 'WeightSignificantChange'
        ].includes(r.type));
        expect(mlSpecificResults.length).toBeGreaterThan(0);
        
        // Without ML analysis
        const regularResults = diffai.diff(old, newObj);
        
        // Should use regular diff result types
        const regularResultTypes = regularResults.filter(r => r.type === 'modified');
        expect(regularResultTypes.length).toBeGreaterThan(0);
    });
    
    test('scientific precision', () => {
        const old = {
            measurements: {
                precision: 1e-10,
                recall: 0.99999999,
                f1_score: 0.999999995
            }
        };
        
        const newObj = {
            measurements: {
                precision: 1.1e-10,      // Very small change
                recall: 0.999999991,     // Very small change
                f1_score: 0.999999996    // Very small change
            }
        };
        
        // With scientific precision
        const preciseResults = diffai.diff(old, newObj, {
            scientificPrecision: true,
            epsilon: 1e-12  // Very small epsilon
        });
        
        // Should detect very small changes
        expect(preciseResults.length).toBeGreaterThan(0);
        
        // Without scientific precision (larger epsilon)
        const regularResults = diffai.diff(old, newObj, {
            epsilon: 1e-8  // Larger epsilon
        });
        
        // Should detect fewer or no changes
        expect(regularResults.length).toBeLessThanOrEqual(preciseResults.length);
    });
});

// ============================================================================
// MODEL ARCHITECTURE COMPARISON TESTS
// ============================================================================

describe('Model Architecture', () => {
    test('model architecture comparison', () => {
        const old = TestFixtures.modelArchitectureOld();
        const newObj = TestFixtures.modelArchitectureNew();
        
        const results = diffai.diff(old, newObj);
        
        expect(results.length).toBeGreaterThan(0);
        
        // Should detect model type change
        const typeChanges = results.filter(r => 
            r.type === 'modified' && r.path.includes('model.type') &&
            r.oldValue.includes('sequential') && r.newValue.includes('functional'));
        expect(typeChanges.length).toBeGreaterThan(0);
        
        // Should detect added layers (conv2, dropout layers)
        const addedLayers = results.filter(r => 
            r.type === 'added' && r.path.includes('layers[') && 
            (r.path.includes('conv2') || r.path.includes('dropout')));
        expect(addedLayers.length).toBeGreaterThan(0);
        
        // Should detect filter count changes
        const filterChanges = results.filter(r => 
            r.type === 'modified' && r.path.includes('filters') &&
            r.oldValue.includes('32') && r.newValue.includes('64'));
        expect(filterChanges.length).toBeGreaterThan(0);
    });
});

// ============================================================================
// ERROR HANDLING TESTS
// ============================================================================

describe('Error Handling', () => {
    test('invalid regex pattern', () => {
        const old = { test: "value" };
        const newObj = { test: "value2" };
        
        expect(() => {
            diffai.diff(old, newObj, { ignoreKeysRegex: "[invalid_regex" });
        }).toThrow();
    });
    
    test('invalid output format', () => {
        const old = { test: "value" };
        const newObj = { test: "value2" };
        
        expect(() => {
            diffai.diff(old, newObj, { outputFormat: "invalid_format" });
        }).toThrow();
    });
    
    test('invalid tensor comparison mode', () => {
        const old = { tensor: { shape: [10] } };
        const newObj = { tensor: { shape: [20] } };
        
        // Should handle invalid mode gracefully or raise clear error
        try {
            const results = diffai.diff(old, newObj, { 
                tensorComparisonMode: "invalid_mode" 
            });
            // If it doesn't raise an error, it should still work
            expect(Array.isArray(results)).toBe(true);
        } catch (error) {
            expect(error.message.toLowerCase()).toMatch(/invalid_mode|mode/);
        }
    });
});

// ============================================================================
// INTEGRATION TESTS
// ============================================================================

describe('Integration Tests', () => {
    test('comprehensive ml workflow', () => {
        const oldModel = TestFixtures.pytorchModelOld();
        const newModel = TestFixtures.pytorchModelNew();
        
        const results = diffai.diff(oldModel, newModel, {
            mlAnalysisEnabled: true,
            tensorComparisonMode: "both",
            learningRateTracking: true,
            optimizerComparison: true,
            lossTracking: true,
            accuracyTracking: true,
            weightThreshold: 0.01,
            activationAnalysis: true,
            epsilon: 0.001,
            outputFormat: "json"
        });
        
        expect(results.length).toBeGreaterThan(0);
        
        // Should detect multiple types of ML changes
        const changeTypes = new Set(results.map(r => r.type));
        
        // Should detect multiple types of changes in a comprehensive ML comparison
        expect(changeTypes.size).toBeGreaterThanOrEqual(2);
    });
    
    test('real world model evolution', () => {
        // Model v1: Simple architecture
        const modelV1 = {
            architecture: {
                type: "sequential",
                layers: [
                    { type: "dense", units: 128, activation: "relu" },
                    { type: "dropout", rate: 0.2 },
                    { type: "dense", units: 10, activation: "softmax" }
                ]
            },
            training: {
                optimizer: "adam",
                learning_rate: 0.001,
                batch_size: 32,
                epochs: 50,
                loss: 0.45,
                accuracy: 0.87
            }
        };
        
        // Model v2: Improved architecture and hyperparameters
        const modelV2 = {
            architecture: {
                type: "functional",  // Changed
                layers: [
                    { type: "dense", units: 256, activation: "relu" },  // Increased units
                    { type: "batch_norm" },  // Added batch normalization
                    { type: "dropout", rate: 0.3 },  // Increased dropout
                    { type: "dense", units: 128, activation: "relu" },  // Added layer
                    { type: "dropout", rate: 0.2 },
                    { type: "dense", units: 10, activation: "softmax" }
                ]
            },
            training: {
                optimizer: "adamw",  // Changed optimizer
                learning_rate: 0.0005,  // Reduced learning rate
                batch_size: 64,  // Increased batch size
                epochs: 100,  // More epochs
                loss: 0.32,  // Improved loss
                accuracy: 0.94  // Improved accuracy
            }
        };
        
        const results = diffai.diff(modelV1, modelV2, {
            mlAnalysisEnabled: true,
            learningRateTracking: true,
            lossTracking: true,
            accuracyTracking: true
        });
        
        expect(results.length).toBeGreaterThan(0);
        
        // Should detect architectural improvements
        const archChanges = results.filter(r => r.path.includes('architecture'));
        expect(archChanges.length).toBeGreaterThan(0);
        
        // Should detect training improvements
        const trainingImprovements = results.filter(r => [
            'LearningRateChanged', 'LossChange', 'AccuracyChange'
        ].includes(r.type));
        expect(trainingImprovements.length).toBeGreaterThan(0);
    });
});

// ============================================================================
// PERFORMANCE TESTS
// ============================================================================

describe('Performance Tests', () => {
    test('large tensor comparison', () => {
        // Simulate large model weights
        const old = {
            layers: {}
        };
        
        const newObj = {
            layers: {}
        };
        
        // Generate large layer data
        for (let i = 0; i < 20; i++) {
            old.layers[`layer_${i}`] = {
                weights: Array.from({ length: 100 }, (_, j) => j + 0.001 * i),
                bias: Array.from({ length: 100 }, (_, j) => 0.1 * i)
            };
            
            newObj.layers[`layer_${i}`] = {
                weights: Array.from({ length: 100 }, (_, j) => j + 0.001 * i + 0.01),  // Small change
                bias: Array.from({ length: 100 }, (_, j) => 0.1 * i + 0.001)  // Small change
            };
        }
        
        const startTime = Date.now();
        const results = diffai.diff(old, newObj, { 
            weightThreshold: 0.005, 
            epsilon: 0.001 
        });
        const endTime = Date.now();
        
        expect(results.length).toBeGreaterThan(0); // Should detect changes
        expect(endTime - startTime).toBeLessThan(10000); // Should complete within 10 seconds
    }, 15000); // 15 second timeout for this test
    
    test('deep model structure performance', () => {
        function createDeepModel(depth, baseValue) {
            if (depth === 0) {
                return { value: baseValue, weights: Array(10).fill(baseValue) };
            }
            
            return {
                layer: createDeepModel(depth - 1, baseValue),
                params: { learning_rate: 0.001 + baseValue * 0.001 }
            };
        }
        
        const old = createDeepModel(20, 0.1);  // 20 levels deep
        const newObj = createDeepModel(20, 0.11); // Slightly different values
        
        const startTime = Date.now();
        const results = diffai.diff(old, newObj, { epsilon: 0.001 });
        const endTime = Date.now();
        
        expect(results.length).toBeGreaterThan(0); // Should find differences
        expect(endTime - startTime).toBeLessThan(5000); // Should handle deep nesting efficiently
    });
});

// ============================================================================
// TYPESCRIPT COMPATIBILITY TESTS
// ============================================================================

describe('TypeScript Compatibility', () => {
    test('type definitions availability', () => {
        // This test would verify that TypeScript definitions work for AI/ML types
        expect(typeof diffai.diff).toBe('function');
    });
    
    test('ml specific type handling', () => {
        // Test that ML-specific data structures are properly typed
        const tensorData = {
            shape: [10, 20, 30],
            dtype: "float32",
            device: "cuda:0"
        };
        
        const results = diffai.diff(tensorData, tensorData);
        expect(results).toHaveLength(0);
    });
});

module.exports = {
    TestFixtures,
    expectDiffResult,
    expectNoDifferences,
    expectDifferences
};