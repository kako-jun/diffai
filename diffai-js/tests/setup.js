// Jest setup file for diffai-js unified API tests

// Global test configuration for AI/ML testing
jest.setTimeout(60000); // 60 seconds default timeout for AI/ML operations

// Console configuration for tests
if (process.env.NODE_ENV === 'test') {
    // Suppress console.log in tests unless explicitly needed
    const originalConsoleLog = console.log;
    console.log = (...args) => {
        if (process.env.DEBUG_TESTS || process.env.DEBUG_ML_TESTS) {
            originalConsoleLog(...args);
        }
    };
}

// Global test helpers for AI/ML testing
global.expectAsync = async (fn) => {
    let error;
    try {
        await fn();
    } catch (e) {
        error = e;
    }
    return expect(() => {
        if (error) throw error;
    });
};

// Helper for testing AI/ML specific result types
global.expectMLResultType = (result, expectedType) => {
    expect(result).toHaveProperty('type', expectedType);
    
    // Check for ML-specific fields based on type
    switch (expectedType) {
        case 'LearningRateChanged':
            expect(result).toHaveProperty('oldLearningRate');
            expect(result).toHaveProperty('newLearningRate');
            break;
        case 'AccuracyChange':
            expect(result).toHaveProperty('oldAccuracy');
            expect(result).toHaveProperty('newAccuracy');
            break;
        case 'LossChange':
            expect(result).toHaveProperty('oldLoss');
            expect(result).toHaveProperty('newLoss');
            break;
        case 'WeightSignificantChange':
            expect(result).toHaveProperty('magnitude');
            expect(typeof result.magnitude).toBe('number');
            break;
        case 'TensorShapeChanged':
            expect(result).toHaveProperty('oldShape');
            expect(result).toHaveProperty('newShape');
            expect(Array.isArray(result.oldShape)).toBe(true);
            expect(Array.isArray(result.newShape)).toBe(true);
            break;
        case 'ModelArchitectureChanged':
            expect(result).toHaveProperty('oldDescription');
            expect(result).toHaveProperty('newDescription');
            break;
    }
};

// Helper for generating test tensor data
global.generateTensorData = (shape, dtype = 'float32', fillValue = 0.1) => {
    const totalSize = shape.reduce((acc, dim) => acc * dim, 1);
    return {
        shape: shape,
        dtype: dtype,
        data: Array(totalSize).fill(fillValue),
        size: totalSize
    };
};

// Helper for generating ML model test data
global.generateMLModel = (layerCount = 3, baseUnits = 64) => {
    const layers = [];
    
    for (let i = 0; i < layerCount; i++) {
        layers.push({
            name: `layer_${i}`,
            type: i === layerCount - 1 ? 'output' : 'hidden',
            units: Math.floor(baseUnits / Math.pow(2, i)),
            activation: i === layerCount - 1 ? 'softmax' : 'relu',
            weights: {
                shape: i === 0 ? [baseUnits, 784] : [Math.floor(baseUnits / Math.pow(2, i)), Math.floor(baseUnits / Math.pow(2, i - 1))],
                mean: 0.01 * (i + 1),
                std: 0.1 * (i + 1)
            }
        });
    }
    
    return {
        architecture: 'sequential',
        layers: layers,
        optimizer: {
            type: 'Adam',
            learning_rate: 0.001,
            beta1: 0.9,
            beta2: 0.999
        },
        training: {
            epochs: 100,
            batch_size: 32,
            loss: 0.5,
            accuracy: 0.85
        }
    };
};

// Memory management for large AI/ML test data
global.cleanupMLTestData = () => {
    // Force garbage collection if available (useful for large tensor data)
    if (global.gc) {
        global.gc();
    }
};

// Error handler for unhandled promise rejections in tests
process.on('unhandledRejection', (reason, promise) => {
    console.error('Unhandled Rejection at:', promise, 'reason:', reason);
});

// Cleanup after tests
afterAll(() => {
    global.cleanupMLTestData();
});

// Setup for each test
beforeEach(() => {
    // Reset any global state that might affect AI/ML tests
});

afterEach(() => {
    // Cleanup after each test to prevent memory leaks
    global.cleanupMLTestData();
});