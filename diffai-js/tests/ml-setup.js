// ML-specific setup for diffai-js tests

// Increase memory limits for large AI/ML data processing
if (process.env.NODE_ENV === 'test') {
    // Set max old space size if not already set
    if (!process.env.NODE_OPTIONS || !process.env.NODE_OPTIONS.includes('--max-old-space-size')) {
        // This is informational since we can't change memory limits at runtime
        console.log('Tip: Set NODE_OPTIONS="--max-old-space-size=4096" for large ML dataset tests');
    }
}

// Configure test environment for AI/ML operations
global.ML_TEST_CONFIG = {
    // Default tensor sizes for testing
    SMALL_TENSOR_SIZE: [10, 10],
    MEDIUM_TENSOR_SIZE: [100, 100], 
    LARGE_TENSOR_SIZE: [1000, 100],
    
    // Default precision settings
    DEFAULT_EPSILON: 1e-6,
    SCIENTIFIC_EPSILON: 1e-12,
    
    // Default thresholds
    WEIGHT_THRESHOLD: 0.01,
    LEARNING_RATE_THRESHOLD: 1e-6,
    
    // Timeout settings for different test sizes
    SMALL_TEST_TIMEOUT: 5000,   // 5 seconds
    MEDIUM_TEST_TIMEOUT: 15000, // 15 seconds  
    LARGE_TEST_TIMEOUT: 60000,  // 60 seconds
};

// Mock AI/ML libraries if not available (for CI environments)
global.mockMLLibraries = () => {
    // Mock PyTorch-like tensor operations
    global.mockTensor = (shape, fillValue = 0.0) => ({
        shape: shape,
        dtype: 'float32',
        data: Array(shape.reduce((a, b) => a * b, 1)).fill(fillValue),
        device: 'cpu'
    });
    
    // Mock model operations
    global.mockModel = (layers = 3) => ({
        layers: Array(layers).fill(null).map((_, i) => ({
            name: `layer_${i}`,
            parameters: global.mockTensor([128, 64])
        })),
        state_dict: () => ({}),
        eval: () => ({}),
        train: () => ({})
    });
};

// Initialize mock libraries
global.mockMLLibraries();

// Performance monitoring for AI/ML tests
global.MLPerformanceMonitor = {
    start: function(testName) {
        this.startTime = Date.now();
        this.startMemory = process.memoryUsage();
        this.testName = testName;
        
        if (process.env.DEBUG_ML_PERFORMANCE) {
            console.log(`[${testName}] Performance monitoring started`);
            console.log(`[${testName}] Initial memory:`, this.startMemory);
        }
    },
    
    end: function() {
        const endTime = Date.now();
        const endMemory = process.memoryUsage();
        const duration = endTime - this.startTime;
        
        const memoryDelta = {
            rss: endMemory.rss - this.startMemory.rss,
            heapUsed: endMemory.heapUsed - this.startMemory.heapUsed,
            heapTotal: endMemory.heapTotal - this.startMemory.heapTotal,
            external: endMemory.external - this.startMemory.external
        };
        
        if (process.env.DEBUG_ML_PERFORMANCE) {
            console.log(`[${this.testName}] Performance monitoring ended`);
            console.log(`[${this.testName}] Duration: ${duration}ms`);
            console.log(`[${this.testName}] Memory delta:`, memoryDelta);
        }
        
        // Warn if memory usage is high
        if (memoryDelta.heapUsed > 50 * 1024 * 1024) { // 50MB
            console.warn(`[${this.testName}] High memory usage detected: ${Math.round(memoryDelta.heapUsed / 1024 / 1024)}MB`);
        }
        
        // Warn if duration is long
        if (duration > 30000) { // 30 seconds
            console.warn(`[${this.testName}] Long test duration detected: ${duration}ms`);
        }
        
        return { duration, memoryDelta };
    }
};

// Utility for creating realistic ML test data
global.createRealisticMLData = {
    // Create realistic model weights
    modelWeights: function(layerSizes) {
        const weights = {};
        for (let i = 0; i < layerSizes.length - 1; i++) {
            const weightShape = [layerSizes[i + 1], layerSizes[i]];
            const biasShape = [layerSizes[i + 1]];
            
            weights[`layer_${i}_weight`] = {
                shape: weightShape,
                dtype: 'float32',
                data: Array(weightShape[0] * weightShape[1])
                    .fill(0)
                    .map(() => (Math.random() - 0.5) * 0.1) // Xavier-like initialization
            };
            
            weights[`layer_${i}_bias`] = {
                shape: biasShape,
                dtype: 'float32', 
                data: Array(biasShape[0]).fill(0)
            };
        }
        return weights;
    },
    
    // Create realistic training metrics
    trainingMetrics: function(epochs) {
        const generateDecreasingLoss = (start, end, noise = 0.1) => {
            return Array(epochs).fill(0).map((_, i) => {
                const progress = i / (epochs - 1);
                const value = start * Math.exp(-progress * Math.log(start / end));
                const noiseValue = value * (Math.random() - 0.5) * noise;
                return Math.max(0, value + noiseValue);
            });
        };
        
        const generateIncreasingAccuracy = (start, end, noise = 0.05) => {
            return Array(epochs).fill(0).map((_, i) => {
                const progress = i / (epochs - 1);
                const value = start + (end - start) * (1 - Math.exp(-progress * 3));
                const noiseValue = value * (Math.random() - 0.5) * noise;
                return Math.min(1, Math.max(0, value + noiseValue));
            });
        };
        
        return {
            train_loss: generateDecreasingLoss(2.0, 0.1),
            val_loss: generateDecreasingLoss(2.2, 0.15),
            train_accuracy: generateIncreasingAccuracy(0.1, 0.95),
            val_accuracy: generateIncreasingAccuracy(0.08, 0.88)
        };
    },
    
    // Create realistic model architecture
    modelArchitecture: function(complexity = 'medium') {
        const complexityMap = {
            simple: { layers: 3, maxUnits: 128 },
            medium: { layers: 5, maxUnits: 512 },
            complex: { layers: 10, maxUnits: 2048 }
        };
        
        const config = complexityMap[complexity] || complexityMap.medium;
        const layers = [];
        
        for (let i = 0; i < config.layers; i++) {
            if (i === 0) {
                layers.push({
                    name: 'input',
                    type: 'Input',
                    shape: [784] // MNIST-like input
                });
            } else if (i === config.layers - 1) {
                layers.push({
                    name: 'output',
                    type: 'Dense',
                    units: 10, // Classification output
                    activation: 'softmax'
                });
            } else {
                layers.push({
                    name: `hidden_${i}`,
                    type: 'Dense', 
                    units: Math.floor(config.maxUnits / Math.pow(2, i - 1)),
                    activation: 'relu'
                });
                
                // Add dropout for complexity
                if (complexity !== 'simple' && i > 1) {
                    layers.push({
                        name: `dropout_${i}`,
                        type: 'Dropout',
                        rate: 0.2 + (i - 1) * 0.1
                    });
                }
            }
        }
        
        return { layers };
    }
};

// Validate test environment
if (process.env.NODE_ENV === 'test') {
    // Check if we have enough memory for ML tests
    const memInfo = process.memoryUsage();
    const availableMemory = memInfo.heapTotal;
    
    if (availableMemory < 50 * 1024 * 1024) { // Less than 50MB
        console.warn('Low memory detected. Large ML tests may fail.');
    }
    
    // Check Node.js version for compatibility
    const nodeVersion = process.version;
    const majorVersion = parseInt(nodeVersion.split('.')[0].substring(1));
    
    if (majorVersion < 16) {
        console.warn('Node.js version < 16 detected. Some AI/ML features may not work correctly.');
    }
}