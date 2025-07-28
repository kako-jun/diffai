#!/usr/bin/env tsx

/**
 * diffai-js TypeScript Examples - UNIFIED API DESIGN
 * 
 * Demonstrates native NAPI-RS API usage for AI/ML model comparison
 * Users load model data themselves and call the unified diff() function
 */

import { diff, DiffOptions, DiffResult, TensorStatistics } from './index';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';

// Colors for console output
const colors = {
    green: '\x1b[32m',
    red: '\x1b[31m',
    yellow: '\x1b[33m',
    blue: '\x1b[34m',
    cyan: '\x1b[36m',
    magenta: '\x1b[35m',
    reset: '\x1b[0m'
} as const;

function log(message: string, color: keyof typeof colors = 'reset'): void {
    console.log(`${colors[color]}${message}${colors.reset}`);
}

function header(message: string): void {
    log(`\n${message}`, 'cyan');
    log('='.repeat(message.length), 'cyan');
}

function example(title: string, description: string): void {
    log(`\n${title}`, 'yellow');
    log(`   ${description}`, 'blue');
}

// Mock model data structures for examples
interface ModelMetadata {
    name: string;
    version: string;
    architecture: {
        layers: number;
        hiddenSize: number;
        attentionHeads: number;
    };
    training: {
        epochs: number;
        learningRate: number;
        optimizer: string;
        lossFunction: string;
    };
    performance: {
        accuracy: number;
        loss: number;
        validationAccuracy: number;
    };
    weights?: Record<string, number[]>;
}

interface TensorData {
    shape: number[];
    dtype: string;
    data: number[];
    gradients?: number[];
}

async function runExamples(): Promise<void> {
    header('diffai-js Native API Examples');
    
    // Create temporary directory
    const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'diffai-examples-'));
    const oldCwd = process.cwd();
    process.chdir(tempDir);

    try {
        // Example 1: Model Architecture Comparison
        header('1. Model Architecture Evolution');
        
        const modelV1: ModelMetadata = {
            name: "bert-base",
            version: "1.0.0",
            architecture: {
                layers: 12,
                hiddenSize: 768,
                attentionHeads: 12
            },
            training: {
                epochs: 10,
                learningRate: 2e-5,
                optimizer: "adam",
                lossFunction: "cross_entropy"
            },
            performance: {
                accuracy: 0.92,
                loss: 0.08,
                validationAccuracy: 0.89
            }
        };

        const modelV2: ModelMetadata = {
            name: "bert-large",
            version: "2.0.0",
            architecture: {
                layers: 24,
                hiddenSize: 1024,
                attentionHeads: 16
            },
            training: {
                epochs: 15,
                learningRate: 1e-5,
                optimizer: "adamw",
                lossFunction: "cross_entropy"
            },
            performance: {
                accuracy: 0.95,
                loss: 0.05,
                validationAccuracy: 0.93
            }
        };

        example(
            'BERT Model Evolution Analysis',
            'Compare model architecture changes from base to large variant'
        );
        
        const mlOptions: DiffOptions = {
            diffaiOptions: {
                mlAnalysisEnabled: true,
                scientificPrecision: true,
                statisticalSummary: true
            },
            showTypes: true
        };

        const results1 = await diff(modelV1, modelV2, mlOptions);
        log('Architecture Changes:', 'green');
        results1.forEach((result: DiffResult) => {
            console.log(`${result.type}: ${result.path}`);
            if (result.type === 'modified') {
                console.log(`  ${result.oldValue} → ${result.newValue}`);
            }
        });

        // Example 2: Tensor Comparison with Statistical Analysis
        header('2. Tensor Weight Analysis');

        const oldWeights: Record<string, TensorData> = {
            "encoder.layer.0.attention.self.query.weight": {
                shape: [768, 768],
                dtype: "float32",
                data: Array.from({ length: 100 }, () => Math.random() * 0.1 - 0.05)
            },
            "encoder.layer.0.attention.self.key.weight": {
                shape: [768, 768], 
                dtype: "float32",
                data: Array.from({ length: 100 }, () => Math.random() * 0.1 - 0.05)
            }
        };

        const newWeights: Record<string, TensorData> = {
            "encoder.layer.0.attention.self.query.weight": {
                shape: [768, 768],
                dtype: "float32",
                data: Array.from({ length: 100 }, () => Math.random() * 0.12 - 0.06)
            },
            "encoder.layer.0.attention.self.key.weight": {
                shape: [768, 768],
                dtype: "float32", 
                data: Array.from({ length: 100 }, () => Math.random() * 0.1 - 0.05)
            },
            "encoder.layer.0.attention.self.value.weight": {
                shape: [768, 768],
                dtype: "float32",
                data: Array.from({ length: 100 }, () => Math.random() * 0.1 - 0.05)
            }
        };

        const tensorOptions: DiffOptions = {
            epsilon: 1e-6,
            diffaiOptions: {
                mlAnalysisEnabled: true,
                tensorComparisonMode: 'statistical',
                weightThreshold: 1e-4,
                statisticalSummary: true,
                scientificPrecision: true
            }
        };

        example(
            'Fine-tuned Weight Analysis',
            'Detect significant weight changes after model fine-tuning'
        );

        const results2 = await diff(oldWeights, newWeights, tensorOptions);
        log('Weight Changes:', 'green');
        results2.forEach((result: DiffResult) => {
            if (result.type === 'weightSignificantChange' && result.statistics) {
                const stats = result.statistics;
                console.log(`📊 ${result.path}:`);
                console.log(`   Mean change: ${stats.meanChange.toExponential(3)}`);
                console.log(`   Std dev: ${stats.stdDev.toExponential(3)}`);
                console.log(`   Changed elements: ${stats.changedElements}/${stats.totalElements}`);
            }
        });

        // Example 3: Training Progress Comparison
        header('3. Training Checkpoint Analysis');

        const checkpoint1 = {
            epoch: 1,
            step: 1000,
            model_state: {
                "layer1.weight": {
                    shape: [128, 64],
                    mean: 0.02,
                    std: 0.15,
                    norm: 2.45
                },
                "layer1.bias": {
                    shape: [128],
                    mean: 0.001,
                    std: 0.05,
                    norm: 0.18
                }
            },
            optimizer_state: {
                learning_rate: 0.001,
                momentum: 0.9,
                weight_decay: 1e-4
            },
            metrics: {
                train_loss: 2.45,
                train_accuracy: 0.65,
                val_loss: 2.67,
                val_accuracy: 0.62
            }
        };

        const checkpoint10 = {
            epoch: 10,
            step: 10000,
            model_state: {
                "layer1.weight": {
                    shape: [128, 64],
                    mean: 0.018,
                    std: 0.142,
                    norm: 2.31
                },
                "layer1.bias": {
                    shape: [128],
                    mean: 0.003,
                    std: 0.048,
                    norm: 0.172
                }
            },
            optimizer_state: {
                learning_rate: 0.0005,
                momentum: 0.9,
                weight_decay: 1e-4
            },
            metrics: {
                train_loss: 0.85,
                train_accuracy: 0.94,
                val_loss: 0.92,
                val_accuracy: 0.91
            }
        };

        const trainingOptions: DiffOptions = {
            diffaiOptions: {
                mlAnalysisEnabled: true,
                gradientAnalysis: true,
                statisticalSummary: true
            },
            epsilon: 1e-5
        };

        example(
            'Training Progress Monitoring',
            'Compare checkpoints to track training convergence'
        );

        const results3 = await diff(checkpoint1, checkpoint10, trainingOptions);
        log('Training Progress:', 'green');
        results3.forEach((result: DiffResult) => {
            if (result.path.includes('metrics')) {
                console.log(`📈 ${result.path}: ${result.oldValue} → ${result.newValue}`);
                if (result.path.includes('accuracy')) {
                    const improvement = Number(result.newValue) - Number(result.oldValue);
                    console.log(`   Improvement: +${(improvement * 100).toFixed(1)}%`);
                }
            }
        });

        // Example 4: Model Precision Comparison
        header('4. Precision Format Analysis');

        const fp32Model = {
            precision: "float32",
            weights: {
                "linear.weight": {
                    dtype: "float32",
                    values: [0.123456789, 0.987654321, -0.456789123]
                }
            },
            memory_usage: "2.4GB",
            inference_speed: "120ms"
        };

        const fp16Model = {
            precision: "float16", 
            weights: {
                "linear.weight": {
                    dtype: "float16",
                    values: [0.1235, 0.9877, -0.4568]  // Reduced precision
                }
            },
            memory_usage: "1.2GB",
            inference_speed: "85ms"
        };

        const precisionOptions: DiffOptions = {
            epsilon: 1e-3,  // Higher tolerance for precision differences
            diffaiOptions: {
                mlAnalysisEnabled: true,
                scientificPrecision: true
            }
        };

        example(
            'Quantization Impact Analysis',
            'Compare FP32 vs FP16 model to assess precision loss'
        );

        const results4 = await diff(fp32Model, fp16Model, precisionOptions);
        log('Precision Changes:', 'green');
        results4.forEach((result: DiffResult) => {
            if (result.type === 'precisionChanged') {
                console.log(`🔢 ${result.path}: ${result.oldType} → ${result.newType}`);
            } else if (result.path.includes('memory_usage') || result.path.includes('inference_speed')) {
                console.log(`⚡ ${result.path}: ${result.oldValue} → ${result.newValue}`);
            }
        });

        // Example 5: Error Handling for ML Data
        header('5. ML-Specific Error Handling');

        example(
            'Handling Invalid Tensor Data',
            'Demonstrate robust error handling for malformed ML data'
        );

        try {
            const invalidModel = {
                weights: {
                    "invalid_tensor": {
                        shape: [100, 100],
                        data: "not_a_number_array"  // Invalid data type
                    }
                }
            };
            
            await diff(modelV1, invalidModel);
        } catch (error) {
            log(`Caught ML data error: ${error}`, 'red');
        }

        // Example 6: Performance Monitoring for Large Models
        header('6. Large Model Performance');

        const largeModel1 = {
            layers: Object.fromEntries(
                Array.from({ length: 50 }, (_, i) => [
                    `layer_${i}`,
                    {
                        weights: Array.from({ length: 1000 }, () => Math.random()),
                        biases: Array.from({ length: 100 }, () => Math.random())
                    }
                ])
            )
        };

        const largeModel2 = {
            layers: Object.fromEntries(
                Array.from({ length: 50 }, (_, i) => [
                    `layer_${i}`,
                    {
                        weights: Array.from({ length: 1000 }, () => Math.random() * 1.1),
                        biases: Array.from({ length: 100 }, () => Math.random() * 0.9)
                    }
                ])
            )
        };

        const perfOptions: DiffOptions = {
            useMemoryOptimization: true,
            batchSize: 10,
            diffaiOptions: {
                mlAnalysisEnabled: true,
                tensorComparisonMode: 'statistical'
            }
        };

        example(
            'Large Model Comparison with Memory Optimization',
            'Efficiently compare large models using batching and statistical analysis'
        );

        const startTime = Date.now();
        const results6 = await diff(largeModel1, largeModel2, perfOptions);
        const endTime = Date.now();

        log(`Analyzed ${results6.length} layer differences in ${endTime - startTime}ms`, 'green');
        log(`Memory optimization enabled for efficient processing`, 'blue');

        // Summary
        header('Summary');
        log('✅ All diffai examples completed successfully!', 'green');
        log('\nML-Specific Benefits:', 'cyan');
        log('  • Native tensor comparison support', 'blue');
        log('  • Statistical analysis of weight changes', 'blue');
        log('  • Training progress monitoring', 'blue');
        log('  • Precision format analysis', 'blue');
        log('  • Memory-efficient large model handling', 'blue');
        log('  • Scientific notation for precise values', 'blue');

        log('\nML Use Cases:', 'cyan');
        log('  • Model version comparison', 'blue');
        log('  • Training checkpoint analysis', 'blue');
        log('  • Quantization impact assessment', 'blue');
        log('  • Architecture evolution tracking', 'blue');
        log('  • Fine-tuning progress monitoring', 'blue');

    } catch (error) {
        log(`\nError running examples: ${error}`, 'red');
        console.error(error);
    } finally {
        // Cleanup
        process.chdir(oldCwd);
        try {
            fs.rmSync(tempDir, { recursive: true, force: true });
        } catch (cleanupErr) {
            log(`Cleanup warning: ${cleanupErr}`, 'yellow');
        }
    }
}

// Run examples if called directly
if (require.main === module) {
    runExamples().catch(console.error);
}

export { runExamples };