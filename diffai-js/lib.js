/**
 * Node.js native bindings for diffai - UNIFIED API DESIGN
 * 
 * This module provides a JavaScript API for the diffai library using native NAPI-RS bindings.
 * It follows the unified API design principle: only the main diff() function is exposed.
 */

const { diffJs } = require('./index.js');

/**
 * @typedef {Object} DiffOptions
 * @property {number} [epsilon] - Tolerance for float comparisons
 * @property {string} [arrayIdKey] - Key to use for array element identification
 * @property {string} [ignoreKeysRegex] - Ignore keys matching regex
 * @property {string} [pathFilter] - Filter differences by path
 * @property {string} [outputFormat] - Output format ('diffai', 'json', 'yaml', 'unified')
 * @property {boolean} [showUnchanged] - Include unchanged values in output
 * @property {boolean} [showTypes] - Include type information in output
 * @property {boolean} [useMemoryOptimization] - Enable memory-efficient processing
 * @property {number} [batchSize] - Batch size for memory optimization
 * @property {Object} [diffaiOptions] - diffai-specific options
 * @property {boolean} [diffaiOptions.mlAnalysisEnabled] - Enable ML analysis
 * @property {string} [diffaiOptions.tensorComparisonMode] - Tensor comparison mode
 * @property {string} [diffaiOptions.modelFormat] - Model format ('pytorch', 'safetensors', etc.)
 * @property {boolean} [diffaiOptions.scientificPrecision] - Use scientific precision
 */

/**
 * @typedef {Object} DiffResult
 * @property {string} type - Type of difference
 * @property {string} path - Path to the changed element
 * @property {*} [oldValue] - Old value (for modified)
 * @property {*} [newValue] - New value (for modified/added)
 * @property {*} [value] - Value (for removed)
 * 
 * AI/ML specific result types:
 * - 'tensorShapeChanged' - Tensor shape change
 * - 'tensorDataChanged' - Tensor data change
 * - 'modelArchitectureChanged' - Model architecture change
 * - 'weightSignificantChange' - Significant weight change
 * - 'activationFunctionChanged' - Activation function change
 * - 'learningRateChanged' - Learning rate change
 * - 'optimizerChanged' - Optimizer change
 * - 'lossChange' - Loss change
 * - 'accuracyChange' - Accuracy change
 * - 'modelVersionChanged' - Model version change
 */

/**
 * Compare two data structures with AI/ML model support
 * 
 * This is the unified entry point for all diffai functionality.
 * Users should parse model files/data themselves and call this function.
 * 
 * @param {*} old - Old data structure (parsed model data, tensors, etc.)
 * @param {*} new - New data structure (parsed model data, tensors, etc.)
 * @param {DiffOptions} [options] - Optional configuration
 * @returns {Promise<DiffResult[]>} Array of differences
 * 
 * @example
 * // For model comparison - users handle model loading
 * const diffai = require('diffai-js');
 * 
 * // Example: Compare model metadata (users load models themselves)
 * const oldModel = {
 *   architecture: { layers: 12, hidden_size: 768 },
 *   training: { learning_rate: 0.001, optimizer: 'adam' },
 *   weights: { ... } // tensor data loaded by user
 * };
 * 
 * const newModel = {
 *   architecture: { layers: 24, hidden_size: 1024 },
 *   training: { learning_rate: 0.0005, optimizer: 'adamw' },
 *   weights: { ... } // tensor data loaded by user
 * };
 * 
 * const results = await diffai.diff(oldModel, newModel, {
 *   diffaiOptions: {
 *     mlAnalysisEnabled: true,
 *     tensorComparisonMode: 'statistical',
 *     scientificPrecision: true
 *   }
 * });
 * 
 * // For regular JSON with AI-enhanced analysis
 * const fs = require('fs');
 * const oldData = JSON.parse(fs.readFileSync('model_config_v1.json', 'utf8'));
 * const newData = JSON.parse(fs.readFileSync('model_config_v2.json', 'utf8'));
 * const results = await diffai.diff(oldData, newData);
 */
async function diff(old, new, options = {}) {
    try {
        return diffJs(old, new, options);
    } catch (error) {
        throw new Error(`Diffai operation failed: ${error.message}`);
    }
}

module.exports = {
    diff
};

// For compatibility with CommonJS and ES modules
module.exports.default = module.exports;