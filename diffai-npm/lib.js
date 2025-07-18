/**
 * Node.js API wrapper for diffai CLI tool
 * 
 * This module provides a JavaScript API for the diffai CLI tool,
 * allowing you to compare AI/ML model files programmatically.
 */

const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const { writeFileSync, mkdtempSync, rmSync } = require('fs');
const { tmpdir } = require('os');

/**
 * @typedef {'safetensors'|'pytorch'|'numpy'|'matlab'|'json'|'yaml'|'toml'|'xml'|'ini'|'csv'} Format
 * @typedef {'cli'|'json'|'yaml'} OutputFormat
 */

/**
 * Options for diffai operations
 * @typedef {Object} DiffaiOptions
 * @property {Format} [format] - Input file format
 * @property {OutputFormat} [output] - Output format
 * @property {boolean} [stats=false] - Show detailed tensor statistics
 * @property {boolean} [verbose=false] - Show verbose processing information
 * @property {boolean} [recursive=false] - Compare directories recursively
 * @property {boolean} [quantizationAnalysis=false] - Enable quantization analysis
 * @property {boolean} [sortByChangeMagnitude=false] - Sort changes by magnitude
 * @property {boolean} [showLayerImpact=false] - Show layer-wise impact analysis
 * @property {boolean} [architectureComparison=false] - Compare model architectures
 * @property {boolean} [memoryAnalysis=false] - Analyze memory usage differences
 * @property {boolean} [anomalyDetection=false] - Detect numerical anomalies
 * @property {boolean} [changeSummary=false] - Show detailed change summary
 * @property {boolean} [convergenceAnalysis=false] - Analyze convergence state
 * @property {boolean} [gradientAnalysis=false] - Analyze gradient information
 * @property {boolean} [similarityMatrix=false] - Generate similarity matrix
 * @property {boolean} [learningProgress=false] - Analyze learning progress between checkpoints
 * @property {boolean} [inferenceSpeedEstimate=false] - Estimate inference speed characteristics
 * @property {boolean} [regressionTest=false] - Perform automated regression testing
 * @property {boolean} [alertOnDegradation=false] - Alert on performance degradation
 * @property {boolean} [reviewFriendly=false] - Generate review-friendly output
 * @property {boolean} [deploymentReadiness=false] - Assess deployment readiness
 * @property {boolean} [paramEfficiencyAnalysis=false] - Analyze parameter efficiency
 * @property {boolean} [hyperparameterImpact=false] - Analyze hyperparameter impact
 * @property {boolean} [learningRateAnalysis=false] - Analyze learning rate effects
 * @property {boolean} [performanceImpactEstimate=false] - Estimate performance impact
 * @property {boolean} [generateReport=false] - Generate comprehensive analysis report
 * @property {boolean} [markdownOutput=false] - Output results in markdown format
 * @property {boolean} [includeCharts=false] - Include charts and visualizations
 * @property {boolean} [embeddingAnalysis=false] - Analyze embedding layer changes
 * @property {boolean} [attentionAnalysis=false] - Analyze attention mechanisms
 * @property {boolean} [headImportance=false] - Analyze attention head importance
 * @property {boolean} [attentionPatternDiff=false] - Compare attention patterns
 * @property {boolean} [clusteringChange=false] - Analyze clustering changes
 * @property {boolean} [hyperparameterComparison=false] - Compare hyperparameters
 * @property {boolean} [learningCurveAnalysis=false] - Analyze learning curves
 * @property {boolean} [statisticalSignificance=false] - Perform statistical significance testing
 * @property {string} [path] - Filter differences by path
 * @property {string} [ignoreKeysRegex] - Ignore keys matching regex pattern
 * @property {string} [arrayIdKey] - Key to use for identifying array elements
 * @property {number} [epsilon] - Tolerance for float comparisons
 */

/**
 * Result of a diffai operation
 * @typedef {Object} DiffaiResult
 * @property {string} type - Type of difference ('Added', 'Removed', 'Modified', 'TypeChanged')
 * @property {string} path - Path to the changed element
 * @property {*} [oldValue] - Old value (for Modified/TypeChanged)
 * @property {*} [newValue] - New value (for Modified/TypeChanged/Added)
 * @property {*} [value] - Value (for Removed)
 * @property {Object} [stats] - Statistical information when stats=true
 * @property {Object} [analysis] - Analysis results when analysis flags are enabled
 */

/**
 * Error thrown when diffai command fails
 */
class DiffaiError extends Error {
  constructor(message, exitCode, stderr) {
    super(message);
    this.name = 'DiffaiError';
    this.exitCode = exitCode;
    this.stderr = stderr;
  }
}

/**
 * Get the path to the diffai binary
 * @returns {string} Path to diffai binary
 */
function getDiffaiBinaryPath() {
  // Determine platform-specific subdirectory
  const platform = process.platform;
  const arch = process.arch;
  let subdir;
  
  if (platform === 'win32') {
    subdir = 'win32-x64';
  } else if (platform === 'darwin') {
    subdir = arch === 'arm64' ? 'darwin-arm64' : 'darwin-x64';
  } else if (platform === 'linux') {
    subdir = arch === 'arm64' ? 'linux-arm64' : 'linux-x64';
  } else {
    throw new Error(`Unsupported platform: ${platform}-${arch}`);
  }
  
  // Check if platform-specific binary exists (OS hierarchy required)
  const binaryName = process.platform === 'win32' ? 'diffai.exe' : 'diffai';
  const platformBinaryPath = path.join(__dirname, 'bin', subdir, binaryName);
  
  if (fs.existsSync(platformBinaryPath)) {
    return platformBinaryPath;
  }
  
  // Error if binary not found - no system PATH fallback allowed
  throw new Error(`Binary not found at ${platformBinaryPath}. Platform: ${platform}-${arch}. This might indicate a packaging issue. Please report this at: https://github.com/kako-jun/diffai/issues`);
}

/**
 * Execute diffai command
 * @param {string[]} args - Command arguments
 * @returns {Promise<{stdout: string, stderr: string}>} Command output
 */
function executeDiffai(args) {
  return new Promise((resolve, reject) => {
    const diffaiPath = getDiffaiBinaryPath();
    
    const child = spawn(diffaiPath, args, {
      stdio: ['pipe', 'pipe', 'pipe']
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
      if (code === 0 || code === 1) {
        // Exit code 1 means differences found, which is expected
        resolve({ stdout, stderr });
      } else {
        reject(new DiffaiError(
          `diffai exited with code ${code}`,
          code,
          stderr
        ));
      }
    });
    
    child.on('error', (err) => {
      if (err.code === 'ENOENT') {
        reject(new DiffaiError(
          'diffai command not found. Please install diffai CLI tool.',
          -1,
          ''
        ));
      } else {
        reject(new DiffaiError(err.message, -1, ''));
      }
    });
  });
}

/**
 * Compare two AI/ML model files using diffai
 * 
 * @param {string} input1 - Path to first model file
 * @param {string} input2 - Path to second model file
 * @param {DiffaiOptions} [options={}] - Comparison options
 * @returns {Promise<string|DiffaiResult[]>} String output for CLI format, or array of DiffaiResult for JSON format
 * 
 * @example
 * // Basic model comparison
 * const result = await diff('model_v1.safetensors', 'model_v2.safetensors');
 * console.log(result);
 * 
 * @example
 * // JSON output format with statistics
 * const jsonResult = await diff('baseline.pt', 'finetuned.pt', {
 *   output: 'json',
 *   stats: true,
 *   convergenceAnalysis: true
 * });
 * for (const diffItem of jsonResult) {
 *   console.log(diffItem);
 * }
 * 
 * @example
 * // Advanced ML analysis
 * const analysis = await diff('model_old.safetensors', 'model_new.safetensors', {
 *   output: 'json',
 *   architectureComparison: true,
 *   memoryAnalysis: true,
 *   anomalyDetection: true
 * });
 * console.log(analysis);
 */
async function diff(input1, input2, options = {}) {
  const args = [input1, input2];
  
  // Add output format option
  if (options.output) {
    args.push('--output', options.output);
  }
  
  // Add format option
  if (options.format) {
    args.push('--format', options.format);
  }
  
  // Add basic options
  if (options.recursive) {
    args.push('--recursive');
  }
  
  if (options.verbose) {
    args.push('--verbose');
  }
  
  if (options.path) {
    args.push('--path', options.path);
  }
  
  if (options.ignoreKeysRegex) {
    args.push('--ignore-keys-regex', options.ignoreKeysRegex);
  }
  
  if (options.epsilon !== undefined) {
    args.push('--epsilon', options.epsilon.toString());
  }
  
  if (options.arrayIdKey) {
    args.push('--array-id-key', options.arrayIdKey);
  }
  
  // Add ML analysis options
  if (options.showLayerImpact) {
    args.push('--show-layer-impact');
  }
  
  if (options.quantizationAnalysis) {
    args.push('--quantization-analysis');
  }
  
  if (options.sortByChangeMagnitude) {
    args.push('--sort-by-change-magnitude');
  }
  
  if (options.stats) {
    args.push('--stats');
  }
  
  if (options.learningProgress) {
    args.push('--learning-progress');
  }
  
  if (options.convergenceAnalysis) {
    args.push('--convergence-analysis');
  }
  
  if (options.anomalyDetection) {
    args.push('--anomaly-detection');
  }
  
  if (options.gradientAnalysis) {
    args.push('--gradient-analysis');
  }
  
  if (options.memoryAnalysis) {
    args.push('--memory-analysis');
  }
  
  if (options.inferenceSpeedEstimate) {
    args.push('--inference-speed-estimate');
  }
  
  if (options.regressionTest) {
    args.push('--regression-test');
  }
  
  if (options.alertOnDegradation) {
    args.push('--alert-on-degradation');
  }
  
  if (options.reviewFriendly) {
    args.push('--review-friendly');
  }
  
  if (options.changeSummary) {
    args.push('--change-summary');
  }
  
  if (options.deploymentReadiness) {
    args.push('--deployment-readiness');
  }
  
  if (options.architectureComparison) {
    args.push('--architecture-comparison');
  }
  
  if (options.paramEfficiencyAnalysis) {
    args.push('--param-efficiency-analysis');
  }
  
  if (options.hyperparameterImpact) {
    args.push('--hyperparameter-impact');
  }
  
  if (options.learningRateAnalysis) {
    args.push('--learning-rate-analysis');
  }
  
  if (options.performanceImpactEstimate) {
    args.push('--performance-impact-estimate');
  }
  
  if (options.generateReport) {
    args.push('--generate-report');
  }
  
  if (options.markdownOutput) {
    args.push('--markdown-output');
  }
  
  if (options.includeCharts) {
    args.push('--include-charts');
  }
  
  if (options.embeddingAnalysis) {
    args.push('--embedding-analysis');
  }
  
  if (options.similarityMatrix) {
    args.push('--similarity-matrix');
  }
  
  if (options.clusteringChange) {
    args.push('--clustering-change');
  }
  
  if (options.attentionAnalysis) {
    args.push('--attention-analysis');
  }
  
  if (options.headImportance) {
    args.push('--head-importance');
  }
  
  if (options.attentionPatternDiff) {
    args.push('--attention-pattern-diff');
  }
  
  if (options.hyperparameterComparison) {
    args.push('--hyperparameter-comparison');
  }
  
  if (options.learningCurveAnalysis) {
    args.push('--learning-curve-analysis');
  }
  
  if (options.statisticalSignificance) {
    args.push('--statistical-significance');
  }
  
  
  const { stdout, stderr } = await executeDiffai(args);
  
  // If output format is JSON, parse the result
  if (options.output === 'json') {
    try {
      const jsonData = JSON.parse(stdout);
      return jsonData.map(item => {
        if (item.Added) {
          return {
            type: 'Added',
            path: item.Added[0],
            newValue: item.Added[1]
          };
        } else if (item.Removed) {
          return {
            type: 'Removed',
            path: item.Removed[0],
            value: item.Removed[1]
          };
        } else if (item.Modified) {
          return {
            type: 'Modified',
            path: item.Modified[0],
            oldValue: item.Modified[1],
            newValue: item.Modified[2]
          };
        } else if (item.TypeChanged) {
          return {
            type: 'TypeChanged',
            path: item.TypeChanged[0],
            oldValue: item.TypeChanged[1],
            newValue: item.TypeChanged[2]
          };
        }
        return item;
      });
    } catch (e) {
      throw new DiffaiError(`Failed to parse JSON output: ${e.message}`, -1, '');
    }
  }
  
  // Return raw output for other formats
  return stdout;
}

/**
 * Compare two model content strings directly (writes to temporary files)
 * 
 * @param {string} content1 - First model content (base64 or binary string)
 * @param {string} content2 - Second model content (base64 or binary string)
 * @param {Format} format - Content format
 * @param {DiffaiOptions} [options={}] - Comparison options
 * @returns {Promise<string|DiffaiResult[]>} String output for CLI format, or array of DiffaiResult for JSON format
 * 
 * @example
 * const modelData1 = fs.readFileSync('model1.safetensors');
 * const modelData2 = fs.readFileSync('model2.safetensors');
 * const result = await diffString(modelData1, modelData2, 'safetensors', { 
 *   output: 'json',
 *   stats: true 
 * });
 * console.log(result);
 */
async function diffString(content1, content2, format, options = {}) {
  // Ensure format is set
  options.format = format;
  
  // Create temporary files
  const tmpDir = mkdtempSync(path.join(tmpdir(), 'diffai-'));
  const extension = format === 'pytorch' ? 'pt' : 
                   format === 'numpy' ? 'npy' : 
                   format === 'matlab' ? 'mat' : format;
  const tmpFile1 = path.join(tmpDir, `file1.${extension}`);
  const tmpFile2 = path.join(tmpDir, `file2.${extension}`);
  
  try {
    // Write content to temporary files
    if (typeof content1 === 'string') {
      writeFileSync(tmpFile1, content1, 'utf8');
      writeFileSync(tmpFile2, content2, 'utf8');
    } else {
      // Handle binary content
      writeFileSync(tmpFile1, content1);
      writeFileSync(tmpFile2, content2);
    }
    
    // Perform diff
    return await diff(tmpFile1, tmpFile2, options);
  } finally {
    // Clean up temporary files
    rmSync(tmpDir, { recursive: true, force: true });
  }
}

/**
 * Analyze a single model file (inspection mode)
 * 
 * @param {string} modelPath - Path to the model file
 * @param {DiffaiOptions} [options={}] - Analysis options
 * @returns {Promise<string|Object>} Analysis result
 * 
 * @example
 * const analysis = await inspect('model.safetensors', {
 *   output: 'json',
 *   stats: true,
 *   memoryAnalysis: true
 * });
 * console.log(analysis);
 */
async function inspect(modelPath, options = {}) {
  // Create a dummy empty file for comparison to enable inspection mode
  const tmpDir = mkdtempSync(path.join(tmpdir(), 'diffai-inspect-'));
  const extension = path.extname(modelPath).slice(1) || 'bin';
  const emptyFile = path.join(tmpDir, `empty.${extension}`);
  
  try {
    // Create minimal empty file
    writeFileSync(emptyFile, '');
    
    // Use diff with special handling for inspection
    const result = await diff(emptyFile, modelPath, options);
    
    // For inspection mode, we're mainly interested in the "Added" items
    // which represent the structure of the single file
    return result;
  } finally {
    rmSync(tmpDir, { recursive: true, force: true });
  }
}

/**
 * Check if diffai command is available in the system
 * 
 * @returns {Promise<boolean>} True if diffai is available, false otherwise
 * 
 * @example
 * if (!(await isDiffaiAvailable())) {
 *   console.error('Please install diffai CLI tool');
 *   process.exit(1);
 * }
 */
async function isDiffaiAvailable() {
  try {
    await executeDiffai(['--version']);
    return true;
  } catch (err) {
    return false;
  }
}

/**
 * Get version information of diffai
 * 
 * @returns {Promise<string>} Version string
 * 
 * @example
 * const version = await getVersion();
 * console.log(`diffai version: ${version}`);
 */
async function getVersion() {
  try {
    const { stdout } = await executeDiffai(['--version']);
    return stdout.trim();
  } catch (err) {
    throw new DiffaiError('Failed to get version information', -1, '');
  }
}

module.exports = {
  diff,
  diffString,
  inspect,
  isDiffaiAvailable,
  getVersion,
  DiffaiError
};