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
 * @property {boolean} [quiet=false] - Suppress output (exit code only)
 * @property {string} [path] - Filter differences by path
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
  // Check if local binary exists (installed via postinstall)
  const binaryName = process.platform === 'win32' ? 'diffai.exe' : 'diffai';
  const localBinaryPath = path.join(__dirname, 'bin', binaryName);
  
  if (fs.existsSync(localBinaryPath)) {
    return localBinaryPath;
  }
  
  // Fall back to system PATH
  return 'diffai';
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
  
  // Add ML analysis options
  if (options.stats) {
    args.push('--stats');
  }
  
  if (options.quantizationAnalysis) {
    args.push('--quantization-analysis');
  }
  
  if (options.sortByChangeMagnitude) {
    args.push('--sort-by-change-magnitude');
  }
  
  if (options.showLayerImpact) {
    args.push('--show-layer-impact');
  }
  
  if (options.architectureComparison) {
    args.push('--architecture-comparison');
  }
  
  if (options.memoryAnalysis) {
    args.push('--memory-analysis');
  }
  
  if (options.anomalyDetection) {
    args.push('--anomaly-detection');
  }
  
  if (options.changeSummary) {
    args.push('--change-summary');
  }
  
  if (options.convergenceAnalysis) {
    args.push('--convergence-analysis');
  }
  
  if (options.gradientAnalysis) {
    args.push('--gradient-analysis');
  }
  
  if (options.similarityMatrix) {
    args.push('--similarity-matrix');
  }
  
  // Add path filter option
  if (options.path) {
    args.push('--path', options.path);
  }
  
  // Add epsilon option
  if (options.epsilon !== undefined) {
    args.push('--epsilon', options.epsilon.toString());
  }
  
  // Add quiet option
  if (options.quiet) {
    args.push('--quiet');
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