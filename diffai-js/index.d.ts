/**
 * TypeScript definitions for diffai-js - UNIFIED API DESIGN  
 * AI/ML model comparison with tensor and statistical analysis
 */

export interface DiffOptions {
  /** Tolerance for float comparisons (default: 1e-9) */
  epsilon?: number;
  
  /** Key to use for array element identification */
  arrayIdKey?: string;
  
  /** Ignore keys matching regex */
  ignoreKeysRegex?: string;
  
  /** Filter differences by path */
  pathFilter?: string;
  
  /** Output format */
  outputFormat?: 'diffai' | 'json' | 'yaml' | 'unified';
  
  /** Include unchanged values in output (default: false) */
  showUnchanged?: boolean;
  
  /** Include type information in output (default: false) */
  showTypes?: boolean;
  
  /** Enable memory-efficient processing (default: false) */
  useMemoryOptimization?: boolean;
  
  /** Batch size for memory optimization (default: 1000) */
  batchSize?: number;
  
  /** diffai-specific ML options */
  diffaiOptions?: DiffaiSpecificOptions;
}

export interface DiffaiSpecificOptions {
  /** Enable ML-specific analysis (default: true) */
  mlAnalysisEnabled?: boolean;
  
  /** Tensor comparison mode */
  tensorComparisonMode?: 'element-wise' | 'statistical' | 'structural';
  
  /** Expected model format for optimized parsing */
  modelFormat?: 'pytorch' | 'safetensors' | 'numpy' | 'matlab' | 'auto';
  
  /** Use scientific notation for numeric output (default: false) */
  scientificPrecision?: boolean;
  
  /** Minimum weight change to report (default: 1e-6) */
  weightThreshold?: number;
  
  /** Analyze gradient-related tensors specially (default: false) */
  gradientAnalysis?: boolean;
  
  /** Include statistical summaries of tensor changes (default: false) */
  statisticalSummary?: boolean;
}

export type DiffResultType = 
  | 'added'
  | 'removed'
  | 'modified'
  | 'typeChanged'
  | 'tensorShapeChanged'
  | 'tensorDataChanged'
  | 'modelArchitectureChanged'
  | 'weightSignificantChange'
  | 'activationFunctionChanged'
  | 'learningRateChanged'
  | 'optimizerChanged'
  | 'lossChange'
  | 'accuracyChange'
  | 'modelVersionChanged';

export interface TensorStatistics {
  meanChange: number;
  stdDev: number;
  maxChange: number;
  minChange: number;
  changedElements: number;
  totalElements: number;
}

export interface DiffResult {
  /** Type of difference */
  type: DiffResultType;
  
  /** Path to the changed element */
  path: string;
  
  /** Old value (for modified/removed) */
  oldValue?: any;
  
  /** New value (for modified/added) */
  newValue?: any;
  
  /** Value (for removed) */
  value?: any;
  
  /** Old type name (for typeChanged) */
  oldType?: string;
  
  /** New type name (for typeChanged) */
  newType?: string;
  
  /** Old tensor shape (for tensorShapeChanged) */
  oldShape?: number[];
  
  /** New tensor shape (for tensorShapeChanged) */
  newShape?: number[];
  
  /** Change magnitude (for weightSignificantChange) */
  magnitude?: number;
  
  /** Statistical information (for ML-specific changes) */
  statistics?: TensorStatistics;
}

/**
 * Compare two AI/ML model structures or tensors with ML-specific analysis
 * 
 * This is the unified entry point for all diffai functionality.
 * Users should parse model files/data themselves and call this function.
 * 
 * @param old - Old data structure (parsed model data, tensors, etc.)
 * @param newData - New data structure (parsed model data, tensors, etc.)
 * @param options - Optional configuration
 * @returns Promise resolving to array of differences
 * 
 * @example
 * ```typescript
 * import { diff, DiffOptions } from 'diffai-js';
 * import * as fs from 'fs';
 * 
 * // Example: Compare model metadata (users load models themselves)
 * const oldModel = {
 *   architecture: { layers: 12, hiddenSize: 768 },
 *   training: { learningRate: 0.001, optimizer: 'adam' },
 *   weights: {} // tensor data loaded by user
 * };
 * 
 * const newModel = {
 *   architecture: { layers: 24, hiddenSize: 1024 },
 *   training: { learningRate: 0.0005, optimizer: 'adamw' },
 *   weights: {} // tensor data loaded by user
 * };
 * 
 * const options: DiffOptions = {
 *   diffaiOptions: {
 *     mlAnalysisEnabled: true,
 *     tensorComparisonMode: 'statistical',
 *     scientificPrecision: true
 *   },
 *   epsilon: 1e-6,
 *   showTypes: true
 * };
 * 
 * const results = await diff(oldModel, newModel, options);
 * console.log('ML differences found:', results.length);
 * 
 * // Filter for significant weight changes
 * const weightChanges = results.filter(r => r.type === 'weightSignificantChange');
 * console.log('Significant weight changes:', weightChanges.length);
 * ```
 */
export function diff(old: any, newData: any, options?: DiffOptions): Promise<DiffResult[]>;

export default diff;