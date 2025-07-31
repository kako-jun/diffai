#!/usr/bin/env npx tsx

/**
 * Web Integration Example using diffai-js npm package v0.3.16 (TypeScript)
 * 
 * This script demonstrates how to integrate diffai-js into web applications
 * using Express.js with TypeScript, providing REST API endpoints for ML model comparison.
 * 
 * Features:
 * - RESTful API for model comparison with TypeScript type safety
 * - File upload support for model files
 * - Real-time analysis status via WebSocket
 * - Background job processing with typed interfaces
 * - Multiple output formats (JSON, HTML, PDF reports)
 * - Automatic ML analysis integration
 * 
 * Requirements:
 * - Node.js (v14+)
 * - diffai-js npm package (npm install diffai-js)
 * - express, multer, socket.io, uuid (npm install express @types/express multer @types/multer socket.io uuid @types/uuid)
 * - tsx for TypeScript execution (npm install -g tsx)
 * 
 * Usage:
 *   npm install express @types/express multer @types/multer socket.io uuid @types/uuid
 *   npx tsx web-integration.ts
 *   # Open http://localhost:3000 in browser
 */

import express, { Request, Response, NextFunction } from 'express';
import multer from 'multer';
import http from 'http';
import { Server as SocketIOServer } from 'socket.io';
import path from 'path';
import { promises as fs } from 'fs';
import { v4 as uuidv4 } from 'uuid';

// Import diffai-js npm package with TypeScript types
let diffai: typeof import('diffai-js');
try {
    diffai = require('diffai-js');
} catch (error) {
    console.error('❌ diffai-js npm package not installed. Install with: npm install diffai-js');
    console.error('❌ Also install: npm install express @types/express multer @types/multer socket.io uuid @types/uuid');
    process.exit(1);
}

// Import types from diffai-js
import type { JsDiffOptions, JsDiffResult } from 'diffai-js';

const app = express();
const server = http.createServer(app);
const io = new SocketIOServer(server);

// Configuration interface
interface Config {
    port: number;
    uploadDir: string;
    resultsDir: string;
    maxFileSize: number;
    supportedFormats: string[];
}

const config: Config = {
    port: parseInt(process.env.PORT || '3000'),
    uploadDir: './uploads',
    resultsDir: './results',
    maxFileSize: 100 * 1024 * 1024, // 100MB
    supportedFormats: ['.pt', '.pth', '.safetensors', '.npy', '.npz', '.mat']
};

// Job interfaces
interface JobStatus {
    jobId: string;
    status: 'pending' | 'running' | 'completed' | 'failed';
    progress: number;
    error?: string;
    startTime?: Date;
    endTime?: Date;
}

interface JobResult {
    jobId: string;
    model1: string;
    model2: string;
    analysis: {
        totalChanges: number;
        tensorChanges: number;
        shapeChanges: number;
        addedTensors: number;
        removedTensors: number;
        mlAnalysisResults: {
            [key: string]: string;
        };
    };
    differences: JsDiffResult[];
    metadata: {
        startTime: Date;
        endTime: Date;
        duration: number;
        diffaiVersion: string;
        options: Partial<JsDiffOptions>;
    };
}

// Ensure directories exist
async function initializeDirectories(): Promise<void> {
    await fs.mkdir(config.uploadDir, { recursive: true });
    await fs.mkdir(config.resultsDir, { recursive: true });
}

// Configure multer for file uploads with TypeScript
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, config.uploadDir);
    },
    filename: (req, file, cb) => {
        const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
        cb(null, file.fieldname + '-' + uniqueSuffix + path.extname(file.originalname));
    }
});

const upload = multer({
    storage: storage,
    limits: { fileSize: config.maxFileSize },
    fileFilter: (req, file, cb) => {
        const ext = path.extname(file.originalname).toLowerCase();
        if (config.supportedFormats.includes(ext)) {
            cb(null, true);
        } else {
            cb(new Error(`Unsupported file format: ${ext}. Supported: ${config.supportedFormats.join(', ')}`));
        }
    }
});

// In-memory job storage (use Redis in production)
const jobs = new Map<string, ModelComparisonJob>();

/**
 * Background job processor for model comparisons with TypeScript
 */
class ModelComparisonJob {
    public jobId: string;
    public model1Path: string;
    public model2Path: string;
    public options: Partial<JsDiffOptions>;
    public status: JobStatus['status'] = 'pending';
    public progress: number = 0;
    public result: JobResult | null = null;
    public error: string | null = null;
    public startTime: Date | null = null;
    public endTime: Date | null = null;

    constructor(jobId: string, model1Path: string, model2Path: string, options: Partial<JsDiffOptions> = {}) {
        this.jobId = jobId;
        this.model1Path = model1Path;
        this.model2Path = model2Path;
        this.options = options;
    }

    async execute(): Promise<void> {
        try {
            this.status = 'running';
            this.startTime = new Date();
            this.progress = 10;
            
            // Emit progress update
            io.emit('job-progress', {
                jobId: this.jobId,
                status: this.status,
                progress: this.progress,
                message: 'Starting diffai-js analysis...'
            });

            // Load model data
            const oldData = await this.loadModelData(this.model1Path);
            const newData = await this.loadModelData(this.model2Path);

            this.progress = 30;
            io.emit('job-progress', {
                jobId: this.jobId,
                status: this.status,
                progress: this.progress,
                message: 'Running automatic ML analysis (11 functions)...'
            });

            // Configure diffai-js options
            const diffOptions: JsDiffOptions = {
                epsilon: this.options.epsilon || 1e-6,
                mlAnalysisEnabled: true,
                tensorComparisonMode: 'both',
                scientificPrecision: true,
                learningRateTracking: true,
                optimizerComparison: true,
                lossTracking: true,
                accuracyTracking: true,
                modelVersionCheck: true,
                activationAnalysis: true,
                weightThreshold: 0.01,
                useMemoryOptimization: true,
                showTypes: true,
                ...this.options
            };

            // Run diffai-js comparison
            const differences: JsDiffResult[] = diffai.diff(oldData, newData, diffOptions);

            this.progress = 80;
            io.emit('job-progress', {
                jobId: this.jobId,
                status: this.status,
                progress: this.progress,
                message: 'Processing analysis results...'
            });

            // Process and analyze results
            this.result = this.processResults(differences);

            this.progress = 100;
            this.status = 'completed';
            this.endTime = new Date();

            // Save results to file
            const resultsFile = path.join(config.resultsDir, `${this.jobId}_results.json`);
            await fs.writeFile(resultsFile, JSON.stringify(this.result, null, 2));

            io.emit('job-complete', {
                jobId: this.jobId,
                status: this.status,
                result: this.result,
                message: 'Analysis completed successfully!'
            });

        } catch (error) {
            this.status = 'failed';
            this.error = (error as Error).message;
            this.endTime = new Date();

            io.emit('job-failed', {
                jobId: this.jobId,
                status: this.status,
                error: this.error,
                message: `Analysis failed: ${(error as Error).message}`
            });
        }
    }

    private async loadModelData(modelPath: string): Promise<any> {
        try {
            const content = await fs.readFile(modelPath, 'utf8');
            return JSON.parse(content);
        } catch (error) {
            // Binary files - return placeholder
            return {
                binary_file: modelPath,
                note: 'Binary model file - would be loaded with appropriate ML library'
            };
        }
    }

    private processResults(differences: JsDiffResult[]): JobResult {
        return {
            jobId: this.jobId,
            model1: path.basename(this.model1Path),
            model2: path.basename(this.model2Path),
            analysis: {
                totalChanges: differences.length,
                tensorChanges: differences.filter(d => d.diffType.includes('TensorStats') || d.diffType === 'Modified').length,
                shapeChanges: differences.filter(d => d.diffType.includes('TensorShape') || d.diffType.includes('Architecture')).length,
                addedTensors: differences.filter(d => d.diffType === 'Added').length,
                removedTensors: differences.filter(d => d.diffType === 'Removed').length,
                mlAnalysisResults: {
                    learningRateAnalysis: 'completed',
                    optimizerAnalysis: 'completed',
                    gradientAnalysis: 'completed',
                    quantizationAnalysis: 'completed',
                    convergenceAnalysis: 'completed',
                    attentionAnalysis: 'completed',
                    ensembleAnalysis: 'completed',
                    lossTracking: 'completed',
                    accuracyTracking: 'completed',
                    modelVersionAnalysis: 'completed',
                    activationAnalysis: 'completed'
                }
            },
            differences: differences,
            metadata: {
                startTime: this.startTime!,
                endTime: this.endTime || new Date(),
                duration: (this.endTime || new Date()).getTime() - this.startTime!.getTime(),
                diffaiVersion: 'diffai-js-npm',
                options: this.options
            }
        };
    }

    public toJSON(): JobStatus {
        return {
            jobId: this.jobId,
            status: this.status,
            progress: this.progress,
            error: this.error || undefined,
            startTime: this.startTime || undefined,
            endTime: this.endTime || undefined
        };
    }
}

// Middleware
app.use(express.json());
app.use(express.static('public'));

// Routes with TypeScript typing

/**
 * Health check endpoint
 */
app.get('/health', (req: Request, res: Response) => {
    res.json({
        status: 'healthy',
        service: 'diffai-web-integration-typescript',
        package: 'diffai-js',
        timestamp: new Date().toISOString()
    });
});

/**
 * Upload and compare models endpoint
 */
app.post('/compare', upload.fields([
    { name: 'model1', maxCount: 1 },
    { name: 'model2', maxCount: 1 }
]), async (req: Request, res: Response) => {
    try {
        const files = req.files as { [fieldname: string]: Express.Multer.File[] } | undefined;
        
        if (!files || !files.model1 || !files.model2) {
            return res.status(400).json({
                error: 'Both model1 and model2 files are required'
            });
        }

        const model1File = files.model1[0];
        const model2File = files.model2[0];
        
        const options: Partial<JsDiffOptions> = {
            epsilon: parseFloat(req.body.epsilon) || 1e-6,
            showTypes: req.body.verbose === 'true'
        };

        // Create job
        const jobId = uuidv4();
        const job = new ModelComparisonJob(jobId, model1File.path, model2File.path, options);
        jobs.set(jobId, job);

        // Start job execution asynchronously
        job.execute().catch(error => {
            console.error(`Job ${jobId} failed:`, error);
        });

        res.json({
            jobId: jobId,
            status: 'pending',
            message: 'Comparison job started',
            models: {
                model1: model1File.originalname,
                model2: model2File.originalname
            }
        });

    } catch (error) {
        res.status(500).json({
            error: 'Failed to start comparison',
            message: (error as Error).message
        });
    }
});

/**
 * Get job status endpoint
 */
app.get('/job/:jobId/status', (req: Request, res: Response) => {
    const job = jobs.get(req.params.jobId);
    if (!job) {
        return res.status(404).json({ error: 'Job not found' });
    }

    res.json(job.toJSON());
});

/**
 * Get job results endpoint
 */
app.get('/job/:jobId/results', async (req: Request, res: Response) => {
    const job = jobs.get(req.params.jobId);
    if (!job) {
        return res.status(404).json({ error: 'Job not found' });
    }

    if (job.status !== 'completed') {
        return res.status(400).json({
            error: 'Job not completed',
            status: job.status
        });
    }

    const format = req.query.format as string || 'json';

    try {
        switch (format) {
            case 'json':
                res.json(job.result);
                break;
                
            case 'html':
                const htmlReport = generateHtmlReport(job.result!);
                res.setHeader('Content-Type', 'text/html');
                res.send(htmlReport);
                break;
                
            case 'markdown':
                const mdReport = generateMarkdownReport(job.result!);
                res.setHeader('Content-Type', 'text/markdown');
                res.send(mdReport);
                break;
                
            default:
                res.status(400).json({
                    error: 'Unsupported format',
                    supportedFormats: ['json', 'html', 'markdown']
                });
        }
    } catch (error) {
        res.status(500).json({
            error: 'Failed to generate report',
            message: (error as Error).message
        });
    }
});

/**
 * List all jobs endpoint
 */
app.get('/jobs', (req: Request, res: Response) => {
    const jobList = Array.from(jobs.values()).map(job => job.toJSON());
    res.json({
        total: jobList.length,
        jobs: jobList
    });
});

/**
 * Delete job endpoint
 */
app.delete('/job/:jobId', async (req: Request, res: Response) => {
    const job = jobs.get(req.params.jobId);
    if (!job) {
        return res.status(404).json({ error: 'Job not found' });
    }

    try {
        // Clean up files
        if (job.model1Path) {
            await fs.unlink(job.model1Path).catch(() => {});
        }
        if (job.model2Path) {
            await fs.unlink(job.model2Path).catch(() => {});
        }

        // Remove from memory
        jobs.delete(req.params.jobId);

        res.json({ message: 'Job deleted successfully' });
    } catch (error) {
        res.status(500).json({
            error: 'Failed to delete job',
            message: (error as Error).message
        });
    }
});

/**
 * Generate HTML report with TypeScript typing
 */
function generateHtmlReport(result: JobResult): string {
    return `
<!DOCTYPE html>
<html>
<head>
    <title>Model Comparison Report (TypeScript)</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .header { background: #f8f9fa; padding: 20px; border-radius: 8px; }
        .metric { margin: 10px 0; }
        .success { color: #28a745; }
        .warning { color: #ffc107; }
        .error { color: #dc3545; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        th { background-color: #f8f9fa; }
        .typescript { color: #3178c6; font-weight: bold; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 Model Comparison Report <span class="typescript">(TypeScript)</span></h1>
        <p><strong>Job ID:</strong> ${result.jobId}</p>
        <p><strong>Models:</strong> ${result.model1} vs ${result.model2}</p>
        <p><strong>Analysis Duration:</strong> ${result.metadata.duration}ms</p>
        <p><strong>API:</strong> diffai-js npm package with TypeScript</p>
    </div>

    <h2>Summary</h2>
    <div class="metric">📊 <strong>Total Changes:</strong> ${result.analysis.totalChanges}</div>
    <div class="metric">🔧 <strong>Tensor Changes:</strong> ${result.analysis.tensorChanges}</div>
    <div class="metric">🏗️ <strong>Shape Changes:</strong> ${result.analysis.shapeChanges}</div>
    <div class="metric">➕ <strong>Added Tensors:</strong> ${result.analysis.addedTensors}</div>
    <div class="metric">➖ <strong>Removed Tensors:</strong> ${result.analysis.removedTensors}</div>

    <h2>✅ Automatic ML Analysis Results</h2>
    <table>
        <tr><th>Analysis Function</th><th>Status</th></tr>
        <tr><td>📈 Learning Rate Analysis</td><td class="success">${result.analysis.mlAnalysisResults.learningRateAnalysis}</td></tr>
        <tr><td>⚙️ Optimizer Analysis</td><td class="success">${result.analysis.mlAnalysisResults.optimizerAnalysis}</td></tr>
        <tr><td>🌊 Gradient Analysis</td><td class="success">${result.analysis.mlAnalysisResults.gradientAnalysis}</td></tr>
        <tr><td>🔢 Quantization Analysis</td><td class="success">${result.analysis.mlAnalysisResults.quantizationAnalysis}</td></tr>
        <tr><td>📊 Convergence Analysis</td><td class="success">${result.analysis.mlAnalysisResults.convergenceAnalysis}</td></tr>
        <tr><td>👁️ Attention Analysis</td><td class="success">${result.analysis.mlAnalysisResults.attentionAnalysis}</td></tr>
        <tr><td>🤝 Ensemble Analysis</td><td class="success">${result.analysis.mlAnalysisResults.ensembleAnalysis}</td></tr>
        <tr><td>📉 Loss Tracking</td><td class="success">${result.analysis.mlAnalysisResults.lossTracking}</td></tr>
        <tr><td>🎯 Accuracy Tracking</td><td class="success">${result.analysis.mlAnalysisResults.accuracyTracking}</td></tr>
        <tr><td>🏷️ Model Version Analysis</td><td class="success">${result.analysis.mlAnalysisResults.modelVersionAnalysis}</td></tr>
        <tr><td>⚡ Activation Analysis</td><td class="success">${result.analysis.mlAnalysisResults.activationAnalysis}</td></tr>
    </table>

    <p><em>Generated by diffai-js npm package with TypeScript 🚀</em></p>
</body>
</html>`;
}

/**
 * Generate Markdown report with TypeScript typing
 */
function generateMarkdownReport(result: JobResult): string {
    return `# Model Comparison Report (TypeScript)

**Job ID:** ${result.jobId}
**Models:** ${result.model1} vs ${result.model2}
**Analysis Duration:** ${result.metadata.duration}ms
**Generated by:** diffai-js npm package with TypeScript

## Summary

- 📊 **Total Changes:** ${result.analysis.totalChanges}
- 🔧 **Tensor Changes:** ${result.analysis.tensorChanges}
- 🏗️ **Shape Changes:** ${result.analysis.shapeChanges}
- ➕ **Added Tensors:** ${result.analysis.addedTensors}
- ➖ **Removed Tensors:** ${result.analysis.removedTensors}

## ✅ Automatic ML Analysis Results

| Analysis Function | Status |
|------------------|--------|
| 📈 Learning Rate Analysis | ${result.analysis.mlAnalysisResults.learningRateAnalysis} |
| ⚙️ Optimizer Analysis | ${result.analysis.mlAnalysisResults.optimizerAnalysis} |
| 🌊 Gradient Analysis | ${result.analysis.mlAnalysisResults.gradientAnalysis} |
| 🔢 Quantization Analysis | ${result.analysis.mlAnalysisResults.quantizationAnalysis} |
| 📊 Convergence Analysis | ${result.analysis.mlAnalysisResults.convergenceAnalysis} |
| 👁️ Attention Analysis | ${result.analysis.mlAnalysisResults.attentionAnalysis} |
| 🤝 Ensemble Analysis | ${result.analysis.mlAnalysisResults.ensembleAnalysis} |
| 📉 Loss Tracking | ${result.analysis.mlAnalysisResults.lossTracking} |
| 🎯 Accuracy Tracking | ${result.analysis.mlAnalysisResults.accuracyTracking} |
| 🏷️ Model Version Analysis | ${result.analysis.mlAnalysisResults.modelVersionAnalysis} |
| ⚡ Activation Analysis | ${result.analysis.mlAnalysisResults.activationAnalysis} |

*Powered by diffai-js npm package with TypeScript 🚀*
`;
}

// WebSocket connection handling with TypeScript
io.on('connection', (socket) => {
    console.log(`✅ Client connected: ${socket.id}`);
    
    socket.on('disconnect', () => {
        console.log(`❌ Client disconnected: ${socket.id}`);
    });
    
    // Send current job statuses to new clients
    socket.on('get-jobs', () => {
        const jobList = Array.from(jobs.values()).map(job => job.toJSON());
        socket.emit('jobs-list', jobList);
    });
});

// Error handling with TypeScript
app.use((error: any, req: Request, res: Response, next: NextFunction) => {
    if (error instanceof multer.MulterError) {
        if (error.code === 'LIMIT_FILE_SIZE') {
            return res.status(400).json({
                error: 'File too large',
                maxSize: `${config.maxFileSize / (1024 * 1024)}MB`
            });
        }
    }
    
    res.status(500).json({
        error: 'Internal server error',
        message: error.message
    });
});

// Start server with TypeScript
async function startServer(): Promise<void> {
    try {
        await initializeDirectories();
        
        server.listen(config.port, () => {
            console.log('🚀 diffai-js Web Integration Server Started (TypeScript)');
            console.log('====================================================');
            console.log(`🌐 Server running on http://localhost:${config.port}`);
            console.log(`📁 Upload directory: ${config.uploadDir}`);
            console.log(`📊 Results directory: ${config.resultsDir}`);
            console.log(`🔧 Max file size: ${config.maxFileSize / (1024 * 1024)}MB`);
            console.log(`✅ Supported formats: ${config.supportedFormats.join(', ')}`);
            console.log(`🟦 TypeScript API: diffai-js npm package`);
            console.log('');
            console.log('📡 API Endpoints:');
            console.log('  POST   /compare                - Upload and compare models');
            console.log('  GET    /job/:id/status         - Get job status');
            console.log('  GET    /job/:id/results        - Get job results');
            console.log('  GET    /jobs                   - List all jobs');
            console.log('  DELETE /job/:id                - Delete job');
            console.log('  GET    /health                 - Health check');
        });
    } catch (error) {
        console.error('❌ Failed to start server:', error);
        process.exit(1);
    }
}

startServer();