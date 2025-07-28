module.exports = {
  testEnvironment: 'node',
  testMatch: [
    '**/__tests__/**/*.js',
    '**/?(*.)+(spec|test).js'
  ],
  testTimeout: 60000, // 60 seconds for potential long-running AI/ML tests
  setupFilesAfterEnv: ['<rootDir>/tests/setup.js'],
  collectCoverageFrom: [
    'src/**/*.{js,ts}',
    'index.js',
    '!**/node_modules/**',
    '!**/tests/**'
  ],
  coverageReporters: ['text', 'lcov', 'html'],
  verbose: true,
  bail: false, // Continue running tests even if some fail
  errorOnDeprecated: true,
  
  // Handle async tests properly
  testRunner: 'jest-circus/runner',
  
  // Transform settings for modern JavaScript
  transform: {
    '^.+\\.js$': 'babel-jest'
  },
  
  // Module settings
  moduleFileExtensions: ['js', 'json', 'node'],
  
  // Globals for tests
  globals: {
    'NODE_ENV': 'test'
  },
  
  // AI/ML specific test settings
  maxWorkers: 1, // Run tests serially for AI/ML data processing
  
  // Handle large datasets in AI/ML tests
  setupFiles: ['<rootDir>/tests/ml-setup.js']
};