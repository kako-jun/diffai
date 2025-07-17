#!/usr/bin/env python3
"""
Unified test runner for diffai Python package
Runs all tests in the tests/ directory
"""

import os
import sys
import importlib.util
from pathlib import Path

TESTS_DIR = Path(__file__).parent / 'tests'


def run_all_tests():
    """Run all tests in the tests directory"""
    print('🧪 Running all diffai Python package tests...')
    print('=' * 60)
    
    # Check if tests directory exists
    if not TESTS_DIR.exists():
        print(f'❌ Tests directory not found: {TESTS_DIR}')
        sys.exit(1)
    
    # Find all test files
    test_files = list(TESTS_DIR.glob('test_*.py'))
    
    if not test_files:
        print('⚠️  No test files found in tests/ directory')
        sys.exit(1)
    
    print(f'Found {len(test_files)} test file(s):')
    for test_file in test_files:
        print(f'  - {test_file.name}')
    print('')
    
    total_passed = 0
    total_tests = 0
    all_passed = True
    
    # Run each test file
    for test_file in test_files:
        print(f'\\n📋 Running {test_file.name}...')
        print('-' * 40)
        
        try:
            # Import and run the test module
            spec = importlib.util.spec_from_file_location(test_file.stem, test_file)
            if spec and spec.loader:
                test_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(test_module)
                
                if hasattr(test_module, 'run_tests'):
                    result = test_module.run_tests()
                    # If run_tests returns a result object, use it
                    if isinstance(result, dict) and 'passed' in result:
                        total_passed += result['passed']
                        total_tests += result['total']
                        if result['passed'] < result['total']:
                            all_passed = False
                else:
                    print('⚠️  Test file does not have run_tests function')
                    all_passed = False
        except Exception as error:
            print(f'❌ Error running test file: {error}')
            all_passed = False
    
    # Final summary
    print('\\n' + '=' * 60)
    print('📊 FINAL RESULTS')
    print('=' * 60)
    
    if total_tests > 0:
        print(f'Total tests: {total_tests}')
        print(f'Passed: {total_passed}')
        print(f'Failed: {total_tests - total_passed}')
    
    if all_passed:
        print('🎉 All test suites passed!')
        sys.exit(0)
    else:
        print('❌ Some test suites failed')
        sys.exit(1)


# Handle uncaught errors
def handle_exception(exc_type, exc_value, exc_traceback):
    print(f'❌ Uncaught exception: {exc_value}')
    sys.exit(1)


def handle_unhandled_rejection(reason):
    print(f'❌ Unhandled rejection: {reason}')
    sys.exit(1)


if __name__ == '__main__':
    # Set up error handlers
    sys.excepthook = handle_exception
    
    try:
        run_all_tests()
    except Exception as error:
        print(f'❌ Test runner error: {error}')
        sys.exit(1)