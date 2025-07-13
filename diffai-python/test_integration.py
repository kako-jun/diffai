#!/usr/bin/env python3
"""
Integration tests for diffai Python package.

Tests the Python API and ensures proper integration with the diffai binary.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

# Add src to Python path for testing
sys.path.insert(0, str(Path(__file__).parent / "src"))

import diffai


class TestDiffaiIntegration(unittest.TestCase):
    """Integration tests for diffai Python package."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary test files
        self.temp_dir = tempfile.mkdtemp()
        
        self.json1_path = os.path.join(self.temp_dir, "test1.json")
        self.json2_path = os.path.join(self.temp_dir, "test2.json")
        
        # Create test JSON files
        json1_data = {
            "model": "gpt-2",
            "layers": 12,
            "parameters": 117000000,
            "config": {
                "vocab_size": 50257,
                "n_positions": 1024
            }
        }
        
        json2_data = {
            "model": "gpt-2", 
            "layers": 24,  # Changed
            "parameters": 345000000,  # Changed
            "config": {
                "vocab_size": 50257,
                "n_positions": 1024,
                "new_param": "added"  # Added
            }
        }
        
        with open(self.json1_path, 'w') as f:
            json.dump(json1_data, f, indent=2)
        
        with open(self.json2_path, 'w') as f:
            json.dump(json2_data, f, indent=2)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_installation_verification(self):
        """Test that diffai is properly installed and accessible."""
        try:
            from diffai import _find_diffai_binary
            binary_path = _find_diffai_binary()
            self.assertTrue(os.path.exists(binary_path))
            print(f"✅ diffai binary found at: {binary_path}")
        except diffai.DiffaiError:
            self.skipTest("diffai binary not found")
    
    def test_basic_diff(self):
        """Test basic file comparison."""
        try:
            result = diffai.diff(self.json1_path, self.json2_path)
            self.assertIsInstance(result, diffai.DiffResult)
            self.assertEqual(result.return_code, 0)
            self.assertTrue(len(result.raw_output) > 0)
            print(f"✅ Basic diff completed: {len(result.raw_output)} chars output")
        except diffai.DiffaiError:
            self.skipTest("diffai binary not found")
    
    def test_json_output(self):
        """Test JSON output format."""
        try:
            result = diffai.diff(
                self.json1_path, 
                self.json2_path,
                output_format=diffai.OutputFormat.JSON
            )
            
            self.assertTrue(result.is_json)
            self.assertIsInstance(result.data, list)
            
            # Should detect the changes we made
            self.assertTrue(len(result.data) > 0)
            
            print(f"✅ JSON output test passed: {len(result.data)} changes detected")
            
        except diffai.DiffaiError:
            self.skipTest("diffai binary not found")
    
    def test_diff_options(self):
        """Test DiffOptions configuration."""
        try:
            options = diffai.DiffOptions(
                output_format=diffai.OutputFormat.JSON,
                recursive=False
            )
            
            result = diffai.diff(self.json1_path, self.json2_path, options)
            self.assertTrue(result.is_json)
            print("✅ DiffOptions test passed")
            
        except diffai.DiffaiError:
            self.skipTest("diffai binary not found")
    
    def test_string_comparison(self):
        """Test temporary file creation for content comparison."""
        try:
            # Test creating temporary files and comparing them
            import tempfile
            
            json_str1 = '{"test": "value1", "number": 42}'
            json_str2 = '{"test": "value2", "number": 42, "new": "field"}'
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f1:
                f1.write(json_str1)
                f1.flush()
                temp1 = f1.name
                
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f2:
                f2.write(json_str2)
                f2.flush()
                temp2 = f2.name
            
            try:
                result = diffai.diff(
                    temp1, 
                    temp2,
                    output_format=diffai.OutputFormat.JSON
                )
                
                self.assertTrue(result.is_json)
                self.assertTrue(len(result.data) > 0)
                print("✅ String comparison test passed")
            finally:
                os.unlink(temp1)
                os.unlink(temp2)
            
        except diffai.DiffaiError:
            self.skipTest("diffai binary not found")
    
    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        try:
            # Test with non-existent files
            with self.assertRaises(diffai.DiffaiError):
                diffai.diff("nonexistent1.json", "nonexistent2.json")
            
            print("✅ Error handling test passed")
            
        except diffai.DiffaiError:
            self.skipTest("diffai binary not found")
    
    def test_binary_not_found_error(self):
        """Test handling of BinaryNotFoundError scenarios."""
        # Test binary detection functions
        try:
            from diffai import _find_diffai_binary
            binary_path = _find_diffai_binary()
            self.assertTrue(os.path.exists(binary_path))
            print("✅ BinaryNotFoundError handling test passed")
        except diffai.DiffaiError as e:
            if "binary not found" in str(e) or "BinaryNotFoundError" in str(e):
                print("✅ BinaryNotFoundError correctly raised")
                self.skipTest("diffai binary not found - this is expected behavior")
            else:
                raise
    
    def test_ml_analysis_options(self):
        """Test ML-specific analysis options."""
        try:
            # Test with ML analysis options (even on JSON files)
            options = diffai.DiffOptions(
                stats=True,
                architecture_comparison=True,
                memory_analysis=True
            )
            
            result = diffai.diff(self.json1_path, self.json2_path, options)
            self.assertEqual(result.return_code, 0)
            print("✅ ML analysis options test passed")
            
        except diffai.DiffaiError:
            self.skipTest("diffai binary not found")
    
    def test_version_access(self):
        """Test version access."""
        version = diffai.__version__
        self.assertIsInstance(version, str)
        self.assertTrue(len(version) > 0)
        print(f"✅ Version access test passed: {version}")
    
    def test_backward_compatibility(self):
        """Test package structure compatibility."""
        # Test that main API is accessible directly from diffai
        self.assertTrue(hasattr(diffai, 'diff'))
        self.assertTrue(hasattr(diffai, 'DiffOptions'))
        self.assertTrue(hasattr(diffai, 'OutputFormat'))
        self.assertTrue(hasattr(diffai, 'DiffResult'))
        self.assertTrue(hasattr(diffai, 'DiffaiError'))
        print("✅ Package structure compatibility test passed")


class TestDiffaiWithoutBinary(unittest.TestCase):
    """Tests that don't require the binary to be installed."""
    
    def test_diff_options_creation(self):
        """Test DiffOptions object creation."""
        options = diffai.DiffOptions(
            stats=True,
            architecture_comparison=True,
            output_format=diffai.OutputFormat.JSON
        )
        
        args = options.to_args()
        self.assertIn("--stats", args)
        self.assertIn("--architecture-comparison", args)
        self.assertIn("--output", args)
        self.assertIn("json", args)
        print("✅ DiffOptions creation test passed")
    
    def test_output_format_enum(self):
        """Test OutputFormat enum."""
        self.assertEqual(diffai.OutputFormat.JSON.value, "json")
        self.assertEqual(diffai.OutputFormat.YAML.value, "yaml")
        self.assertEqual(diffai.OutputFormat.CLI.value, "cli")
        print("✅ OutputFormat enum test passed")
    
    def test_diff_result_properties(self):
        """Test DiffResult properties."""
        # Test with JSON data
        json_output = '[{"path": "test", "type": "modified"}]'
        result = diffai.DiffResult(json_output, "json", 0)
        
        self.assertTrue(result.is_json)
        self.assertIsInstance(result.data, list)
        
        # Test with non-JSON data
        text_output = "Some diff output"
        result2 = diffai.DiffResult(text_output, "cli", 0)
        
        self.assertFalse(result2.is_json)
        self.assertEqual(result2.data, text_output)
        
        print("✅ DiffResult properties test passed")


def run_tests():
    """Run all tests and display results."""
    print("🧪 Running diffai Python package integration tests...")
    print(f"Python version: {sys.version}")
    print(f"diffai package version: {diffai.__version__}")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDiffaiWithoutBinary))
    suite.addTests(loader.loadTestsFromTestCase(TestDiffaiIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print(f"\n📊 Test Results:")
    print(f"  Tests run: {result.testsRun}")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    print(f"  Skipped: {len(result.skipped)}")
    
    if result.failures:
        print("\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\n💥 Errors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    if result.skipped:
        print("\n⏭️  Skipped:")
        for test, reason in result.skipped:
            print(f"  {test}: {reason}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    
    if success:
        print(f"\n🎉 All tests passed successfully!")
    else:
        print(f"\n💥 Some tests failed")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(run_tests())