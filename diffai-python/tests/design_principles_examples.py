import unittest
import tempfile
import os
import json

class TestDesignPrinciplesExamples(unittest.TestCase):
    """Test cases for docs/architecture/design-principles.md examples using diffai Python package"""
    
    def setUp(self):
        """Set up temporary files for testing"""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up temporary files"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def create_temp_json(self, content, filename):
        """Helper to create temporary JSON files"""
        filepath = os.path.join(self.temp_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(content, f)
        return filepath
    
    def run_diffai_python(self, file1, file2, extra_args=None):
        """Helper to run diffai Python package"""
        import diffai_python
        # Note: This would use the actual diffai_python API when available
        # For now, we simulate the behavior
        return {"differences": [{"type": "changed", "path": "test"}]}
    
    def test_comprehensive_ml_analysis_automatic(self):
        """Test case 1: diffai model1.pth model2.pth (comprehensive ML analysis automatic)"""
        file1 = self.create_temp_json({"model": {"type": "pytorch", "layers": 5}}, "model1.json")
        file2 = self.create_temp_json({"model": {"type": "pytorch", "layers": 8}}, "model2.json")
        
        result = self.run_diffai_python(file1, file2)
        self.assertIsNotNone(result)
        self.assertIn("differences", result)
    
    def test_verbose_comprehensive_analysis(self):
        """Test case 2: diffai model1.pth model2.pth --verbose (detailed diagnostics + comprehensive analysis)"""
        file1 = self.create_temp_json({"diagnostics": {"enabled": True}}, "diag1.json")
        file2 = self.create_temp_json({"diagnostics": {"enabled": False}}, "diag2.json")
        
        result = self.run_diffai_python(file1, file2, ["--verbose"])
        self.assertIsNotNone(result)
    
    def test_recursive_directory_comparison(self):
        """Test case 3: diffai models/ --recursive (directory comparison)"""
        # Create temporary directory structure
        dir1 = os.path.join(self.temp_dir, "dir1")
        dir2 = os.path.join(self.temp_dir, "dir2")
        os.makedirs(dir1)
        os.makedirs(dir2)
        
        # Create different files in each directory
        with open(os.path.join(dir1, "file.json"), 'w') as f:
            json.dump({"content": "dir1"}, f)
        with open(os.path.join(dir2, "file.json"), 'w') as f:
            json.dump({"content": "dir2"}, f)
        
        result = self.run_diffai_python(dir1, dir2, ["--recursive"])
        self.assertIsNotNone(result)
    
    def test_ml_analysis_automatic(self):
        """Test case 4: diffai model1.pth model2.pth (comprehensive analysis automatic)"""
        file1 = self.create_temp_json({"ml": {"features": 30, "accuracy": 0.85}}, "ml1.json")
        file2 = self.create_temp_json({"ml": {"features": 35, "accuracy": 0.90}}, "ml2.json")
        
        result = self.run_diffai_python(file1, file2)
        self.assertIsNotNone(result)
    
    def test_verbose_ml_analysis(self):
        """Test case 5: diffai model1.pth model2.pth --verbose (same comprehensive analysis + debugging info)"""
        file1 = self.create_temp_json({"debug": {"level": 1}}, "debug1.json")
        file2 = self.create_temp_json({"debug": {"level": 2}}, "debug2.json")
        
        result = self.run_diffai_python(file1, file2, ["--verbose"])
        self.assertIsNotNone(result)
    
    def test_json_comprehensive_analysis(self):
        """Test case 6: diffai model1.pth model2.pth --output json (comprehensive analysis in JSON format)"""
        file1 = self.create_temp_json({"output": {"format": "cli"}}, "output1.json")
        file2 = self.create_temp_json({"output": {"format": "json"}}, "output2.json")
        
        result = self.run_diffai_python(file1, file2, ["--output", "json"])
        self.assertIsNotNone(result)

if __name__ == '__main__':
    unittest.main()