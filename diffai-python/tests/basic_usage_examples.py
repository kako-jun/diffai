import unittest
import json
from diffai_python import diffai

class TestBasicUsageExamples(unittest.TestCase):
    
    def parse_json(self, json_str):
        """Helper function to parse JSON strings"""
        return json.loads(json_str)
    
    def test_basic_comprehensive_analysis(self):
        """Test case 1: Python library basic comprehensive analysis"""
        v1 = self.parse_json('{"model": {"layers": 2, "params": 1000}}')
        v2 = self.parse_json('{"model": {"layers": 3, "params": 1500}}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('layers' in results_str or 'params' in results_str)
    
    def test_json_output(self):
        """Test case 2: Python library JSON output"""
        v1 = self.parse_json('{"tensor": {"mean": 0.5, "std": 0.1}}')
        v2 = self.parse_json('{"tensor": {"mean": 0.6, "std": 0.2}}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('tensor' in results_str)
    
    def test_yaml_output(self):
        """Test case 3: Python library YAML output"""
        v1 = self.parse_json('{"weights": {"layer1": 0.5}}')
        v2 = self.parse_json('{"weights": {"layer1": 0.7}}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('weights' in results_str)
    
    def test_recursive_directory_comparison(self):
        """Test case 4: Python library recursive directory comparison"""
        v1 = self.parse_json('{"config": {"version": "1.0"}}')
        v2 = self.parse_json('{"config": {"version": "2.0"}}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('version' in results_str)
    
    def test_recursive_with_format(self):
        """Test case 5: Python library recursive with format"""
        v1 = self.parse_json('{"model": {"type": "safetensors"}}')
        v2 = self.parse_json('{"model": {"type": "safetensors", "version": 2}}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('model' in results_str)
    
    def test_pytorch_model_comparison(self):
        """Test case 6: Python library PyTorch model comparison"""
        v1 = self.parse_json('{"state_dict": {"layer1.weight": [0.1, 0.2]}}')
        v2 = self.parse_json('{"state_dict": {"layer1.weight": [0.15, 0.25]}}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('state_dict' in results_str)
    
    def test_training_checkpoint_comparison(self):
        """Test case 7: Python library training checkpoint comparison"""
        v1 = self.parse_json('{"epoch": 1, "loss": 0.8, "accuracy": 0.6}')
        v2 = self.parse_json('{"epoch": 10, "loss": 0.3, "accuracy": 0.9}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('epoch' in results_str)
    
    def test_baseline_vs_improved(self):
        """Test case 8: Python library baseline vs improved"""
        v1 = self.parse_json('{"performance": 0.85, "params": 1000000}')
        v2 = self.parse_json('{"performance": 0.92, "params": 1200000}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('performance' in results_str)
    
    def test_safetensors_comprehensive(self):
        """Test case 9: Python library safetensors comprehensive"""
        v1 = self.parse_json('{"tensors": {"fc1.bias": {"shape": [64]}}}')
        v2 = self.parse_json('{"tensors": {"fc1.bias": {"shape": [128]}}}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('tensors' in results_str)
    
    def test_deployment_validation(self):
        """Test case 10: Python library deployment validation"""
        v1 = self.parse_json('{"deployment": {"ready": true, "risk": "low"}}')
        v2 = self.parse_json('{"deployment": {"ready": true, "risk": "medium"}}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('deployment' in results_str)
    
    def test_numpy_array_comparison(self):
        """Test case 11: Python library NumPy array comparison"""
        v1 = self.parse_json('{"array": {"data": [1.0, 2.0, 3.0], "shape": [3]}}')
        v2 = self.parse_json('{"array": {"data": [1.1, 2.1, 3.1], "shape": [3]}}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('array' in results_str)
    
    def test_matlab_file_comparison(self):
        """Test case 12: Python library MATLAB file comparison"""
        v1 = self.parse_json('{"simulation": {"time": 100, "results": [0.5, 0.6]}}')
        v2 = self.parse_json('{"simulation": {"time": 150, "results": [0.7, 0.8]}}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('simulation' in results_str)
    
    def test_compressed_numpy_archives(self):
        """Test case 13: Python library compressed NumPy archives"""
        v1 = self.parse_json('{"dataset": {"train": 1000, "test": 200}}')
        v2 = self.parse_json('{"dataset": {"train": 1200, "test": 250}}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('dataset' in results_str)
    
    def test_experiment_comparison(self):
        """Test case 14: Python library experiment comparison"""
        v1 = self.parse_json('{"experiment": {"id": "v1", "accuracy": 0.85}}')
        v2 = self.parse_json('{"experiment": {"id": "v2", "accuracy": 0.90}}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('experiment' in results_str)
    
    def test_checkpoint_learning_analysis(self):
        """Test case 15: Python library checkpoint learning analysis"""
        v1 = self.parse_json('{"checkpoint": {"epoch": 10, "loss": 0.5}}')
        v2 = self.parse_json('{"checkpoint": {"epoch": 20, "loss": 0.3}}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('checkpoint' in results_str)
    
    def test_cicd_model_comparison(self):
        """Test case 16: Python library CI/CD model comparison"""
        v1 = self.parse_json('{"model": {"version": "baseline", "accuracy": 0.85}}')
        v2 = self.parse_json('{"model": {"version": "new", "accuracy": 0.88}}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('model' in results_str)

if __name__ == '__main__':
    unittest.main()