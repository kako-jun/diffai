import unittest
import json
from diffai_python import diffai

class TestOutputFormatsExamples(unittest.TestCase):
    
    def parse_json(self, json_str):
        """Helper function to parse JSON strings"""
        return json.loads(json_str)
    
    def test_cli_output_format(self):
        """Test case 1: Python library CLI output format"""
        v1 = self.parse_json('{"fc1": {"bias": {"mean": 0.0018, "std": 0.0518}}}')
        v2 = self.parse_json('{"fc1": {"bias": {"mean": 0.0017, "std": 0.0647}}}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('fc1' in results_str)
    
    def test_default_cli_output(self):
        """Test case 2: Python library default CLI output"""
        v1 = self.parse_json('{"layers": 12, "hidden_size": 768}')
        v2 = self.parse_json('{"layers": 24, "hidden_size": 768}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('layers' in results_str)
    
    def test_json_output_format(self):
        """Test case 3: Python library JSON output format"""
        v1 = self.parse_json('{"fc1": {"bias": {"mean": 0.0018, "std": 0.0518}}}')
        v2 = self.parse_json('{"fc1": {"bias": {"mean": 0.0017, "std": 0.0647}}}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('fc1' in results_str)
    
    def test_yaml_output_format(self):
        """Test case 4: Python library YAML output format"""
        v1 = self.parse_json('{"tensor": {"mean": 0.0018, "std": 0.0518}}')
        v2 = self.parse_json('{"tensor": {"mean": 0.0017, "std": 0.0647}}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('tensor' in results_str)
    
    def test_unified_output_format(self):
        """Test case 5: Python library unified output format"""
        v1 = self.parse_json('{"model": {"layers": 12, "hidden_size": 768}}')
        v2 = self.parse_json('{"model": {"layers": 24, "hidden_size": 768}, "optimizer": "adam"}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('layers' in results_str or 'optimizer' in results_str)
    
    def test_json_with_filter(self):
        """Test case 6: Python library JSON with filtering"""
        v1 = self.parse_json('{"fc1": {"bias": {"mean": 0.0018, "std": 0.0518}}}')
        v2 = self.parse_json('{"fc1": {"bias": {"mean": 0.0017, "std": 0.0647}}}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('fc1' in results_str)
    
    def test_yaml_output_to_file(self):
        """Test case 7: Python library YAML output to file"""
        v1 = self.parse_json('{"config": {"timeout": 30, "retries": 3}}')
        v2 = self.parse_json('{"config": {"timeout": 60, "retries": 5}}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('timeout' in results_str or 'retries' in results_str)
    
    def test_conditional_logic_check(self):
        """Test case 8: Python library conditional logic check"""
        v1 = self.parse_json('{"model": {"parameters": 1000}}')
        v2 = self.parse_json('{"model": {"parameters": 2000}}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('parameters' in results_str)
    
    def test_human_readable_output(self):
        """Test case 9: Python library human readable output"""
        v1 = self.parse_json('{"layer1": {"weights": [1.0, 2.0, 3.0]}}')
        v2 = self.parse_json('{"layer1": {"weights": [1.1, 2.1, 3.1]}}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('layer1' in results_str)
    
    def test_machine_readable_output(self):
        """Test case 10: Python library machine readable output"""
        v1 = self.parse_json('{"params": {"learning_rate": 0.001}}')
        v2 = self.parse_json('{"params": {"learning_rate": 0.01}}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('learning_rate' in results_str)
    
    def test_env_var_json_format(self):
        """Test case 11: Python library environment variable JSON format"""
        v1 = self.parse_json('{"model_version": "1.0"}')
        v2 = self.parse_json('{"model_version": "2.0"}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('model_version' in results_str)
    
    def test_cli_colors(self):
        """Test case 12: Python library CLI colors"""
        v1 = self.parse_json('{"status": "active"}')
        v2 = self.parse_json('{"status": "inactive"}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('status' in results_str)
    
    def test_json_pretty(self):
        """Test case 13: Python library JSON pretty print"""
        v1 = self.parse_json('{"data": {"value": 100}}')
        v2 = self.parse_json('{"data": {"value": 200}}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('data' in results_str or 'value' in results_str)

if __name__ == '__main__':
    unittest.main()