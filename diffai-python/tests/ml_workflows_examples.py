import unittest
import json
from diffai_python import diffai

class TestMLWorkflowsExamples(unittest.TestCase):
    
    def parse_json(self, json_str):
        """Helper function to parse JSON strings"""
        return json.loads(json_str)
    
    def test_model_development_improvement(self):
        """Test case 1: Python library model development improvement"""
        v1 = self.parse_json('{"architecture": "resnet18", "layers": 18, "parameters": 11000000}')
        v2 = self.parse_json('{"architecture": "resnet34", "layers": 34, "parameters": 21000000}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('architecture' in results_str)
    
    def test_finetuning_comparison(self):
        """Test case 2: Python library finetuning comparison"""
        v1 = self.parse_json('{"model": {"pretrained": true, "weights": {"classifier": [0.0, 0.0]}}}')
        v2 = self.parse_json('{"model": {"pretrained": false, "weights": {"classifier": [0.8, 0.9]}}}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('classifier' in results_str)
    
    def test_experiment_results_comparison(self):
        """Test case 3: Python library experiment results comparison"""
        v1 = self.parse_json('{"experiment": {"id": "001", "accuracy": 0.85, "loss": 0.3}}')
        v2 = self.parse_json('{"experiment": {"id": "002", "accuracy": 0.88, "loss": 0.25}}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('experiment' in results_str)
    
    def test_hyperparameter_differences(self):
        """Test case 4: Python library hyperparameter differences"""
        v1 = self.parse_json('{"config": {"learning_rate": 0.01, "batch_size": 32, "epochs": 100}}')
        v2 = self.parse_json('{"config": {"learning_rate": 0.001, "batch_size": 64, "epochs": 150}}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('learning_rate' in results_str)
    
    def test_quantization_comparison(self):
        """Test case 5: Python library quantization comparison"""
        v1 = self.parse_json('{"model": {"precision": "fp32", "size_mb": 100, "weights": {"layer1": [0.123456]}}}')
        v2 = self.parse_json('{"model": {"precision": "int8", "size_mb": 25, "weights": {"layer1": [0.125]}}}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('precision' in results_str)
    
    def test_pruning_effects(self):
        """Test case 6: Python library pruning effects"""
        v1 = self.parse_json('{"model": {"parameters": 1000000, "layers": {"conv1": {"weights": [0.1, 0.2, 0.3]}, "conv2": {"weights": [0.4, 0.5, 0.6]}}}}')
        v2 = self.parse_json('{"model": {"parameters": 500000, "layers": {"conv1": {"weights": [0.1, 0.0, 0.3]}, "conv2": {"weights": [0.0, 0.5, 0.0]}}}}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('parameters' in results_str)
    
    def test_workflow_comparison(self):
        """Test case 7: Python library workflow comparison"""
        v1 = self.parse_json('{"workflow": {"baseline": true, "results": {"accuracy": 0.85, "f1": 0.82}}}')
        v2 = self.parse_json('{"workflow": {"baseline": false, "results": {"accuracy": 0.90, "f1": 0.88}}}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('workflow' in results_str)

if __name__ == '__main__':
    unittest.main()