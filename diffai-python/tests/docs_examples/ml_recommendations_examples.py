import unittest
import json
from diffai_python import diffai

class TestMLRecommendationsExamples(unittest.TestCase):
    
    def parse_json(self, json_str):
        """Helper function to parse JSON strings"""
        return json.loads(json_str)
    
    def test_deployment_recommendations(self):
        """Test case 1: Python library deployment recommendations"""
        v1 = self.parse_json('{"model": {"performance": 0.85, "memory": 512}}')
        v2 = self.parse_json('{"model": {"performance": 0.75, "memory": 1024}}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        # Check for performance and memory differences
        results_str = str(results)
        self.assertTrue('performance' in results_str or 'memory' in results_str)
    
    def test_json_recommendations_output(self):
        """Test case 2: Python library JSON recommendations processing"""
        v1 = self.parse_json('{"recommendations": {"enabled": true, "level": "high"}}')
        v2 = self.parse_json('{"recommendations": {"enabled": true, "level": "critical"}}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        # Check for recommendations level difference
        results_str = str(results)
        self.assertTrue('level' in results_str or 'recommendations' in results_str)
    
    def test_training_progress_recommendations(self):
        """Test case 3: Python library training progress recommendations"""
        v1 = self.parse_json('{"epoch": 10, "loss": 0.5, "accuracy": 0.80}')
        v2 = self.parse_json('{"epoch": 20, "loss": 0.3, "accuracy": 0.85}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        # Check for epoch, loss, and accuracy differences
        results_str = str(results)
        self.assertTrue(any(field in results_str for field in ['epoch', 'loss', 'accuracy']))

if __name__ == '__main__':
    unittest.main()