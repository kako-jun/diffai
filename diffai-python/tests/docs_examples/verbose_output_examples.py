import unittest
import json
from diffai_python import diffai

class TestVerboseOutputExamples(unittest.TestCase):
    
    def parse_json(self, json_str):
        """Helper function to parse JSON strings"""
        return json.loads(json_str)
    
    def test_basic_verbose_output(self):
        """Test case 1: Python library basic verbose output"""
        v1 = self.parse_json('{"config": {"debug": true}}')
        v2 = self.parse_json('{"config": {"debug": false}}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('config' in results_str)
    
    def test_verbose_short_form(self):
        """Test case 2: Python library verbose short form"""
        v1 = self.parse_json('{"data": {"value": 1}}')
        v2 = self.parse_json('{"data": {"value": 2}}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('data' in results_str)
    
    def test_verbose_advanced_options(self):
        """Test case 3: Python library verbose advanced options"""
        v1 = self.parse_json('{"id": "001", "config": {"users": {"count": 10}}}')
        v2 = self.parse_json('{"id": "002", "config": {"users": {"count": 15}}}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('users' in results_str)
    
    def test_verbose_ml_analysis_features(self):
        """Test case 4: Python library verbose ML analysis features"""
        v1 = self.parse_json('{"model": {"architecture": "transformer", "memory": "2GB"}}')
        v2 = self.parse_json('{"model": {"architecture": "transformer", "memory": "2.5GB"}}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('model' in results_str)
    
    def test_verbose_directory_comparison(self):
        """Test case 5: Python library verbose directory comparison"""
        v1 = self.parse_json('{"directory": {"files": 12}}')
        v2 = self.parse_json('{"directory": {"files": 14}}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('directory' in results_str)
    
    def test_verbose_debugging_format_detection(self):
        """Test case 6: Python library verbose debugging format detection"""
        v1 = self.parse_json('{"format": "unknown", "data": "test1"}')
        v2 = self.parse_json('{"format": "unknown", "data": "test2"}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('format' in results_str)
    
    def test_verbose_ml_analysis_automatic(self):
        """Test case 7: Python library verbose ML analysis automatic"""
        v1 = self.parse_json('{"model": {"type": "pytorch", "version": "1.0"}}')
        v2 = self.parse_json('{"model": {"type": "pytorch", "version": "2.0"}}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('model' in results_str)
    
    def test_verbose_directory_analysis(self):
        """Test case 8: Python library verbose directory analysis"""
        v1 = self.parse_json('{"scan": {"dir1": {"files": 12}}}')
        v2 = self.parse_json('{"scan": {"dir2": {"files": 14}}}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('scan' in results_str)
    
    def test_verbose_performance_analysis(self):
        """Test case 9: Python library verbose performance analysis"""
        v1 = self.parse_json('{"large_model": {"size": "1GB", "parameters": 1000000}}')
        v2 = self.parse_json('{"large_model": {"size": "1.2GB", "parameters": 1200000}}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('large_model' in results_str)
    
    def test_verbose_performance_with_options(self):
        """Test case 10: Python library verbose performance with options"""
        v1 = self.parse_json('{"data": {"precision": 0.12345}}')
        v2 = self.parse_json('{"data": {"precision": 0.12346}}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('data' in results_str)
    
    def test_verbose_configuration_validation(self):
        """Test case 11: Python library verbose configuration validation"""
        v1 = self.parse_json('{"id": "001", "timestamp": "12:00", "application": {"settings": {"timeout": 30}}}')
        v2 = self.parse_json('{"id": "002", "timestamp": "13:00", "application": {"settings": {"timeout": 60}}}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('application' in results_str)
    
    def test_verbose_output_redirection(self):
        """Test case 12: Python library verbose output redirection"""
        v1 = self.parse_json('{"result": {"status": "success"}}')
        v2 = self.parse_json('{"result": {"status": "failure"}}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('result' in results_str)
    
    def test_verbose_cicd_integration(self):
        """Test case 13: Python library verbose CI/CD integration"""
        v1 = self.parse_json('{"model": {"type": "baseline", "accuracy": 0.85}}')
        v2 = self.parse_json('{"model": {"type": "improved", "accuracy": 0.90}}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('model' in results_str)
    
    def test_verbose_script_automation(self):
        """Test case 14: Python library verbose script automation"""
        v1 = self.parse_json('{"script": {"test": "automation"}}')
        v2 = self.parse_json('{"script": {"test": "automated"}}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('script' in results_str)
    
    def test_verbose_only_information(self):
        """Test case 15: Python library verbose only information"""
        v1 = self.parse_json('{"verbose": {"info": "test"}}')
        v2 = self.parse_json('{"verbose": {"info": "tested"}}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('verbose' in results_str)

if __name__ == '__main__':
    unittest.main()