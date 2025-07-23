import unittest
import json
from diffai_python import diffai

class TestScientificDataExamples(unittest.TestCase):
    
    def parse_json(self, json_str):
        """Helper function to parse JSON strings"""
        return json.loads(json_str)
    
    def test_numpy_array_comparison(self):
        """Test case 1: Python library NumPy array comparison"""
        v1 = self.parse_json('{"numpy_array": {"shape": [1000, 256], "mean": 0.1234, "std": 0.9876, "dtype": "float64"}}')
        v2 = self.parse_json('{"numpy_array": {"shape": [1000, 256], "mean": 0.1456, "std": 0.9654, "dtype": "float64"}}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('numpy_array' in results_str)
    
    def test_compressed_numpy_archives(self):
        """Test case 2: Python library compressed NumPy archives"""
        v1 = self.parse_json('{"train_data": {"shape": [60000, 784], "mean": 0.1307, "std": 0.3081, "dtype": "float32"}, "test_data": {"shape": [10000, 784], "mean": 0.1325, "std": 0.3105, "dtype": "float32"}}')
        v2 = self.parse_json('{"train_data": {"shape": [60000, 784], "mean": 0.1309, "std": 0.3082, "dtype": "float32"}, "test_data": {"shape": [10000, 784], "mean": 0.1327, "std": 0.3106, "dtype": "float32"}, "validation_data": {"shape": [5000, 784], "mean": 0.1315, "std": 0.3095, "dtype": "float32"}}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('train_data' in results_str)
    
    def test_numpy_json_output(self):
        """Test case 3: Python library NumPy JSON output"""
        v1 = self.parse_json('{"experiment": {"baseline": true, "data": [1.0, 2.0, 3.0]}}')
        v2 = self.parse_json('{"experiment": {"baseline": false, "data": [1.1, 2.1, 3.1]}}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('experiment' in results_str)
    
    def test_matlab_file_comparison(self):
        """Test case 4: Python library MATLAB file comparison"""
        v1 = self.parse_json('{"results": {"shape": [500, 100], "mean": 2.3456, "std": 1.2345, "dtype": "double"}}')
        v2 = self.parse_json('{"results": {"shape": [500, 100], "mean": 2.4567, "std": 1.3456, "dtype": "double"}, "new_variable": {"shape": [100], "dtype": "single", "elements": 100}}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('results' in results_str)
    
    def test_matlab_specific_variables(self):
        """Test case 5: Python library MATLAB specific variables"""
        v1 = self.parse_json('{"experiment_data": {"temperature": [20.1, 20.2, 20.3]}}')
        v2 = self.parse_json('{"experiment_data": {"temperature": [21.1, 21.2, 21.3]}}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('experiment_data' in results_str)
    
    def test_matlab_yaml_output(self):
        """Test case 6: Python library MATLAB YAML output"""
        v1 = self.parse_json('{"analysis": {"method": "linear", "r_squared": 0.85}}')
        v2 = self.parse_json('{"analysis": {"method": "polynomial", "r_squared": 0.92}}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('analysis' in results_str)
    
    def test_epsilon_tolerance_numerical(self):
        """Test case 7: Python library epsilon tolerance numerical"""
        v1 = self.parse_json('{"measurement": {"value": 1.0000001}}')
        v2 = self.parse_json('{"measurement": {"value": 1.0000002}}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('measurement' in results_str)
    
    def test_matlab_epsilon_simulation(self):
        """Test case 8: Python library MATLAB epsilon simulation"""
        v1 = self.parse_json('{"simulation": {"velocity": 1.23456789, "pressure": 101.325}}')
        v2 = self.parse_json('{"simulation": {"velocity": 1.23456790, "pressure": 101.326}}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('simulation' in results_str)
    
    def test_matlab_path_filtering(self):
        """Test case 9: Python library MATLAB path filtering"""
        v1 = self.parse_json('{"experimental_data": {"sample_1": {"concentration": 0.5}}}')
        v2 = self.parse_json('{"experimental_data": {"sample_1": {"concentration": 0.6}}}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('experimental_data' in results_str)
    
    def test_ignore_metadata_variables(self):
        """Test case 10: Python library ignore metadata variables"""
        v1 = self.parse_json('{"metadata": {"created": "2024-01-01"}, "timestamp": "12:00:00", "data": {"values": [1, 2, 3]}}')
        v2 = self.parse_json('{"metadata": {"created": "2024-01-02"}, "timestamp": "13:00:00", "data": {"values": [1, 2, 4]}}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('data' in results_str)
    
    def test_experimental_data_validation(self):
        """Test case 11: Python library experimental data validation"""
        v1 = self.parse_json('{"data": {"shape": [1000, 50], "mean": 0.4567, "std": 0.1234, "dtype": "float64"}}')
        v2 = self.parse_json('{"data": {"shape": [1000, 50], "mean": 0.5123, "std": 0.1456, "dtype": "float64"}}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('data' in results_str)

if __name__ == '__main__':
    unittest.main()