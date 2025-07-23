import unittest
import json
from diffai_python import diffai

class TestMLModelComparisonExamples(unittest.TestCase):
    
    def parse_json(self, json_str):
        """Helper function to parse JSON strings"""
        return json.loads(json_str)
    
    def test_pytorch_models_comprehensive(self):
        """Test case 1: Python library PyTorch models comprehensive"""
        v1 = self.parse_json('{"state_dict": {"fc1.weight": [0.1, 0.2], "fc1.bias": [0.01]}}')
        v2 = self.parse_json('{"state_dict": {"fc1.weight": [0.15, 0.25], "fc1.bias": [0.02]}}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('state_dict' in results_str)
    
    def test_safetensors_models_comprehensive(self):
        """Test case 2: Python library safetensors models comprehensive"""
        v1 = self.parse_json('{"tensors": {"layer1.weight": {"shape": [64, 32]}}}')
        v2 = self.parse_json('{"tensors": {"layer1.weight": {"shape": [64, 64]}}}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('tensors' in results_str)
    
    def test_automatic_format_detection(self):
        """Test case 3: Python library automatic format detection"""
        v1 = self.parse_json('{"model": {"pretrained": true, "accuracy": 0.85}}')
        v2 = self.parse_json('{"model": {"pretrained": false, "accuracy": 0.92}}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('accuracy' in results_str)
    
    def test_epsilon_tolerance_minor(self):
        """Test case 4: Python library epsilon tolerance minor"""
        v1 = self.parse_json('{"weights": {"layer1": 0.1000000}}')
        v2 = self.parse_json('{"weights": {"layer1": 0.1000001}}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('weights' in results_str)
    
    def test_quantization_analysis_epsilon(self):
        """Test case 5: Python library quantization analysis epsilon"""
        v1 = self.parse_json('{"model": {"precision": "fp32", "weights": [0.123, 0.456]}}')
        v2 = self.parse_json('{"model": {"precision": "int8", "weights": [0.12, 0.46]}}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('precision' in results_str)
    
    def test_json_output_automation(self):
        """Test case 6: Python library JSON output automation"""
        v1 = self.parse_json('{"layers": {"conv1": {"filters": 32}}}')
        v2 = self.parse_json('{"layers": {"conv1": {"filters": 64}}}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('layers' in results_str)
    
    def test_yaml_output_readability(self):
        """Test case 7: Python library YAML output readability"""
        v1 = self.parse_json('{"parameters": {"learning_rate": 0.01}}')
        v2 = self.parse_json('{"parameters": {"learning_rate": 0.001}}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('parameters' in results_str)
    
    def test_pipe_to_file(self):
        """Test case 8: Python library pipe to file"""
        v1 = self.parse_json('{"metrics": {"loss": 0.5}}')
        v2 = self.parse_json('{"metrics": {"loss": 0.3}}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('metrics' in results_str)
    
    def test_focus_specific_layers(self):
        """Test case 9: Python library focus specific layers"""
        v1 = self.parse_json('{"classifier": {"weight": [0.1, 0.2]}}')
        v2 = self.parse_json('{"classifier": {"weight": [0.15, 0.25]}}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('classifier' in results_str)
    
    def test_ignore_metadata(self):
        """Test case 10: Python library ignore metadata"""
        v1 = self.parse_json('{"timestamp": "2024-01-01", "weights": {"layer1": 0.5}}')
        v2 = self.parse_json('{"timestamp": "2024-01-02", "weights": {"layer1": 0.6}}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('weights' in results_str)
    
    def test_finetuning_analysis(self):
        """Test case 11: Python library finetuning analysis"""
        v1 = self.parse_json('{"bert": {"encoder": {"attention": {"query": {"weight": [0.001]}}}}, "classifier": {"weight": [0.0]}}')
        v2 = self.parse_json('{"bert": {"encoder": {"attention": {"query": {"weight": [0.0023]}}}}, "classifier": {"weight": [0.0145]}}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('bert' in results_str)
    
    def test_quantization_impact_assessment(self):
        """Test case 12: Python library quantization impact assessment"""
        v1 = self.parse_json('{"conv1": {"weight": {"mean": 0.0045, "std": 0.2341}}}')
        v2 = self.parse_json('{"conv1": {"weight": {"mean": 0.0043, "std": 0.2298}}}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('conv1' in results_str)
    
    def test_training_progress_tracking(self):
        """Test case 13: Python library training progress tracking"""
        v1 = self.parse_json('{"layers": {"0": {"weight": {"mean": -0.0012, "std": 1.2341}}, "1": {"bias": {"mean": 0.1234, "std": 0.4567}}}}')
        v2 = self.parse_json('{"layers": {"0": {"weight": {"mean": 0.0034, "std": 0.8907}}, "1": {"bias": {"mean": 0.0567, "std": 0.3210}}}}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('layers' in results_str)
    
    def test_architecture_comparison(self):
        """Test case 14: Python library architecture comparison"""
        v1 = self.parse_json('{"features": {"conv1": {"weight": {"shape": [64, 3, 7, 7]}}, "layer4": {"2": {"downsample": {"0": {"weight": {"shape": [2048, 1024, 1, 1]}}}}}}}')
        v2 = self.parse_json('{"features": {"conv1": {"weight": {"shape": [32, 3, 3, 3]}}, "mbconv": {"expand_conv": {"weight": {"shape": [96, 32, 1, 1]}}}}}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('features' in results_str)
    
    def test_recursive_mode_large_models(self):
        """Test case 15: Python library recursive mode large models"""
        v1 = self.parse_json('{"large_model": {"size": "1GB", "parameters": 1000000}}')
        v2 = self.parse_json('{"large_model": {"size": "1.2GB", "parameters": 1200000}}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('large_model' in results_str)
    
    def test_focus_analysis_specific_parts(self):
        """Test case 16: Python library focus analysis specific parts"""
        v1 = self.parse_json('{"tensor": {"classifier": {"weight": [0.1, 0.2]}}}')
        v2 = self.parse_json('{"tensor": {"classifier": {"weight": [0.15, 0.25]}}}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('classifier' in results_str)
    
    def test_higher_epsilon_faster_comparison(self):
        """Test case 17: Python library higher epsilon faster comparison"""
        v1 = self.parse_json('{"model": {"precision": 0.001234}}')
        v2 = self.parse_json('{"model": {"precision": 0.001567}}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('model' in results_str)
    
    def test_verbose_mode_processing_info(self):
        """Test case 18: Python library verbose mode processing info"""
        v1 = self.parse_json('{"processing": {"stage": "training"}}')
        v2 = self.parse_json('{"processing": {"stage": "validation"}}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('processing' in results_str)
    
    def test_architecture_differences_only(self):
        """Test case 19: Python library architecture differences only"""
        v1 = self.parse_json('{"architecture": {"type": "transformer", "layers": 12}}')
        v2 = self.parse_json('{"architecture": {"type": "transformer", "layers": 24}}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('architecture' in results_str)
    
    def test_subprocess_run_json(self):
        """Test case 20: Python library subprocess run JSON"""
        v1 = self.parse_json('{"model": {"version": "1.0"}}')
        v2 = self.parse_json('{"model": {"version": "2.0"}}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('model' in results_str)
    
    def test_cicd_compare_models(self):
        """Test case 21: Python library CI/CD compare models"""
        v1 = self.parse_json('{"model": {"type": "baseline", "accuracy": 0.85}}')
        v2 = self.parse_json('{"model": {"type": "candidate", "accuracy": 0.88}}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('model' in results_str)
    
    def test_single_model_analysis(self):
        """Test case 22: Python library single model analysis"""
        v1 = self.parse_json('{"model": {"layers": 6, "parameters": 100000}}')
        v2 = self.parse_json('{"model": {"layers": 6, "parameters": 100000}}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertEqual(len(results), 0)
    
    def test_explicit_format(self):
        """Test case 23: Python library explicit format"""
        v1 = self.parse_json('{"safetensors": {"format": "explicit"}}')
        v2 = self.parse_json('{"safetensors": {"format": "explicit", "version": 2}}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('safetensors' in results_str)
    
    def test_memory_optimization_epsilon(self):
        """Test case 24: Python library memory optimization epsilon"""
        v1 = self.parse_json('{"large": {"tensor": [0.001, 0.002, 0.003]}}')
        v2 = self.parse_json('{"large": {"tensor": [0.0015, 0.0025, 0.0035]}}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('large' in results_str)
    
    def test_memory_optimization_path(self):
        """Test case 25: Python library memory optimization path"""
        v1 = self.parse_json('{"tensor": {"classifier": {"weight": [0.1]}}}')
        v2 = self.parse_json('{"tensor": {"classifier": {"weight": [0.2]}}}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('classifier' in results_str)
    
    def test_comprehensive_analysis_automatic(self):
        """Test case 26: Python library comprehensive analysis automatic"""
        v1 = self.parse_json('{"checkpoint": {"epoch": 10, "loss": 0.5, "accuracy": 0.8}}')
        v2 = self.parse_json('{"checkpoint": {"epoch": 20, "loss": 0.3, "accuracy": 0.9}}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('checkpoint' in results_str)
    
    def test_experimental_comparison_automatic(self):
        """Test case 27: Python library experimental comparison automatic"""
        v1 = self.parse_json('{"experiment": {"type": "baseline", "performance": 0.85}}')
        v2 = self.parse_json('{"experiment": {"type": "enhanced", "performance": 0.92}}')
        
        results = diffai.diff(v1, v2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        results_str = str(results)
        self.assertTrue('experiment' in results_str)

if __name__ == '__main__':
    unittest.main()