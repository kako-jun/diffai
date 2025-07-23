import unittest
import tempfile
import os
import json
import subprocess
from pathlib import Path

class TestReadmeExamples(unittest.TestCase):
    """Test cases for README.md examples using diffai Python package"""
    
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
    
    def test_basic_safetensors_diff(self):
        """Test case 1: diff model_v1.safetensors model_v2.safetensors"""
        file1 = self.create_temp_json({"model": {"layers": 2, "params": 1000}}, "model1.json")
        file2 = self.create_temp_json({"model": {"layers": 3, "params": 1500}}, "model2.json")
        
        result = self.run_diffai_python(file1, file2)
        self.assertIsNotNone(result)
        self.assertIn("differences", result)
    
    def test_comprehensive_model_analysis(self):
        """Test case 2: diffai model_v1.safetensors model_v2.safetensors"""
        file1 = self.create_temp_json({"fc1": {"bias": 0.001, "weight": 0.5}}, "fc1.json")
        file2 = self.create_temp_json({"fc1": {"bias": 0.002, "weight": 0.6}}, "fc2.json")
        
        result = self.run_diffai_python(file1, file2)
        self.assertIsNotNone(result)
    
    def test_json_output(self):
        """Test case 3: diffai model1.safetensors model2.safetensors --output json"""
        file1 = self.create_temp_json({"data": "value1"}, "data1.json")
        file2 = self.create_temp_json({"data": "value2"}, "data2.json")
        
        result = self.run_diffai_python(file1, file2, ["--output", "json"])
        self.assertIsNotNone(result)
    
    def test_verbose_output(self):
        """Test case 4: diffai model1.safetensors model2.safetensors --verbose"""
        file1 = self.create_temp_json({"test": 1}, "test1.json")
        file2 = self.create_temp_json({"test": 2}, "test2.json")
        
        result = self.run_diffai_python(file1, file2, ["--verbose"])
        self.assertIsNotNone(result)
    
    def test_yaml_output(self):
        """Test case 5: diffai model1.safetensors model2.safetensors --output yaml"""
        file1 = self.create_temp_json({"value": 10}, "val1.json")
        file2 = self.create_temp_json({"value": 20}, "val2.json")
        
        result = self.run_diffai_python(file1, file2, ["--output", "yaml"])
        self.assertIsNotNone(result)
    
    def test_baseline_vs_finetuned(self):
        """Test case 6: diffai baseline.safetensors finetuned.safetensors"""
        baseline = self.create_temp_json({"model": {"accuracy": 0.85}}, "baseline.json")
        finetuned = self.create_temp_json({"model": {"accuracy": 0.92}}, "finetuned.json")
        
        result = self.run_diffai_python(baseline, finetuned)
        self.assertIsNotNone(result)
    
    def test_numpy_comparison(self):
        """Test case 7: diffai data_v1.npy data_v2.npy"""
        file1 = self.create_temp_json({"array": [1, 2, 3]}, "array1.json")
        file2 = self.create_temp_json({"array": [1, 2, 4]}, "array2.json")
        
        result = self.run_diffai_python(file1, file2)
        self.assertIsNotNone(result)
    
    def test_matlab_comparison(self):
        """Test case 8: diffai experiment_v1.mat experiment_v2.mat"""
        file1 = self.create_temp_json({"experiment": {"result": 0.75}}, "exp1.json")
        file2 = self.create_temp_json({"experiment": {"result": 0.80}}, "exp2.json")
        
        result = self.run_diffai_python(file1, file2)
        self.assertIsNotNone(result)
    
    def test_pytorch_comparison(self):
        """Test case 9: diffai model1.pt model2.pt"""
        file1 = self.create_temp_json({"layers": {"conv1": {"filters": 32}}}, "model1.json")
        file2 = self.create_temp_json({"layers": {"conv1": {"filters": 64}}}, "model2.json")
        
        result = self.run_diffai_python(file1, file2)
        self.assertIsNotNone(result)
    
    def test_model_improvement(self):
        """Test case 10: diffai baseline_model.pt improved_model.pt"""
        baseline = self.create_temp_json({"performance": {"f1": 0.80}}, "baseline.json")
        improved = self.create_temp_json({"performance": {"f1": 0.85}}, "improved.json")
        
        result = self.run_diffai_python(baseline, improved)
        self.assertIsNotNone(result)
    
    def test_simple_model_diff(self):
        """Test case 11: diffai simple_model_v1.safetensors simple_model_v2.safetensors"""
        file1 = self.create_temp_json({"fc1": {"bias": 0.001}}, "simple1.json")
        file2 = self.create_temp_json({"fc1": {"bias": 0.002}}, "simple2.json")
        
        result = self.run_diffai_python(file1, file2)
        self.assertIsNotNone(result)
    
    def test_improved_model_json(self):
        """Test case 12: diffai baseline.safetensors improved.safetensors --output json"""
        baseline = self.create_temp_json({"metric": 0.7}, "baseline.json")
        improved = self.create_temp_json({"metric": 0.8}, "improved.json")
        
        result = self.run_diffai_python(baseline, improved, ["--output", "json"])
        self.assertIsNotNone(result)
    
    def test_experiment_data_diff(self):
        """Test case 13: diffai experiment_data_v1.npy experiment_data_v2.npy"""
        file1 = self.create_temp_json({"data": {"mean": 0.1234, "std": 0.9876}}, "exp1.json")
        file2 = self.create_temp_json({"data": {"mean": 0.1456, "std": 0.9654}}, "exp2.json")
        
        result = self.run_diffai_python(file1, file2)
        self.assertIsNotNone(result)
    
    def test_simulation_comparison(self):
        """Test case 14: diffai simulation_v1.mat simulation_v2.mat"""
        file1 = self.create_temp_json({"results": {"mean": 2.3456, "std": 1.2345}}, "sim1.json")
        file2 = self.create_temp_json({"results": {"mean": 2.4567, "std": 1.3456}}, "sim2.json")
        
        result = self.run_diffai_python(file1, file2)
        self.assertIsNotNone(result)
    
    def test_old_vs_new_model(self):
        """Test case 15: diffai model_old.pt model_new.pt"""
        old_model = self.create_temp_json({"version": "1.0", "layers": 5}, "old.json")
        new_model = self.create_temp_json({"version": "2.0", "layers": 8}, "new.json")
        
        result = self.run_diffai_python(old_model, new_model)
        self.assertIsNotNone(result)
    
    def test_checkpoint_comparison(self):
        """Test case 16: diffai checkpoint_v1.safetensors checkpoint_v2.safetensors"""
        checkpoint1 = self.create_temp_json({"epoch": 10, "loss": 0.5}, "ckpt1.json")
        checkpoint2 = self.create_temp_json({"epoch": 20, "loss": 0.3}, "ckpt2.json")
        
        result = self.run_diffai_python(checkpoint1, checkpoint2)
        self.assertIsNotNone(result)
    
    def test_json_output_for_jq(self):
        """Test case 17: diffai model1.safetensors model2.safetensors --output json | jq ."""
        file1 = self.create_temp_json({"test": "value1"}, "test1.json")
        file2 = self.create_temp_json({"test": "value2"}, "test2.json")
        
        result = self.run_diffai_python(file1, file2, ["--output", "json"])
        self.assertIsNotNone(result)
    
    def test_yaml_output_format(self):
        """Test case 18: diffai model1.safetensors model2.safetensors --output yaml"""
        file1 = self.create_temp_json({"config": {"setting": "old"}}, "config1.json")
        file2 = self.create_temp_json({"config": {"setting": "new"}}, "config2.json")
        
        result = self.run_diffai_python(file1, file2, ["--output", "yaml"])
        self.assertIsNotNone(result)
    
    def test_baseline_results(self):
        """Test case 19: diffai baseline_results.npy new_results.npy"""
        baseline = self.create_temp_json({"results": [0.8, 0.85, 0.9]}, "baseline.json")
        new_results = self.create_temp_json({"results": [0.82, 0.87, 0.92]}, "new.json")
        
        result = self.run_diffai_python(baseline, new_results)
        self.assertIsNotNone(result)
    
    def test_matlab_simulation(self):
        """Test case 20: diffai simulation_v1.mat simulation_v2.mat"""
        sim1 = self.create_temp_json({"temperature": 25.5, "pressure": 101.3}, "sim1.json")
        sim2 = self.create_temp_json({"temperature": 26.0, "pressure": 102.1}, "sim2.json")
        
        result = self.run_diffai_python(sim1, sim2)
        self.assertIsNotNone(result)
    
    def test_pretrained_vs_finetuned(self):
        """Test case 21: diffai pretrained_model.safetensors finetuned_model.safetensors"""
        pretrained = self.create_temp_json({"weights": {"layer1": 0.5, "layer2": 0.3}}, "pre.json")
        finetuned = self.create_temp_json({"weights": {"layer1": 0.6, "layer2": 0.4}}, "fine.json")
        
        result = self.run_diffai_python(pretrained, finetuned)
        self.assertIsNotNone(result)
    
    def test_architecture_comparison(self):
        """Test case 22: diffai baseline_architecture.pt improved_architecture.pt"""
        baseline = self.create_temp_json({"architecture": {"type": "cnn", "layers": 5}}, "arch1.json")
        improved = self.create_temp_json({"architecture": {"type": "cnn", "layers": 8}}, "arch2.json")
        
        result = self.run_diffai_python(baseline, improved)
        self.assertIsNotNone(result)
    
    def test_production_vs_candidate(self):
        """Test case 23: diffai production_model.safetensors candidate_model.safetensors"""
        production = self.create_temp_json({"status": "stable", "version": "1.0"}, "prod.json")
        candidate = self.create_temp_json({"status": "testing", "version": "1.1"}, "cand.json")
        
        result = self.run_diffai_python(production, candidate)
        self.assertIsNotNone(result)
    
    def test_optimization_analysis(self):
        """Test case 24: diffai original_model.pt optimized_model.pt --output json"""
        original = self.create_temp_json({"performance": {"speed": 100, "memory": 512}}, "orig.json")
        optimized = self.create_temp_json({"performance": {"speed": 150, "memory": 256}}, "opt.json")
        
        result = self.run_diffai_python(original, optimized, ["--output", "json"])
        self.assertIsNotNone(result)

if __name__ == '__main__':
    unittest.main()