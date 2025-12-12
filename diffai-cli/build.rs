use std::collections::HashMap;
use std::fs;
use std::path::Path;

fn main() {
    // Generate test fixtures for trycmd tests
    let fixtures_dir = Path::new("tests/fixtures");
    if !fixtures_dir.exists() {
        fs::create_dir_all(fixtures_dir).unwrap();
    }

    // Generate safetensors test files
    generate_safetensors_fixtures(fixtures_dir);

    println!("cargo:rerun-if-changed=build.rs");
}

fn generate_safetensors_fixtures(dir: &Path) {
    // model_v1.safetensors
    let model_v1_path = dir.join("model_v1.safetensors");
    if !model_v1_path.exists() {
        let tensors_v1 = create_model_v1_tensors();
        let metadata_v1: HashMap<String, String> = HashMap::from([
            ("format".to_string(), "pt".to_string()),
            ("version".to_string(), "1.0".to_string()),
        ]);
        safetensors::serialize_to_file(&tensors_v1, &Some(metadata_v1), &model_v1_path).unwrap();
    }

    // model_v2.safetensors (with changes)
    let model_v2_path = dir.join("model_v2.safetensors");
    if !model_v2_path.exists() {
        let tensors_v2 = create_model_v2_tensors();
        let metadata_v2: HashMap<String, String> = HashMap::from([
            ("format".to_string(), "pt".to_string()),
            ("version".to_string(), "2.0".to_string()),
        ]);
        safetensors::serialize_to_file(&tensors_v2, &Some(metadata_v2), &model_v2_path).unwrap();
    }
}

fn create_model_v1_tensors() -> HashMap<String, safetensors::tensor::TensorView<'static>> {
    // Simple tensors with known statistics - uniform distribution around 0
    let fc1_weight: Vec<f32> = (0..512 * 256)
        .map(|i| {
            let x = (i as f32 / (512.0 * 256.0)) * 2.0 - 1.0; // range [-1, 1]
            x * 0.1 // mean ~0, std ~0.058
        })
        .collect();
    let fc2_weight: Vec<f32> = (0..256 * 128)
        .map(|i| {
            let x = (i as f32 / (256.0 * 128.0)) * 2.0 - 1.0;
            x * 0.15 // mean ~0, std ~0.087
        })
        .collect();

    // Leak to get 'static lifetime for test fixtures
    let fc1_data: &'static [u8] = Box::leak(
        fc1_weight
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect::<Vec<u8>>()
            .into_boxed_slice(),
    );
    let fc2_data: &'static [u8] = Box::leak(
        fc2_weight
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect::<Vec<u8>>()
            .into_boxed_slice(),
    );

    HashMap::from([
        (
            "fc1.weight".to_string(),
            safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![512, 256], fc1_data)
                .unwrap(),
        ),
        (
            "fc2.weight".to_string(),
            safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![256, 128], fc2_data)
                .unwrap(),
        ),
    ])
}

fn create_model_v2_tensors() -> HashMap<String, safetensors::tensor::TensorView<'static>> {
    // Modified tensors - clearly different statistics
    // Mean shifted by +0.05, std increased
    let fc1_weight: Vec<f32> = (0..512 * 256)
        .map(|i| {
            let x = (i as f32 / (512.0 * 256.0)) * 2.0 - 1.0;
            x * 0.15 + 0.05 // mean ~0.05, std ~0.087 (different from v1)
        })
        .collect();
    let fc2_weight: Vec<f32> = (0..256 * 128)
        .map(|i| {
            let x = (i as f32 / (256.0 * 128.0)) * 2.0 - 1.0;
            x * 0.2 - 0.03 // mean ~-0.03, std ~0.115 (different from v1)
        })
        .collect();

    let fc1_data: &'static [u8] = Box::leak(
        fc1_weight
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect::<Vec<u8>>()
            .into_boxed_slice(),
    );
    let fc2_data: &'static [u8] = Box::leak(
        fc2_weight
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect::<Vec<u8>>()
            .into_boxed_slice(),
    );

    HashMap::from([
        (
            "fc1.weight".to_string(),
            safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![512, 256], fc1_data)
                .unwrap(),
        ),
        (
            "fc2.weight".to_string(),
            safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![256, 128], fc2_data)
                .unwrap(),
        ),
    ])
}
