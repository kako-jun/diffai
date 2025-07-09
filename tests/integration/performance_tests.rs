use diffai_core::diff_ml_models;
use std::path::Path;
use std::time::Instant;

#[test]
fn test_large_model_performance() {
    // 大容量モデルのパフォーマンステスト
    let model1_path = "tests/fixtures/ml_models/large_model.safetensors";
    let model2_path = "tests/fixtures/ml_models/large_model.pt";

    let start = Instant::now();

    // メモリ使用量測定開始
    let initial_memory = get_memory_usage();

    // 比較実行
    let result = diff_ml_models(Path::new(model1_path), Path::new(model2_path));

    let duration = start.elapsed();
    let peak_memory = get_memory_usage();
    let memory_used = peak_memory - initial_memory;

    println!("Performance Test Results:");
    println!("  Duration: {:?}", duration);
    println!("  Memory Used: {} MB", memory_used / 1024 / 1024);
    println!("  Differences Found: {}", result.unwrap_or_default().len());

    // パフォーマンス基準チェック
    assert!(
        duration.as_secs() < 10,
        "処理時間が10秒を超えています: {:?}",
        duration
    );
    assert!(
        memory_used < 500 * 1024 * 1024,
        "メモリ使用量が500MBを超えています: {} MB",
        memory_used / 1024 / 1024
    );
}

#[test]
fn test_multiple_models_batch_performance() {
    // 複数モデルのバッチ処理パフォーマンステスト
    let model_pairs = vec![
        (
            "tests/fixtures/ml_models/simple_base.safetensors",
            "tests/fixtures/ml_models/simple_modified.safetensors",
        ),
        (
            "tests/fixtures/ml_models/model_fp32.safetensors",
            "tests/fixtures/ml_models/model_quantized.safetensors",
        ),
        (
            "tests/fixtures/ml_models/normal_model.safetensors",
            "tests/fixtures/ml_models/anomalous_model.safetensors",
        ),
    ];

    let start = Instant::now();
    let initial_memory = get_memory_usage();

    let mut total_differences = 0;

    for (model1, model2) in model_pairs {
        let result = diff_ml_models(Path::new(model1), Path::new(model2));
        if let Ok(diffs) = result {
            total_differences += diffs.len();
        }
    }

    let duration = start.elapsed();
    let peak_memory = get_memory_usage();
    let memory_used = peak_memory - initial_memory;

    println!("Batch Performance Test Results:");
    println!("  Total Duration: {:?}", duration);
    println!("  Memory Used: {} MB", memory_used / 1024 / 1024);
    println!("  Total Differences: {}", total_differences);

    // バッチ処理パフォーマンス基準
    assert!(
        duration.as_secs() < 15,
        "バッチ処理時間が15秒を超えています: {:?}",
        duration
    );
    assert!(
        memory_used < 1000 * 1024 * 1024,
        "メモリ使用量が1GBを超えています: {} MB",
        memory_used / 1024 / 1024
    );
}

#[test]
fn test_memory_efficiency_with_large_tensors() {
    // 大規模テンソルでのメモリ効率テスト
    let model1_path = "tests/fixtures/ml_models/large_model.safetensors";
    let model2_path = "tests/fixtures/ml_models/large_model.pt";

    // メモリ使用量の段階的測定
    let memory_before = get_memory_usage();

    let result = diff_ml_models(Path::new(model1_path), Path::new(model2_path));

    let memory_after = get_memory_usage();
    let memory_growth = memory_after - memory_before;

    println!("Memory Efficiency Test Results:");
    println!("  Memory Growth: {} MB", memory_growth / 1024 / 1024);
    println!("  Analysis Success: {}", result.is_ok());

    // メモリ効率基準
    assert!(
        memory_growth < 200 * 1024 * 1024,
        "メモリ増加が200MBを超えています: {} MB",
        memory_growth / 1024 / 1024
    );
}

#[test]
fn test_concurrent_analysis_performance() {
    // 並行解析のパフォーマンステスト
    use std::sync::Arc;
    use std::thread;

    let model_pairs = Arc::new(vec![
        (
            "tests/fixtures/ml_models/simple_base.safetensors",
            "tests/fixtures/ml_models/simple_modified.safetensors",
        ),
        (
            "tests/fixtures/ml_models/model_fp32.safetensors",
            "tests/fixtures/ml_models/model_quantized.safetensors",
        ),
        (
            "tests/fixtures/ml_models/normal_model.safetensors",
            "tests/fixtures/ml_models/anomalous_model.safetensors",
        ),
    ]);

    let start = Instant::now();
    let initial_memory = get_memory_usage();

    let mut handles = vec![];

    for i in 0..3 {
        let pairs = Arc::clone(&model_pairs);

        let handle = thread::spawn(move || {
            let (model1, model2) = pairs[i];
            diff_ml_models(Path::new(model1), Path::new(model2))
        });

        handles.push(handle);
    }

    let mut total_differences = 0;
    for handle in handles {
        if let Ok(Ok(diffs)) = handle.join() {
            total_differences += diffs.len();
        }
    }

    let duration = start.elapsed();
    let peak_memory = get_memory_usage();
    let memory_used = peak_memory - initial_memory;

    println!("Concurrent Analysis Performance Test Results:");
    println!("  Duration: {:?}", duration);
    println!("  Memory Used: {} MB", memory_used / 1024 / 1024);
    println!("  Total Differences: {}", total_differences);

    // 並行処理パフォーマンス基準
    assert!(
        duration.as_secs() < 20,
        "並行処理時間が20秒を超えています: {:?}",
        duration
    );
    assert!(
        memory_used < 1500 * 1024 * 1024,
        "メモリ使用量が1.5GBを超えています: {} MB",
        memory_used / 1024 / 1024
    );
}

fn get_memory_usage() -> usize {
    // メモリ使用量取得のヘルパー関数
    use std::process::Command;

    if cfg!(target_os = "linux") {
        // Linux でのメモリ使用量取得
        if let Ok(output) = Command::new("ps")
            .args(&["-o", "rss=", "-p"])
            .arg(std::process::id().to_string())
            .output()
        {
            if let Ok(s) = String::from_utf8(output.stdout) {
                if let Ok(rss_kb) = s.trim().parse::<usize>() {
                    return rss_kb * 1024; // KB to bytes
                }
            }
        }
    }

    // フォールバック: 0を返す
    0
}

#[cfg(test)]
mod benchmark_utils {
    use super::*;

    pub fn run_performance_benchmark(
        name: &str,
        iterations: usize,
        test_fn: impl Fn() -> Result<(), Box<dyn std::error::Error>>,
    ) {
        let mut durations = Vec::new();
        let mut memory_usages = Vec::new();

        for _ in 0..iterations {
            let start = Instant::now();
            let initial_memory = get_memory_usage();

            let _ = test_fn();

            let duration = start.elapsed();
            let peak_memory = get_memory_usage();
            let memory_used = peak_memory - initial_memory;

            durations.push(duration);
            memory_usages.push(memory_used);
        }

        let avg_duration = durations.iter().sum::<std::time::Duration>() / iterations as u32;
        let avg_memory = memory_usages.iter().sum::<usize>() / iterations;

        println!("Benchmark Results for {}:", name);
        println!("  Average Duration: {:?}", avg_duration);
        println!("  Average Memory: {} MB", avg_memory / 1024 / 1024);
        println!("  Iterations: {}", iterations);
    }
}
