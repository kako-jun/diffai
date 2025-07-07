#![allow(clippy::uninlined_format_args)]

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use diffai_core::{diff_ml_models_enhanced, parse_safetensors_model};
use std::path::Path;

/// Benchmark ML model comparison with different model sizes
fn benchmark_ml_model_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("ml_model_comparison");

    // Use existing test models for benchmarking
    let test_model_paths = vec![
        (
            "simple_base",
            "tests/fixtures/ml_models/simple_base.safetensors",
        ),
        (
            "simple_modified",
            "tests/fixtures/ml_models/simple_modified.safetensors",
        ),
        (
            "small_model",
            "tests/fixtures/ml_models/small_model.safetensors",
        ),
    ];

    let mut available_models = Vec::new();
    for (name, path) in test_model_paths {
        if Path::new(path).exists() {
            available_models.push((name, Path::new(path)));
        }
    }

    if available_models.len() >= 2 {
        let (name1, model1_path) = &available_models[0];
        let (name2, model2_path) = &available_models[1];
        let comparison_name = format!("{}_{}", name1, name2);

        group.bench_function(format!("basic_comparison_{}", comparison_name), |b| {
            b.iter(|| {
                black_box(diff_ml_models_enhanced(
                    black_box(*model1_path),
                    black_box(*model2_path),
                    Some(1e-6),
                    false, // show_layer_impact
                    false, // quantization_analysis
                    false, // detailed_stats
                    false, // learning_progress
                    false, // convergence_analysis
                    false, // anomaly_detection
                    false, // gradient_analysis
                    false, // memory_analysis
                    false, // inference_speed_estimate
                    false, // regression_test
                    false, // alert_on_degradation
                    false, // review_friendly
                    false, // change_summary
                    false, // risk_assessment
                    false, // architecture_comparison
                    false, // param_efficiency_analysis
                    false, // hyperparameter_impact
                    false, // learning_rate_analysis
                    false, // deployment_readiness
                    false, // performance_impact_estimate
                    false, // generate_report
                    false, // markdown_output
                    false, // include_charts
                    false, // embedding_analysis
                    false, // similarity_matrix
                    false, // clustering_change
                    false, // attention_analysis
                    false, // head_importance
                    false, // attention_pattern_diff
                ))
            })
        });

        // Benchmark with advanced features enabled
        group.bench_function(format!("advanced_features_{}", comparison_name), |b| {
            b.iter(|| {
                black_box(diff_ml_models_enhanced(
                    black_box(*model1_path),
                    black_box(*model2_path),
                    Some(1e-6),
                    true,  // show_layer_impact
                    true,  // quantization_analysis
                    true,  // detailed_stats
                    true,  // learning_progress
                    true,  // convergence_analysis
                    true,  // anomaly_detection
                    false, // gradient_analysis
                    true,  // memory_analysis
                    true,  // inference_speed_estimate
                    false, // regression_test
                    false, // alert_on_degradation
                    false, // review_friendly
                    false, // change_summary
                    false, // risk_assessment
                    true,  // architecture_comparison
                    true,  // param_efficiency_analysis
                    false, // hyperparameter_impact
                    false, // learning_rate_analysis
                    false, // deployment_readiness
                    false, // performance_impact_estimate
                    false, // generate_report
                    false, // markdown_output
                    false, // include_charts
                    false, // embedding_analysis
                    false, // similarity_matrix
                    false, // clustering_change
                    false, // attention_analysis
                    false, // head_importance
                    false, // attention_pattern_diff
                ))
            })
        });
    }

    group.finish();
}

/// Benchmark memory efficiency with large models
fn benchmark_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_efficiency");

    // Test with different sized models if available
    let test_model_paths = vec![
        ("small", "tests/fixtures/ml_models/small_model.safetensors"),
        ("large", "tests/fixtures/ml_models/large_model.safetensors"),
    ];

    for (size_name, path) in test_model_paths {
        if Path::new(path).exists() {
            group.bench_function(format!("parse_model_{}", size_name), |b| {
                b.iter(|| black_box(parse_safetensors_model(black_box(Path::new(path)))))
            });
        }
    }

    group.finish();
}

/// Benchmark real model files if available
fn benchmark_real_models(c: &mut Criterion) {
    let mut group = c.benchmark_group("real_models");

    // Test with real model files from test fixtures
    let test_model_paths = vec![
        "tests/fixtures/ml_models/simple_base.safetensors",
        "tests/fixtures/ml_models/simple_modified.safetensors",
        "tests/fixtures/ml_models/small_model.safetensors",
        "tests/fixtures/ml_models/large_model.safetensors",
    ];

    let mut available_models = Vec::new();
    for path in test_model_paths {
        if Path::new(path).exists() {
            available_models.push(Path::new(path));
        }
    }

    if available_models.len() >= 2 {
        let model1_path = available_models[0];
        let model2_path = available_models[1];

        group.bench_function("real_model_comparison", |b| {
            b.iter(|| {
                black_box(diff_ml_models_enhanced(
                    black_box(model1_path),
                    black_box(model2_path),
                    Some(1e-6),
                    true,  // show_layer_impact
                    false, // quantization_analysis
                    true,  // detailed_stats
                    false, // learning_progress
                    false, // convergence_analysis
                    false, // anomaly_detection
                    false, // gradient_analysis
                    false, // memory_analysis
                    false, // inference_speed_estimate
                    false, // regression_test
                    false, // alert_on_degradation
                    false, // review_friendly
                    false, // change_summary
                    false, // risk_assessment
                    false, // architecture_comparison
                    false, // param_efficiency_analysis
                    false, // hyperparameter_impact
                    false, // learning_rate_analysis
                    false, // deployment_readiness
                    false, // performance_impact_estimate
                    false, // generate_report
                    false, // markdown_output
                    false, // include_charts
                    false, // embedding_analysis
                    false, // similarity_matrix
                    false, // clustering_change
                    false, // attention_analysis
                    false, // head_importance
                    false, // attention_pattern_diff
                ))
            })
        });
    }

    group.finish();
}

/// Benchmark tensor statistics calculations
fn benchmark_tensor_stats(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_statistics");

    // Test with available model files
    let test_model_paths = vec![
        (
            "simple_base",
            "tests/fixtures/ml_models/simple_base.safetensors",
        ),
        (
            "small_model",
            "tests/fixtures/ml_models/small_model.safetensors",
        ),
    ];

    for (name, path) in test_model_paths {
        if Path::new(path).exists() {
            group.bench_function(format!("stats_calculation_{}", name), |b| {
                b.iter(|| {
                    let model = parse_safetensors_model(black_box(Path::new(path))).unwrap();
                    black_box(model)
                })
            });
        }
    }

    group.finish();
}

criterion_group!(
    ml_benches,
    benchmark_ml_model_comparison,
    benchmark_memory_efficiency,
    benchmark_real_models,
    benchmark_tensor_stats
);
criterion_main!(ml_benches);
