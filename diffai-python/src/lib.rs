#![allow(clippy::useless_conversion)]

use diffai_core::{diff, DiffOptions, DiffResult, DiffaiSpecificOptions, OutputFormat};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use serde_json::Value;

/// Convert Python object to serde_json::Value
fn python_to_value(_py: Python, obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    if obj.is_none() {
        Ok(Value::Null)
    } else if let Ok(b) = obj.extract::<bool>() {
        Ok(Value::Bool(b))
    } else if let Ok(i) = obj.extract::<i64>() {
        Ok(Value::Number(serde_json::Number::from(i)))
    } else if let Ok(f) = obj.extract::<f64>() {
        if let Some(n) = serde_json::Number::from_f64(f) {
            Ok(Value::Number(n))
        } else {
            Ok(Value::Null)
        }
    } else if let Ok(s) = obj.extract::<String>() {
        Ok(Value::String(s))
    } else if obj.is_instance_of::<pyo3::types::PyList>() {
        let list = obj.downcast::<pyo3::types::PyList>()?;
        let mut vec = Vec::new();
        for item in list.iter() {
            vec.push(python_to_value(_py, &item)?);
        }
        Ok(Value::Array(vec))
    } else if obj.is_instance_of::<pyo3::types::PyDict>() {
        let dict = obj.downcast::<pyo3::types::PyDict>()?;
        let mut map = serde_json::Map::new();
        for (key, value) in dict.iter() {
            let key_str = key.extract::<String>()?;
            map.insert(key_str, python_to_value(_py, &value)?);
        }
        Ok(Value::Object(map))
    } else {
        // Try to convert to string as fallback
        Ok(Value::String(obj.str()?.extract::<String>()?))
    }
}

/// Convert serde_json::Value to Python object
fn value_to_python(py: Python, value: &Value) -> PyResult<PyObject> {
    match value {
        Value::Null => Ok(py.None()),
        Value::Bool(b) => Ok(b.to_object(py)),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i.to_object(py))
            } else if let Some(f) = n.as_f64() {
                Ok(f.to_object(py))
            } else {
                Ok(py.None())
            }
        }
        Value::String(s) => Ok(s.to_object(py)),
        Value::Array(arr) => {
            let py_list = pyo3::types::PyList::empty_bound(py);
            for item in arr {
                py_list.append(value_to_python(py, item)?)?;
            }
            Ok(py_list.to_object(py))
        }
        Value::Object(obj) => {
            let py_dict = pyo3::types::PyDict::new_bound(py);
            for (key, value) in obj {
                py_dict.set_item(key, value_to_python(py, value)?)?;
            }
            Ok(py_dict.to_object(py))
        }
    }
}

/// Convert DiffResult to Python dictionary
fn diff_result_to_python(py: Python, result: &DiffResult) -> PyResult<PyObject> {
    let dict = pyo3::types::PyDict::new_bound(py);

    match result {
        DiffResult::Added(path, value) => {
            dict.set_item("type", "Added")?;
            dict.set_item("path", path)?;
            dict.set_item("value", value_to_python(py, value)?)?;
        }
        DiffResult::Removed(path, value) => {
            dict.set_item("type", "Removed")?;
            dict.set_item("path", path)?;
            dict.set_item("value", value_to_python(py, value)?)?;
        }
        DiffResult::Modified(path, old_val, new_val) => {
            dict.set_item("type", "Modified")?;
            dict.set_item("path", path)?;
            dict.set_item("old_value", value_to_python(py, old_val)?)?;
            dict.set_item("new_value", value_to_python(py, new_val)?)?;
        }
        DiffResult::TypeChanged(path, old_val, new_val) => {
            dict.set_item("type", "TypeChanged")?;
            dict.set_item("path", path)?;
            dict.set_item("old_value", value_to_python(py, old_val)?)?;
            dict.set_item("new_value", value_to_python(py, new_val)?)?;
        }
        // AI/ML specific diff results
        DiffResult::TensorShapeChanged(path, old_shape, new_shape) => {
            dict.set_item("type", "TensorShapeChanged")?;
            dict.set_item("path", path)?;
            dict.set_item("old_shape", old_shape.to_object(py))?;
            dict.set_item("new_shape", new_shape.to_object(py))?;
        }
        DiffResult::TensorDataChanged(path, old_norm, new_norm) => {
            dict.set_item("type", "TensorDataChanged")?;
            dict.set_item("path", path)?;
            dict.set_item("old_norm", old_norm)?;
            dict.set_item("new_norm", new_norm)?;
        }
        DiffResult::ModelArchitectureChanged(path, old_desc, new_desc) => {
            dict.set_item("type", "ModelArchitectureChanged")?;
            dict.set_item("path", path)?;
            dict.set_item("old_description", old_desc)?;
            dict.set_item("new_description", new_desc)?;
        }
        DiffResult::WeightSignificantChange(path, magnitude) => {
            dict.set_item("type", "WeightSignificantChange")?;
            dict.set_item("path", path)?;
            dict.set_item("magnitude", magnitude)?;
        }
        DiffResult::ActivationFunctionChanged(path, old_fn, new_fn) => {
            dict.set_item("type", "ActivationFunctionChanged")?;
            dict.set_item("path", path)?;
            dict.set_item("old_function", old_fn)?;
            dict.set_item("new_function", new_fn)?;
        }
        DiffResult::LearningRateChanged(path, old_lr, new_lr) => {
            dict.set_item("type", "LearningRateChanged")?;
            dict.set_item("path", path)?;
            dict.set_item("old_learning_rate", old_lr)?;
            dict.set_item("new_learning_rate", new_lr)?;
        }
        DiffResult::OptimizerChanged(path, old_opt, new_opt) => {
            dict.set_item("type", "OptimizerChanged")?;
            dict.set_item("path", path)?;
            dict.set_item("old_optimizer", old_opt)?;
            dict.set_item("new_optimizer", new_opt)?;
        }
        DiffResult::LossChange(path, old_loss, new_loss) => {
            dict.set_item("type", "LossChange")?;
            dict.set_item("path", path)?;
            dict.set_item("old_loss", old_loss)?;
            dict.set_item("new_loss", new_loss)?;
        }
        DiffResult::AccuracyChange(path, old_acc, new_acc) => {
            dict.set_item("type", "AccuracyChange")?;
            dict.set_item("path", path)?;
            dict.set_item("old_accuracy", old_acc)?;
            dict.set_item("new_accuracy", new_acc)?;
        }
        DiffResult::ModelVersionChanged(path, old_version, new_version) => {
            dict.set_item("type", "ModelVersionChanged")?;
            dict.set_item("path", path)?;
            dict.set_item("old_version", old_version)?;
            dict.set_item("new_version", new_version)?;
        }
    }

    Ok(dict.to_object(py))
}

/// Unified diff function for Python
///
/// Compare two Python objects or values and return differences with AI/ML specific analysis.
///
/// # Arguments
///
/// * `old` - The old value (Python object, list, or primitive)
/// * `new` - The new value (Python object, list, or primitive)  
/// * `**kwargs` - Optional keyword arguments for configuration
///
/// # Returns
///
/// List of difference dictionaries with AI/ML specific difference types
///
/// # Example
///
/// ```python
/// import diffai
///
/// old = {"model": {"layers": [{"type": "dense", "units": 128}]}}
/// new = {"model": {"layers": [{"type": "dense", "units": 256}]}}
/// result = diffai.diff(old, new)
/// print(result)  # [{"type": "Modified", "path": "model.layers[0].units", "old_value": 128, "new_value": 256}]
/// ```
#[pyfunction]
#[pyo3(signature = (old, new, **kwargs))]
fn diff_py(
    py: Python,
    old: &Bound<'_, PyAny>,
    new: &Bound<'_, PyAny>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<PyObject> {
    // Convert Python objects to serde_json::Value
    let old_value = python_to_value(py, old)?;
    let new_value = python_to_value(py, new)?;

    // Build options from kwargs
    let mut options = DiffOptions::default();
    let mut diffai_options = DiffaiSpecificOptions::default();
    let mut has_diffai_options = false;

    if let Some(kwargs) = kwargs {
        for (key, value) in kwargs.iter() {
            let key_str = key.extract::<String>()?;
            match key_str.as_str() {
                "epsilon" => {
                    if let Ok(eps) = value.extract::<f64>() {
                        options.epsilon = Some(eps);
                    }
                }
                "array_id_key" => {
                    if let Ok(key) = value.extract::<String>() {
                        options.array_id_key = Some(key);
                    }
                }
                "ignore_keys_regex" => {
                    if let Ok(pattern) = value.extract::<String>() {
                        options.ignore_keys_regex =
                            Some(regex::Regex::new(&pattern).map_err(|e| {
                                pyo3::exceptions::PyValueError::new_err(format!(
                                    "Invalid regex: {e}"
                                ))
                            })?);
                    }
                }
                "path_filter" => {
                    if let Ok(filter) = value.extract::<String>() {
                        options.path_filter = Some(filter);
                    }
                }
                "output_format" => {
                    if let Ok(format_str) = value.extract::<String>() {
                        let format = OutputFormat::parse_format(&format_str).map_err(|e| {
                            pyo3::exceptions::PyValueError::new_err(format!(
                                "Invalid format: {e}"
                            ))
                        })?;
                        options.output_format = Some(format);
                    }
                }
                "show_unchanged" => {
                    if let Ok(show) = value.extract::<bool>() {
                        options.show_unchanged = Some(show);
                    }
                }
                "show_types" => {
                    if let Ok(show) = value.extract::<bool>() {
                        options.show_types = Some(show);
                    }
                }
                "use_memory_optimization" => {
                    if let Ok(opt) = value.extract::<bool>() {
                        options.use_memory_optimization = Some(opt);
                    }
                }
                "batch_size" => {
                    if let Ok(size) = value.extract::<usize>() {
                        options.batch_size = Some(size);
                    }
                }
                // diffai-specific options
                "ml_analysis_enabled" => {
                    if let Ok(analysis) = value.extract::<bool>() {
                        diffai_options.ml_analysis_enabled = Some(analysis);
                        has_diffai_options = true;
                    }
                }
                "tensor_comparison_mode" => {
                    if let Ok(mode) = value.extract::<String>() {
                        diffai_options.tensor_comparison_mode = Some(mode);
                        has_diffai_options = true;
                    }
                }
                "model_format" => {
                    if let Ok(format) = value.extract::<String>() {
                        diffai_options.model_format = Some(format);
                        has_diffai_options = true;
                    }
                }
                "scientific_precision" => {
                    if let Ok(precision) = value.extract::<bool>() {
                        diffai_options.scientific_precision = Some(precision);
                        has_diffai_options = true;
                    }
                }
                "weight_threshold" => {
                    if let Ok(threshold) = value.extract::<f64>() {
                        diffai_options.weight_threshold = Some(threshold);
                        has_diffai_options = true;
                    }
                }
                "activation_analysis" => {
                    if let Ok(analysis) = value.extract::<bool>() {
                        diffai_options.activation_analysis = Some(analysis);
                        has_diffai_options = true;
                    }
                }
                "learning_rate_tracking" => {
                    if let Ok(tracking) = value.extract::<bool>() {
                        diffai_options.learning_rate_tracking = Some(tracking);
                        has_diffai_options = true;
                    }
                }
                "optimizer_comparison" => {
                    if let Ok(comparison) = value.extract::<bool>() {
                        diffai_options.optimizer_comparison = Some(comparison);
                        has_diffai_options = true;
                    }
                }
                "loss_tracking" => {
                    if let Ok(tracking) = value.extract::<bool>() {
                        diffai_options.loss_tracking = Some(tracking);
                        has_diffai_options = true;
                    }
                }
                "accuracy_tracking" => {
                    if let Ok(tracking) = value.extract::<bool>() {
                        diffai_options.accuracy_tracking = Some(tracking);
                        has_diffai_options = true;
                    }
                }
                "model_version_check" => {
                    if let Ok(check) = value.extract::<bool>() {
                        diffai_options.model_version_check = Some(check);
                        has_diffai_options = true;
                    }
                }
                _ => {
                    // Ignore unknown options
                }
            }
        }
    }

    if has_diffai_options {
        options.diffai_options = Some(diffai_options);
    }

    // Perform diff
    let results = diff(&old_value, &new_value, Some(&options))
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Diff error: {e:?}")))?;

    // Convert results to Python objects
    let py_list = pyo3::types::PyList::empty_bound(py);
    for result in results {
        py_list.append(diff_result_to_python(py, &result)?)?;
    }

    Ok(py_list.to_object(py))
}

/// A Python module for AI/ML focused structured data comparison
#[pymodule]
fn diffai_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(diff_py, m)?)?;

    // Add module-level diff function for easier access
    m.add("diff", m.getattr("diff_py")?)?;

    Ok(())
}
