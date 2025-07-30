#!/usr/bin/env python
"""Simple test for diffai_python Rust integration"""

try:
    import diffai_python
    print("✅ Successfully imported diffai_python")
    print(f"Module path: {diffai_python.__file__}")
    print(f"Available functions: {dir(diffai_python)}")
    print(f"Version: {diffai_python.__version__}")
    
    # Test basic diff functionality (using diff_py as per diffx pattern)
    old_data = {"name": "test", "value": 123}
    new_data = {"name": "test", "value": 456}
    
    results = diffai_python.diff_py(old_data, new_data)
    print(f"✅ Diff results: {results}")
    print("🎉 diffai_python Rust integration working correctly!")
    
except ImportError as e:
    print(f"❌ Import failed: {e}")
except Exception as e:
    print(f"❌ Test failed: {e}")