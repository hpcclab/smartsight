import sys
import os

import pytest
import pkgutil
import importlib
import numpy as np
_script_dir = os.path.dirname(__file__)
# Add project root to python path so src modules can be imported
_project_root = os.path.abspath(os.path.join(_script_dir, '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Many modules use imports like `from config.config import ...` or `from modules.shared_buffer import ...`
# which implies they expect the `src` directory itself to be in `sys.path`.
_src_dir = os.path.join(_project_root, 'src')
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

# ---------------------------------------------------------
# Test 1: Basic Module Import Check
# ---------------------------------------------------------
def get_modules():
    """Discover all modules inside the src.modules package."""
    try:
        import src.modules
    except ImportError:
        return []
    
    package = src.modules
    prefix = package.__name__ + "."
    modules = []
    
    for importer, modname, ispkg in pkgutil.iter_modules(package.__path__, prefix):
        modules.append(modname)
        
    return modules

@pytest.mark.parametrize("module_name", get_modules())
def test_module_import(module_name):
    """
    Module-by-module test to ensure each module can be imported 
    without syntax errors or missing dependencies.
    """
    try:
        importlib.import_module(module_name)
    except Exception as e:
        pytest.fail(f"Failed to import {module_name}: {e}")

# ---------------------------------------------------------
# Test 2: Face Recognition Unit Test (Modeled from src/tests)
# ---------------------------------------------------------
def test_face_recognition():
    """
    Test facial recognition module with a black image to ensure it handles 
    no-face scenarios gracefully, similar to src/tests/test_face_recognition.py
    """
    try:
        from src.modules.facial_recognition_ai_module import FacialRecognitionAIModule
    except ImportError as e:
        pytest.skip(f"Could not import FacialRecognitionAIModule: {e}")

    config_path = os.path.join(_project_root, "src", "config", "config.yaml")
    
    try:
        recognizer = FacialRecognitionAIModule()
    except Exception as e:
        pytest.fail(f"Failed to initialize module: {e}")

    # Test Black image (No faces)
    black_image = np.zeros((640, 640, 3), dtype=np.uint8)
    
    try:
        result = recognizer.execute("test_black_image", black_image)
        # Should gracefully handle no faces
        assert any(msg in result for msg in ["No faces detected", "No image data", "No one was recognized"]), f"Unexpected result: {result}"
    except Exception as e:
        pytest.fail(f"Face recognition execution failed: {e}")

# ---------------------------------------------------------
# Test 3: Object Detection Unit Test (Modeled from src/tests)
# ---------------------------------------------------------
def test_object_detection():
    """
    Test object detection module with a black image to ensure it handles 
    no-object scenarios correctly, similar to src/tests/test_object_detection.py
    """
    try:
        from src.modules.object_detection_ai_module import ObjectDetectionAIModule
    except ImportError as e:
        pytest.skip(f"Could not import ObjectDetectionAIModule: {e}")

    config_path = os.path.join(_project_root, "src", "config", "config.yaml")
    
    try:
        detector = ObjectDetectionAIModule()
    except Exception as e:
        pytest.fail(f"Failed to initialize module: {e}")

    # Test Black image (No objects)
    black_image = np.zeros((640, 640, 3), dtype=np.uint8)
    
    try:
        result = detector.execute("test_black_image", black_image)
        # Assuming the module returns "No objects detected." or something similar
        assert result in ["No objects detected.", "No objects detected"], f"Unexpected result: {result}"
    except Exception as e:
        pytest.fail(f"Object detection execution failed: {e}")

if __name__ == "__main__":
    # If the user runs `python Testing/pytest.py`, execute pytest on this file automatically
    sys.exit(pytest.main(["-v", __file__]))
