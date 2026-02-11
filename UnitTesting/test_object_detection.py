import sys
import os
import cv2
import numpy as np
import logging




# Configure logging
logging.basicConfig(level=logging.INFO)

# Add src to python path
# os.chdir("..")
sys.path.append(os.path.join(os.getcwd(), "src"))

from modules.object_detection_ai_module import ObjectDetectionAIModule

from config.config import get_config

cfg = get_config()

print(cfg["yolo_vision"]["model_path"])
print(cfg["openrouter_api"]["model"])

def test_detection():
    # Path to config
    config_path = os.path.join(os.getcwd(), "src", "config", "config.yaml")
    
    # Initialize module
    print(f"Initializing module with config: {config_path}")
    try:
        detector = ObjectDetectionAIModule(config_path=config_path)
    except Exception as e:
        print(f"Failed to initialize module: {e}")
        return

    # Test 1: Black image (No objects)
    print("\nTest 1: Black image (No objects)")
    black_image = np.zeros((640, 640, 3), dtype=np.uint8)
    
    # Using 'execute' which calls 'run_inference'
    try:
        result = detector.execute("test_black_image", black_image)
        print(f"Result: {result}")
        
        if result == "No objects detected." or result == "No objects detected":
            print("Test 1 Passed!")
        else:
            print(f"Test 1 Failed. Expected 'No objects detected.', got '{result}'")
    except Exception as e:
        print(f"Test 1 Error: {e}")

# if __name__ == "__main__":
#     test_detection()
