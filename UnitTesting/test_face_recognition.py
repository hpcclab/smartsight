import sys
import os
import cv2
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Add src to python path
sys.path.append(os.path.join(os.getcwd(), "src"))

from modules.facial_recognition_ai_module import FacialRecognitionAIModule

def test_recognition():
    # Path to config
    config_path = os.path.join(os.getcwd(), "src", "config", "config.yaml")
    
    # Initialize module
    print(f"Initializing module with config: {config_path}")
    try:
        recognizer = FacialRecognitionAIModule(config_path=config_path)
    except Exception as e:
        print(f"Failed to initialize module: {e}")
        return

    # Test 1: Black image (No faces)
    print("\nTest 1: Black image (No faces)")
    black_image = np.zeros((640, 640, 3), dtype=np.uint8)
    
    try:
        result = recognizer.execute("test_black_image", black_image)
        print(f"Result 1: {result}")
        
        if "No faces detected" in result or "No image data" in result or "No one was recognized" in result:
             print("Test 1 Passed (Graceful handling of no faces)")
        else:
             print(f"Test 1 Failed. Unexpected output: {result}")

    except Exception as e:
        print(f"Test 1 Error: {e}")

    # Test 2: Dummy Logic Check (if models load)
    # We can't easily synthesize a face that matches an encoding without the original image data used for encoding.
    # But we can check if the model loaded successfully by inspecting internal state if we wanted, 
    # or just rely on the logs printed during execution.
    
    if recognizer.detector is not None and recognizer.data is not None:
        print("\nTest 2: Models loaded successfully.")
    else:
        print("\nTest 2: Models failed to load (check logs).")

if __name__ == "__main__":
    test_recognition()
