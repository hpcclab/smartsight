import sys
import os
import time
import cv2
import pytest

# Setup path so src can be imported
_script_dir = os.path.dirname(__file__)
_project_root = os.path.abspath(os.path.join(_script_dir, '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
_src_dir = os.path.join(_project_root, 'src')
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from modules.shared_buffer import video_buffer
from src.modules.edge_cloud_merger import EdgeCloudMerger
from modules.ai_manager import AI_manager

def test_real_data_edge_cloud_fusion():
    print("\n==================================================")
    print("   REAL DATA TEST: RESPONSE FUSION ENGINE (ALG 1)  ")
    print("==================================================")

    image_path = os.path.join(_script_dir, "test.jpg")
    print(f"[REAL TEST] Loading real test image from: {image_path}")
    assert os.path.exists(image_path), f"Test image not found at {image_path}"

    frame = cv2.imread(image_path)
    assert frame is not None, "Failed to load test image with cv2.imread"
    print(f"[REAL TEST] Image loaded successfully. Resolution: {frame.shape[1]}x{frame.shape[0]}")

    # Load image frame into shared video buffer
    video_buffer.update(frame)
    print("[REAL TEST] Frame successfully populated in shared video_buffer.")

    # Initialize EdgeCloudMerger helper module
    merger = EdgeCloudMerger(ttsr=15.0, t_est=0.5)

    prompt = "Describe what you see in this image in 2-3 sentences."
    print(f"[REAL TEST] Sending prompt to Edge & Cloud VLMs: '{prompt}'")
    print("[REAL TEST] Starting real-time parallel inference and fusion streaming...\n")

    start_time = time.time()
    tokens = []
    
    try:
        for token in merger.run_inference(prompt, use_image=True, frame=frame):
            sys.stdout.write(token)
            sys.stdout.flush()
            tokens.append(token)
    except Exception as e:
        print(f"\n[REAL TEST ERROR] Exception during real inference: {e}")

    elapsed_time = time.time() - start_time
    full_response = "".join(tokens).strip()

    print("\n\n==================================================")
    print("         REAL DATA INFERENCE SUMMARY FINDINGS     ")
    print("==================================================")
    print(f"Total Response Time : {elapsed_time:.2f} seconds")
    print(f"Total Tokens Received: {len(tokens)}")
    print(f"Full Assembled Text :\n\"{full_response}\"")
    print("==================================================")

    # Basic assertions
    assert len(tokens) > 0, "No tokens were received during real data fusion test."
    assert len(full_response) > 0, "Full response string is empty."
    print("RESULT: REAL DATA RESPONSE FUSION TEST COMPLETED SUCCESSFULLY!\n")

if __name__ == "__main__":
    test_real_data_edge_cloud_fusion()
