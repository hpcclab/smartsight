import cv2
import os
import numpy as np
from DollarDetection import DollarBillDetection

def test_model_loading():
    """Test if model loads successfully"""
    print("\n=== Testing Model Loading ===")
    try:
        detector = DollarBillDetection(
            model_path="C:/Users/Crack/2025_AI/DeepLearning/DollarDetection/runs/detect/train/weights/best.pt",
            name="DollarDetector",
            conf=0.5
        )
        print("✓ Model loaded successfully")
        print(f"  Model name: {detector.name}")
        print(f"  Confidence threshold: {detector.conf}")
        return detector
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return None

def test_single_image(detector, image_path):
    """Test detection on a single image"""
    print(f"\n=== Testing Detection on Image ===")
    print(f"Image path: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"✗ Image not found: {image_path}")
        return
    
    try:
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"✗ Failed to read image")
            return
        
        print(f"✓ Image loaded: {frame.shape}")
        
        # Run detection
        detections = detector.detect(frame)
        
        print(f"✓ Detection complete")
        print(f"  Detections found: {len(detections)}")
        for i, detection in enumerate(detections):
            print(f"    {i+1}. {detection}")
        
        return detections
    except Exception as e:
        print(f"✗ Detection failed: {e}")
        return None

def test_video_stream(detector, video_path=None, frames_to_test=10):
    """Test detection on video stream or webcam"""
    print(f"\n=== Testing Video Stream ===")
    
    if video_path and not os.path.exists(video_path):
        print(f"✗ Video file not found: {video_path}")
        return
    
    try:
        # Use webcam if no video path provided
        if video_path:
            cap = cv2.VideoCapture(video_path)
            print(f"✓ Video loaded: {video_path}")
        else:
            cap = cv2.VideoCapture(0)
            print(f"✓ Webcam opened")
        
        if not cap.isOpened():
            print("✗ Failed to open video source")
            return
        
        frame_count = 0
        detections_log = []
        
        while frame_count < frames_to_test:
            ret, frame = cap.read()
            if not ret:
                print("✗ Failed to read frame")
                break
            
            # Run detection
            detections = detector.detect(frame)
            detections_log.append(detections)
            
            # Display frame with detections
            cv2.putText(frame, f"Detections: {len(detections)}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Dollar Detection", frame)
            
            frame_count += 1
            print(f"  Frame {frame_count}: {len(detections)} detections")
            
            # Press 'q' to quit early
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"✓ Video test complete ({frame_count} frames tested)")
        return detections_log
    
    except Exception as e:
        print(f"✗ Video stream test failed: {e}")
        return None

def test_confidence_threshold(detector, image_path):
    """Test different confidence thresholds"""
    print(f"\n=== Testing Confidence Thresholds ===")
    
    if not os.path.exists(image_path):
        print(f"✗ Image not found: {image_path}")
        return
    
    try:
        frame = cv2.imread(image_path)
        thresholds = [0.3, 0.5, 0.7, 0.9]
        
        for threshold in thresholds:
            detector.conf = threshold
            detections = detector.detect(frame)
            print(f"  Confidence {threshold}: {len(detections)} detections")
        
        print(f"✓ Threshold testing complete")
    except Exception as e:
        print(f"✗ Threshold test failed: {e}")

def display_menu():
    """Display test menu"""
    print("\n=== Dollar Detection Test Menu ===")
    print("1. Test Model Loading")
    print("2. Test Single Image Detection")
    print("3. Test Video Stream / Webcam")
    print("4. Test Confidence Thresholds")
    print("5. Run All Tests")
    print("0. Exit")
    print("==================================")
    return input("Select an option: ")

def main():
    detector = None
    
    while True:
        choice = display_menu()
        
        if choice == '1':
            detector = test_model_loading()
        
        elif choice == '2':
            if detector is None:
                print("✗ Model not loaded. Please test model loading first (option 1).")
            else:
                image_path = input("Enter image path (or press Enter for a sample): ")
                if not image_path:
                    # Try to find a sample image in common locations
                    image_path = "C:/Users/Crack/OneDrive/Documents/GitHub/smartsight/simulate/images/test.jpg"
                test_single_image(detector, image_path)
        
        elif choice == '3':
            if detector is None:
                print("✗ Model not loaded. Please test model loading first (option 1).")
            else:
                video_path = input("Enter video path (or press Enter to use webcam): ")
                if not video_path:
                    test_video_stream(detector, video_path=None, frames_to_test=30)
                else:
                    test_video_stream(detector, video_path=video_path, frames_to_test=100)
        
        elif choice == '4':
            if detector is None:
                print("✗ Model not loaded. Please test model loading first (option 1).")
            else:
                image_path = input("Enter image path: ")
                test_confidence_threshold(detector, image_path)
        
        elif choice == '5':
            print("\n=== Running All Tests ===")
            detector = test_model_loading()
            if detector:
                print("\n(Skipping video/image tests - run manually as needed)")
                print("To complete full testing:")
                print("  - Use option 2 to test with real images")
                print("  - Use option 3 to test with video/webcam")
        
        elif choice == '0':
            print("Exiting... Goodbye!")
            break
        
        else:
            print("✗ Invalid option. Please try again.")

if __name__ == "__main__":
    main()
