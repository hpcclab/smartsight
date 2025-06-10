import cv2 as cv
import os
import time

# Compute absolute paths
script_dir = os.path.dirname(os.path.abspath(__file__))
temp_dir = os.path.abspath(os.path.join(script_dir, '..', 'temp'))
os.makedirs(temp_dir, exist_ok=True)

# Output filename in temp
output_file = os.path.join(temp_dir, 'image.jpg')

# Open default camera (0)
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Error: Unable to open the camera.")
    exit(1)

# Set desired capture rate (frames per second)
fps = 2  # capture 2 frames per second (slightly faster)
interval = 1.0 / fps

print(f"Capturing live camera frames to {output_file} at ~{fps} fps. Press Ctrl+C to stop.")
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Warning: failed to read frame from camera.")
            time.sleep(interval)
            continue

        # Optional: rotate if your downstream code expects portrait orientation
        # frame = cv.rotate(frame, cv.ROTATE_90_COUNTERCLOCKWISE)

        # Write out to temp/image.jpg
        cv.imwrite(output_file, frame)
        print(f"Captured frame at {time.strftime('%H:%M:%S')}")

        # Wait before next capture
        time.sleep(interval)
except KeyboardInterrupt:
    print("Stopping live camera capture.")
finally:
    cap.release() 
