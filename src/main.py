import modules.shared_buffer as shared_buffer
import cv2 as cv
from modules.StreamingManager import CameraStream

# Start the camera stream in a background thread
stream = CameraStream()
stream.start()



try:
    while True:
        # Retrieve the most recent frame from the shared buffer
        frame = shared_buffer.video_buffer.retrieve_frame()
        if frame is not None:
            cv.imshow("SmartSight Live Stream", frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
except KeyboardInterrupt:
    print("Shutting down...")
    stream.stop()