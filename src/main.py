import modules.shared_buffer as shared_buffer
import cv2 as cv
from modules.StreamingManager import CameraStream
from modules.TTS_module import TTSModule
from modules.passive_detector_module import PassiveDetectorModule
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# TTS = TTSModule()
# TTS.execute("test_inference", "Hello, this is a test of the Piper text to speech module.")

# Start the camera stream in a background thread
stream = CameraStream()
stream.start()

passive_detector = PassiveDetectorModule()
passive_detector.start()

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
finally:
    passive_detector.stop()
    stream.stop()
    cv.destroyAllWindows()