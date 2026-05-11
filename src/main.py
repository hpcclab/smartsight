import modules.shared_buffer as shared_buffer
import cv2 as cv
from modules.StreamingManager import CameraStream
from modules.global_response_module import GlobalResponseModule
from modules.passive_detector_module import PassiveDetectorModule
from modules.active_module import ActiveModule
from modules.ai_manager import AI_manager
from modules.input_event_manager import InputEventManager
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Start the camera stream in a background thread
stream = CameraStream()
stream.start()

# Load all AI models
AI_manager.load_all_models()

# Start the global response module
global_response = GlobalResponseModule()
global_response.start()

# Start the passive detector, passing the global response module instance
passive_detector = PassiveDetectorModule(global_response)
passive_detector.start()

# Start the active module
active_module = ActiveModule(global_response)

# Start the Input Event Manager
input_manager = InputEventManager(active_module=active_module)
input_manager.start()

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
    input_manager.stop()
    passive_detector.stop()
    global_response.stop()
    stream.stop()
    cv.destroyAllWindows()
