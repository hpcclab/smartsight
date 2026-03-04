import threading
import time
import logging
from modules.shared_buffer import video_buffer
from modules.object_detection_ai_module import ObjectDetectionAIModule

class PassiveDetectorModule:
    """
    Runs the ObjectDetectionAIModule continuously on frames from the shared buffer.
    """
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.detector = ObjectDetectionAIModule()
        self.running = False
        self.thread = None

    def start(self):
        """Starts the passive detection thread."""
        if not self.running:
            self.running = True
            self.logger.info("Starting PassiveDetectorModule thread...")
            self.thread = threading.Thread(target=self._detection_loop, daemon=True)
            self.thread.start()

    def stop(self):
        """Stops the passive detection thread."""
        if self.running:
            self.running = False
            self.logger.info("Stopping PassiveDetectorModule thread...")
            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=2.0)

    def _detection_loop(self):
        """Continuous loop to run inference on retrieved frames."""
        while self.running:
            frame = video_buffer.retrieve_frame()
            if frame is not None:
                # pass 'detect' as the input_key to the execute method
                try:
                    result = self.detector.execute("detect", frame)
                    if result and result not in ("No objects detected.", "An unexpected error occurred during object detection."):
                        self.logger.info(f"Passive Detection Found: {result}")
                except Exception as e:
                    self.logger.error(f"Detection loop error: {e}")
            
            # Small sleep to prevent tight looping when inference is fast or frames are missing
            time.sleep(0.01)
