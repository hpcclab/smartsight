import threading
import time
import logging
from modules.shared_buffer import video_buffer
from modules.object_detection_ai_module import ObjectDetectionAIModule
from modules.ai_manager import AI_manager
from config.config import get_config

class PassiveDetectorModule:
    """
    Runs the ObjectDetectionAIModule continuously on frames from the shared buffer.
    """
    def __init__(self, global_response):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.global_response = global_response
        self.running = False
        self.thread = None
        # Recent detections list to prevent spamming the same detection
        # Dictionary mapping object_name -> {"count": int, "timestamp": float}
        self.recent_detections = {}
        self.config = get_config()
        self.detection_cooldown = self.config.get("passive_detection", {}).get("detection_cooldown", 5) # seconds

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
                    result = AI_manager.execute_module(lambda m: isinstance(m, ObjectDetectionAIModule), "detect", frame)
                    if result and result not in ("No objects detected.", "An unexpected error occurred during object detection."):
                        filtered_result = self._filter_recent_detections(result)
                        if filtered_result:
                            self.logger.info(f"Passive Detection Found: {filtered_result}")
                            self.global_response.add_message(f"{filtered_result}", priority=50)
                except Exception as e:
                    self.logger.error(f"Detection loop error: {e}")
            
            # Small sleep to prevent tight looping when inference is fast or frames are missing
            time.sleep(0.01)

    def _parse_detections(self, result_str):
        """Parses a string like '2 persons, 1 dog' into a dictionary format."""
        detections = {}
        if not result_str:
            return detections
        
        parts = result_str.split(',')
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            # Format is typically "count label" or just "label"
            words = part.split(' ', 1)
            count = 1
            label = part
            if len(words) == 2 and words[0].isdigit():
                count = int(words[0])
                label = words[1].strip()
                
            # Basic normalization (e.g., remove trailing 's' for plural, though naive)
            if label.endswith('s') and label != 'bus': # simple heuristic, might need improvement based on actual model output
                 label = label[:-1]
            
            detections[label] = detections.get(label, 0) + count
        return detections

    def _format_detections(self, detections_dict):
         """Formats dictionary back to string."""
         parts = []
         for label, count in detections_dict.items():
              if count > 1:
                   parts.append(f"{count} {label}s")
              else:
                   parts.append(f"1 {label}")
         return ", ".join(parts)

    def _filter_recent_detections(self, result_str):
        """
        Implements the logic:
        1. Parse input
        2. Cull out Old detections
        3. Find New items
        4. Find increased count items
        5. Update Timestamps, and add new items
        6. Return filtered string to Send to Response Manager
        """
        current_time = time.time()
        
        # 1. Cull out Old detections
        keys_to_remove = []
        for label, data in self.recent_detections.items():
            if current_time - data["timestamp"] >= self.detection_cooldown:
                keys_to_remove.append(label)
        for label in keys_to_remove:
            del self.recent_detections[label]

        # Parse current detections
        current_detections = self._parse_detections(result_str)
        
        report_detections = {}

        for label, count in current_detections.items():
            # 2. Find New items
            if label not in self.recent_detections:
                report_detections[label] = count
                self.recent_detections[label] = {"count": count, "timestamp": current_time}
            else:
                # 3. Find increased count items
                prev_count = self.recent_detections[label]["count"]
                if count > prev_count:
                    # Report only the difference or the new total? Let's report the new total count seen
                    report_detections[label] = count
                    
                # 4. Update Timestamps, and counts
                self.recent_detections[label]["timestamp"] = current_time
                self.recent_detections[label]["count"] = count

        if report_detections:
             return self._format_detections(report_detections)
        return ""
