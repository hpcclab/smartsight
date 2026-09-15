import threading
import time
import logging
from modules.shared_buffer import video_buffer
from modules.object_detection_ai_module import ObjectDetectionAIModule
from modules.dollar_detection_ai_module import DollarDetectionAIModule
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
        self.passive_config = self.config.get("passive_detection", {})
        self.detection_cooldown = self.passive_config.get("detection_cooldown", 5) # seconds
        self.object_detection_rate = self.passive_config.get("object_detection_max_rate", 1.0)
        self.facial_recognition_rate = self.passive_config.get("facial_recognition_max_rate", 1.0)
        self.text_detection_rate = self.passive_config.get("text_detection_max_rate", 1.0)
        self.dollar_detection_enabled = self.passive_config.get("dollar_detection_enabled", False)
        self.dollar_detection_rate = self.passive_config.get("dollar_detection_max_rate", 1.0)

        from modules.ocr_module import OCRModule
        from modules.facial_recognition_ai_module import FacialRecognitionAIModule

        self.main_tasks = [
            {"name": "object_detection", "module_class": ObjectDetectionAIModule, "max_rate": self.object_detection_rate, "func": self._run_object_detection},
            {"name": "facial_recognition", "module_class": FacialRecognitionAIModule, "max_rate": self.facial_recognition_rate, "func": self._run_facial_recognition},
            {"name": "text_detection", "module_class": OCRModule, "max_rate": self.text_detection_rate, "func": self._run_text_detection}
        ]
        self.last_completed_time = {
            "object_detection": 0.0,
            "facial_recognition": 0.0,
            "text_detection": 0.0
        }

        if self.dollar_detection_enabled:
            self.main_tasks.append(
                {"name": "dollar_detection", "module_class": DollarDetectionAIModule, "max_rate": self.dollar_detection_rate, "func": self._run_dollar_detection}
            )
            self.last_completed_time["dollar_detection"] = 0.0

        self.hold_list = []

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
            if not self.main_tasks or len(self.main_tasks) == 0:
                time.sleep(0.1)
                if self.hold_list:
                    for task in self.hold_list:
                        time_left = (self.last_completed_time[task["name"]] + task["max_rate"]) - time.time()
                        if time_left <= 0:
                            self.main_tasks.append(task)
                            self.hold_list.remove(task)
                continue

            current_task = self.main_tasks.pop(0)
            task_name = current_task["name"]
            
            # Check if the task is within its max_rate, otherwise shelve it and continue to next task for now.
            if time.time() - self.last_completed_time[task_name] < current_task["max_rate"]:
                self.hold_list.append(current_task)
                continue
                
            frame = video_buffer.retrieve_frame()
            if frame is not None:
                try:
                    print(f"Running {task_name} on frame")
                    current_task["func"](frame, current_task["module_class"])
                except Exception as e:
                    self.logger.error(f"Detection loop error for {task_name}: {e}")
                    
            self.last_completed_time[task_name] = time.time()
            self.main_tasks.append(current_task)
            
            if self.hold_list:
                self.main_tasks = self.hold_list + self.main_tasks
                self.hold_list = []
            
            # Small sleep to prevent tight looping when inference is fast or frames are missing
            time.sleep(0.1)

    def _run_object_detection(self, frame, module_class):
        result = AI_manager.execute_module(lambda m: isinstance(m, module_class), "detect", frame)
        if result and result not in ("No objects detected.", "An unexpected error occurred during object detection."):
            filtered_result = self._filter_recent_detections(result, "object")
            if filtered_result:
                self.logger.info(f"Passive Object Detection Found: {filtered_result}")
                self.global_response.add_message(f"{filtered_result}", priority=50)

    def _run_facial_recognition(self, frame, module_class):
        result = AI_manager.execute_module(lambda m: isinstance(m, module_class), "detect", frame)
        if result and result not in ("No faces detected.", "An error occurred during facial recognition.", "Facial recognition models are not loaded.", "No image data.", "No one was recognized."):
            filtered_result = self._filter_recent_detections(result.replace("Detected ", ""), "face")
            if filtered_result:
                self.logger.info(f"Passive Face Detection Found: {filtered_result}")
                self.global_response.add_message(f"{filtered_result}", priority=50)

    def _run_dollar_detection(self, frame, module_class):
        result = AI_manager.execute_module(lambda m: isinstance(m, module_class), "detect", frame)
        if result and result not in ("No dollar bills detected.", "An unexpected error occurred during dollar detection."):
            filtered_result = self._filter_recent_detections(result, "dollar")
            if filtered_result:
                self.logger.info(f"Passive Dollar Detection Found: {filtered_result}")
                self.global_response.add_message(f"{filtered_result}", priority=50)

    def _run_text_detection(self, frame, module_class):
        result = AI_manager.execute_module(lambda m: isinstance(m, module_class), "detect", frame, simple_text_detection=True)
        if result and result != "No text detected.":
            filtered_result = self._filter_recent_detections(result, "text")
            if filtered_result:
                self.logger.info(f"Passive Text Detection Found: {filtered_result}")
                self.global_response.add_message(f"{filtered_result}", priority=50)

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

    def _filter_recent_detections(self, result_str, detection_type="object"):
        """
        Implements the logic:
        Parse input, cull out old detections, find new items, find increased count items, update timestamps, and add new items, and return filtered string to Send to Response Manager
        """
        current_time = time.time()
        
        # We namespace the detections by type to prevent overlap
        
        # Cull out old detections
        keys_to_remove = []
        for label, data in self.recent_detections.items():
            if data.get("type") == detection_type and current_time - data["timestamp"] >= self.detection_cooldown:
                keys_to_remove.append(label)
                # print(f"Removing old detection: {label}")
        for label in keys_to_remove:
            del self.recent_detections[label]

        # Parse current detections
        if detection_type in ("object", "face", "dollar"):
            current_detections = self._parse_detections(result_str)
        else:
            # Handle text generically, maybe just counting instances
            current_detections = {result_str.strip(): 1} if result_str.strip() else {}
        
        report_detections = {}

        for label, count in current_detections.items():
            namespaced_label = f"{detection_type}_{label}"
            # Find New items
            if namespaced_label not in self.recent_detections:
                report_detections[label] = count
                self.recent_detections[namespaced_label] = {"count": count, "timestamp": current_time, "type": detection_type}
            else:
                # Find increased count items
                prev_count = self.recent_detections[namespaced_label]["count"]
                if count > prev_count:
                    # Report the new total count seen
                    report_detections[label] = count
                    # print(f"Increased count for {label}: {count}")
                # Update Timestamps, and counts
                self.recent_detections[namespaced_label]["timestamp"] = current_time
                self.recent_detections[namespaced_label]["count"] = count

        if report_detections:
            if detection_type == "text":
                return ", ".join(report_detections.keys())
            return self._format_detections(report_detections)
        return ""
