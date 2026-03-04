import cv2 as cv
from collections import Counter
from ultralytics import YOLO
from .ai_module_base import BaseAIModel

class ObjectDetectionAIModule(BaseAIModel):
    def __init__(self):
        super().__init__("yolo_vision")

    def load_model(self):
        """Initializes the YOLO model using the path from config."""
        model_path = self.config.get("model_path", "yolov8s.pt")
        self.logger.info(f"Loading YOLO model from {model_path}...")
        self.model = YOLO(model_path)

    def run_inference(self, input_data: str, **kwargs) -> str:
        """
        Runs object detection on the input image path.
        Returns a string summary of detected objects.
        """
        try:
            # Check if input is a path or numpy array. 
            # The original code used cv.imread on a path.
            # Base class execute allows passing arbitrary input_data.
            # We assume input_data is a file path string or numpy array.
            
            frame = None
            if isinstance(input_data, str):
                frame = cv.imread(input_data)
            else:
                frame = input_data # Assume it's already an image array

            if frame is None or frame.size == 0:
                self.logger.warning("Empty frame received or could not load image.")
                return "No objects detected."

            # Ensure model is valid
            if self.model is None:
                self.load_model()

            yolo_results = self.model(frame, verbose=False)
            object_counts = Counter()
            min_conf = float(self.config.get("min_conf", 0.5))

            if yolo_results:
                for box in yolo_results[0].boxes:
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]
                    conf = box.conf.item()
                    
                    if conf >= min_conf:
                        object_counts[class_name] += 1
            
            if len(object_counts) < 1:
                return "No objects detected."

            output_parts = []
            for o, count in object_counts.items():
                if count > 1:
                    output_parts.append(f"{count} {o}")
                else:
                    output_parts.append(o)
            
            output = " " + ", ".join(output_parts)
            return output

        except Exception as e:
            self.logger.error(f"Error during inference: {e}")
            return "An unexpected error occurred during object detection."