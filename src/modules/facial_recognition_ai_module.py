import cv2 as cv
import face_recognition
import pickle
import os
from .ai_module_base import BaseAIModel

class FacialRecognitionAIModule(BaseAIModel):
    def __init__(self, config_path: str = "../config/config.yaml"):
        super().__init__(config_path, "face_recognition")
        self.detector = None
        self.data = None

    def load_model(self):
        """Loads the face detection cascade and recognition encodings."""
        try:
            cascade_path = self.config.get("cascade_path", "models/facial_recognition/haarcascade_frontalface_default.xml")
            encodings_path = self.config.get("encodings_path", "models/facial_recognition/encodings.pickle")

            self.logger.info(f"Loading Face Cascade from {cascade_path}")
            if not os.path.exists(cascade_path):
                self.logger.error(f"Cascade file not found at {cascade_path}")
                raise FileNotFoundError(f"Cascade file not found at {cascade_path}")

            self.detector = cv.CascadeClassifier(cascade_path)
            if self.detector.empty():
                self.logger.error("Failed to load CascadeClassifier")
                raise IOError("Failed to load CascadeClassifier")

            self.logger.info(f"Loading Encodings from {encodings_path}")
            if not os.path.exists(encodings_path):
                self.logger.error(f"Encodings file not found at {encodings_path}")
                raise FileNotFoundError(f"Encodings file not found at {encodings_path}")

            with open(encodings_path, "rb") as f:
                self.data = pickle.load(f)

        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            self.detector = None
            self.data = None

    def run_inference(self, input_data: str, **kwargs) -> str:
        """
        Recognizes faces in the input image.
        Returns a string listing recognised people.
        """
        # Ensure models are loaded
        if self.detector is None or self.data is None:
            self.load_model()
            if self.detector is None or self.data is None:
                return "Facial recognition models are not loaded."

        try:
            frame = None
            if isinstance(input_data, str):
                frame = cv.imread(input_data)
            else:
                frame = input_data
            
            if frame is None or frame.size == 0:
                self.logger.warning("Empty frame received.")
                return "No image data."

            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            # Detect faces
            rects = self.detector.detectMultiScale(
                gray, scaleFactor=1.3, minNeighbors=6, 
                minSize=(40, 40), flags=cv.CASCADE_SCALE_IMAGE
            )
            
            if len(rects) == 0:
                return "No faces detected."

            boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]
            encodings = face_recognition.face_encodings(rgb, boxes)
            
            names = []
            tolerance = float(self.config.get("tolerance", 0.5))
            
            for encoding in encodings:
                distances = face_recognition.face_distance(self.data["encodings"], encoding)
                matches = [d <= tolerance for d in distances]
                
                name = "Unknown"
                if True in matches:
                    matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                    counts = {}
                    for i in matchedIdxs:
                        name = self.data["names"][i]
                        counts[name] = counts.get(name, 0) + 1
                    
                    name = max(counts, key=counts.get)
                
                names.append(name)

            if not names:
                return "No one was recognized."

            # Format output logic similar to the deprecated manager
            # If all are Unknown, say "No one was recognized."
            # Otherwise list names.
            known_names = [n for n in names if n != "Unknown"]
            
            if not known_names:
                 return "No one was recognized."
            
            # Remove duplicates for the announcement while preserving order or count?
            # The deprecated code lists names: "Detected Name1, Name2, "
            # Let's match that format but maybe clean up the trailing comma logic
            
            output = "Detected " + ", ".join(known_names)
            return output

        except Exception as e:
            self.logger.error(f"Error during face recognition inference: {e}")
            return "An error occurred during facial recognition."