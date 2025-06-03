import cv2 as cv
from collections import Counter

class ObjectDetection:
    def __init__(self, model, minConf):
        self.model = model
        self.minConf = minConf

    def detect(self, img):
        # Dictionary to store object counts
        object_counts = Counter()
        conf = 0
        results = self.model(img)  # Run inference
        # Process results and count objects
        for result in results:
            for box in result.boxes:
                conf = box.conf
                print(f"Detected {self.model.names[int(box.cls[0])]}. Confidence: {conf}")
                if conf >= self.minConf:
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]
                    object_counts[class_name] += 1
        return object_counts 
