import numpy 
import cv2 
import os 
from ultralytics import YOLO

class DollarBillDetection:
    
    def __init__(self, model_path, name, conf=0.5):
        self.model = YOLO(model_path)
        self.conf = conf 
        self.name = name 
    
    def detect(self, frame):
        results = self.model.predict(frame, conf=self.conf, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls)
                label = self.model.name[cls]
                conf = float(box.conf)
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                msg = f"{label.capitalize()} detected"
                detections.append(msg)
        return detections 
    

