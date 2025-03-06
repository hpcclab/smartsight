import cv2 as cv
import os
import time

from ultralytics import YOLO
import pyttsx3 
from collections import Counter

# Load the YOLOv8 model (pretrained on COCO dataset)
model = YOLO("yolov8s.pt")

# Load an image
image_path = "image.jpg"

# minimum model confidence to claim a detected item.
minConf = 0.75

# After seeing something, if the confidence is >= this value, it'll continue remembering it.
# PermConf = 0.55

# Number of frames to remember an object for.
ObjectPerminanceFrames = 3

# Initialize text-to-speech engine
engine = pyttsx3.init()
voices = engine.getProperty('voices')
# Set a specific voice if needed (e.g., first available voice)
engine.setProperty('voice', voices[1].id)

# Adjust volume
engine.setProperty('volume', 1.0)

# Adjust speaking rate
engine.setProperty('rate', 150)

file = "image.jpg"

MaxTime = time.time() + 120

class objectInfo:
    def __init__(self, obj, count, frames):
        self.obj = obj
        self.count = count
        self.frames = frames
    def __str__(self):
        return f"obj: {self.obj}, count: {self.count}, frames:{self.frames}"

prev_detected_objects = []

# while time.time() < MaxTime:
while True:
    if os.path.exists(file) and os.access(file, os.R_OK):
        try:
            img = cv.imread(file)
            os.remove(file)
        except Exception as e:
            img = None
        if img is not None:
            img = cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE)
            # os.remove(file)

            ####### DETECTION AND TTS ########
            # Dictionary to store object counts
            object_counts = Counter()
            conf = 0
            results = model(img)  # Run inference
            # Process results and count objects
            for result in results:
                for box in result.boxes:
                    conf = box.conf
                    print(f"Detected {model.names[int(box.cls[0])]}. Confidence: {conf}")
                    if conf >= minConf:
                        class_id = int(box.cls[0])
                        class_name = model.names[class_id]
                        object_counts[class_name] += 1

            # Identify newly detected objects
            
            # broken version
            # new_objects = {obj: count for obj, count in object_counts.items() if (for oldObj in prev_detected_objects if obj != oldObj.obj or (obj == oldObj.obj and count > oldObj.count))}
            
            new_objects = {obj: count for obj, count in object_counts.items() if not any(obj == oldObj.obj and count <= oldObj.count for oldObj in prev_detected_objects)}
            for obj in prev_detected_objects:
                print(str(obj))
            
            for oldObj in prev_detected_objects: # iterate through old objs, and update those that aren't in new objs.
                if oldObj.obj not in object_counts:
                    oldObj.frames -= 1
                    if oldObj.frames < 0:
                        prev_detected_objects.remove(oldObj)
                else:
                    oldObj.frames = ObjectPerminanceFrames
                    oldObj.count = object_counts[oldObj.obj]
            

            # If there are new detections, update the file and speak out
            if new_objects:
                
                for obj, count in new_objects.items(): # update previously detected objects with new objects
                    newObjInfo = objectInfo(obj, count, ObjectPerminanceFrames)
                    prev_detected_objects.append(newObjInfo)

                # Prepare speech text
                speech_text = " " + ", ".join([f"{count} {obj}" for obj, count in new_objects.items()])
                
                print("New detections, speaking out:", speech_text)
                engine.say(speech_text)
                engine.runAndWait()
            else:
                print("No new objects detected.")

            ######################


    # print("TimeRemaining: " + str(MaxTime - time.time()))

