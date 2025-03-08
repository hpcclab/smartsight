import cv2 as cv
import os
import time

import face_recognition
import pickle

from ultralytics import YOLO
import pyttsx3 
from collections import Counter

# FACIAL recognition section
encodings_path = "FaceDetection\\encodings.pickle"
face_cascade_path = "FaceDetection\\haarcascade_frontalface_default.xml"

# Load facial recognition encodings
data = pickle.loads(open(encodings_path, "rb").read())
detector = cv.CascadeClassifier()
detector.load(face_cascade_path)


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

PeoplePhonics = {"Doctor Mosen Amini Salehi":"Doctor Ahmeeni", "cup":"drink"}
PeopleInfo = {"Doctor Mosen Amini Salehi":"Your professor from the University of North Texas"}
EverDetected = {}

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
            
            # If there are any people in the image, we try to detect and recognize faces. 
            # If we find any faces, we change the objects list to include the person 
            # (count = 0 b/c there aren't going to be more than 1 person)
            # This will make it where if the same person is detected, they won't be spoken.
            # We'll need to include logic for
            #       1. Detecting the first time we've ever seen each person
            #       2. Changing the name of the person to the information from a small database with details.

            faces = []
            # Facial recognition
            if "person" in object_counts:
                # Convert the image to grayscale for face detection and RGB for face recognition
                gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

                # Detect faces in the grayscale image
                rects = detector.detectMultiScale(gray, scaleFactor=1.3, 
                                                minNeighbors=6, minSize=(40, 40),
                                                flags=cv.CASCADE_SCALE_IMAGE)
                boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]
                # Compute the facial embeddings for each face bounding box
                encodings = face_recognition.face_encodings(rgb, boxes)
                names = []
                tolerance = 0.5
                confidences = []
                currentname = "Unknown"
                # Loop over the facial embeddings
                for encoding in encodings:
                    # Attempt to match each face in the input image to our known encodings
                    distances = face_recognition.face_distance(data["encodings"], encoding)
                    
                    matches = []
                    for i in distances:
                        m = True
                        if i > tolerance:
                            m = False
                        matches.append(m)
                        # print(i, m)
                        if m:
                            confidences.append(1-i)
                    
                    name = "Unknown"  # Default to "Unknown" if no match is found
                    # Check if a match was found
                    if True in matches:
                        # Get the indices of all matched faces and count each match
                        matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                        counts = {}
                        
                    # Count the number of times each face was matched
                        for i in matchedIdxs:
                            name = data["names"][i]
                            counts[name] = counts.get(name, 0) + 1

                        # Determine the recognized face with the most matches
                        name = max(counts, key=counts.get)

                        # If a new person is identified, print their name
                        if currentname != name:
                            currentname = name
                            #print(currentname)

                    # Add the name to the list of recognized names
                    names.append(name)
                for name in names:
                    if name != "Unknown":
                        # TODO: Add logic for person bio data
                        object_counts[name] = 1
            
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
                
                keys_list = list(new_objects.keys())
                for obj in keys_list:
                    phonicName = "none"
                    # get phonics
                    if obj in PeoplePhonics:
                        phonicName = PeoplePhonics[obj]
                    if obj not in EverDetected and obj in PeopleInfo:
                        new_objects[phonicName + " is " + PeopleInfo[obj]] = new_objects.pop(obj)
                        EverDetected[obj] = 1
                    # replace with
                    elif phonicName != "none":
                        new_objects[phonicName] = new_objects.pop(obj)
                
                

                # Prepare speech text
                speech_text = " " + ", ".join([f"{count} {obj}" if count > 1 else f"{obj}" for obj, count in new_objects.items()])
                
                print("New detections, speaking out:", speech_text)
                engine.say(speech_text)
                engine.runAndWait()
            else:
                print("No new objects detected.")
            
            # Show the image:
            # cv.imshow("image", img)
            # cv.pollKey()
            
            ######################


    # print("TimeRemaining: " + str(MaxTime - time.time()))

