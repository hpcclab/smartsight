import cv2 as cv
import os
import time
import pickle

from ultralytics import YOLO
import pyttsx3
from collections import Counter

import threading
import keyboard
import pyaudio

from operations.object_detection import ObjectDetection
from operations.face_perception import FacePerception
from operations.active_mode import ActiveMode
from operations.text_detection import TextDetector
from operations.commands import build_ocr 

# ----------------------------------------------------------------------------
# Setup path
# ----------------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))

# Facial recognition files
encodings_path = os.path.join(script_dir, 'models', 'FaceDetection', 'encodings.pickle')
face_cascade_path = os.path.join(script_dir, 'models', 'FaceDetection', 'haarcascade_frontalface_default.xml')

data = pickle.loads(open(encodings_path, "rb").read())
detector = cv.CascadeClassifier()
detector.load(face_cascade_path)

# YOLO model
yolo_model_path = os.path.join(script_dir, 'models', 'yolov8s.pt')
model = YOLO(yolo_model_path)

# --- Consolidated OCR Engine ---
# Initialize the full OCR engine once to be shared across modules.
print("Initializing OCR engine...")
ocr_engine = build_ocr()
print("OCR engine initialized.")
# -----------------------------

# Shared temp directory for incoming frames
temp_dir = os.path.abspath(os.path.join(script_dir, '..', 'temp'))
file = os.path.join(temp_dir, 'image.jpg')
image_path = file
UploadImage_path = os.path.join(temp_dir, 'image2.jpg')

# Detection parameters
minConf = 0.73
ObjectPerminanceFrames = 3

# ----------------------------------------------------------------------------
# Initialize TTS engine (passive mode)
# ----------------------------------------------------------------------------
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)
engine.setProperty('volume', 1.0)
engine.setProperty('rate', 150)


# ----------------------------------------------------------------------------
# Passive-mode persistence
# ----------------------------------------------------------------------------
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

# ----------------------------------------------------------------------------
# Time-tracking variables for active mode
# ----------------------------------------------------------------------------
TimeKeyPressed = 0
TimeKeyReleased = 0
TimeSpeechRecognitionStart = 0
TimeMLLMStart = 0
TimeNemoStart = 0
TimeTTSStart = 0

# ----------------------------------------------------------------------------
# Instantiate modular operation classes (active mode, passive mode, etc)
# ----------------------------------------------------------------------------
object_detector = ObjectDetection(model, minConf)
face_perceiver  = FacePerception(detector, data)
text_detector = TextDetector(ocr_engine, iou_threshold=0.02)
active_mode = ActiveMode(ocr_engine)

# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------

def save_image_2():
    """
    Wait until temp/image.jpg is readable, rotate it, and save as image2.jpg
    """
    global image_path, UploadImage_path
    while True:
        if os.path.exists(image_path) and os.access(image_path, os.R_OK):
            try:
                img2 = cv.imread(image_path)
            except Exception:
                img2 = None
            if img2 is not None:
                img2 = cv.rotate(img2, cv.ROTATE_90_COUNTERCLOCKWISE)
                cv.imwrite(UploadImage_path, img2)
                break

# ----------------------------------------------------------------------------
# Input thread for active-mode trigger
# ----------------------------------------------------------------------------
def check_input():
    global passive, Recording, TimeKeyPressed, RecordingTranscription
    while True:
        if keyboard.is_pressed('space'):
            if not Recording:
                TimeKeyPressed = time.time()
                passive = False
                Recording = True
                print("Recording started by user")
                save_image_2()
                print("Started recording...")
                audio = pyaudio.PyAudio()
                active_mode.record_audio(audio)
                audio.terminate()
                RecordingTranscription = active_mode.recognize_speech()
                Recording = False
                break
        time.sleep(0.05)

# Start the input listener thread
passive = True
input_thread = threading.Thread(target=check_input)
input_thread.start()
Recording = False
RecordingTranscription = "Transcription not found."

# --- FPS Benchmarking variables ---
frame_count = 0
start_time = time.time()
# ----------------------------------

# ----------------------------------------------------------------------------
# Main loop: alternate between active and passive modes
# ----------------------------------------------------------------------------

while True:
    # ---------------------- Active Mode ----------------------
    if not passive:
        while Recording:
            time.sleep(0.05)
        print("Finished getting Active transcription")

        UserRequest = RecordingTranscription
        active_mode.MLLMAnalyzeImage(UserRequest, UploadImage_path)
        passive = True 

        time.sleep(0.2)
        while not passive:
            print("paused")
            time.sleep(0.5)
        time.sleep(0.2)

        input_thread = threading.Thread(target=check_input)
        input_thread.start()

    # --------------------- Passive Mode ---------------------
    else:
        frame_start_time = time.time()
        print("Passive perception is active...")
        if os.path.exists(file) and os.access(file, os.R_OK):
            try:
                img = cv.imread(file)
            except Exception:
                img = None

            if img is not None:
                img = cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE)

                yolo_results = object_detector.detect(img)

                object_counts = Counter()
                text_announcements = {}

                if yolo_results:
                    text_announcements = text_detector.analyze_frame(yolo_results, model.names, minConf)
                    
                    for box in yolo_results[0].boxes:
                        class_id = int(box.cls[0])
                        class_name = model.names[class_id]
                        conf = box.conf
                        print(f"Detected {class_name}. Confidence: {conf}")
                        if conf >= minConf:
                            if f"{class_name} with text" not in text_announcements:
                                object_counts[class_name] += 1
                object_counts.update(text_announcements)
                object_counts = face_perceiver.recognize(img, object_counts, passive)
                new_objects = {
                    obj: count
                    for obj, count in object_counts.items()
                    if not any(obj == old.obj and count <= old.count for old in prev_detected_objects)
                }
                for old in prev_detected_objects:
                    print(str(old))

                for old in list(prev_detected_objects): 
                    if old.obj not in object_counts:
                        old.frames -= 1
                        if old.frames < 0:
                            prev_detected_objects.remove(old)
                    else:
                        old.frames = ObjectPerminanceFrames
                        old.count = object_counts[old.obj]

                if new_objects:
                    for obj, count in new_objects.items():
                        prev_detected_objects.append(objectInfo(obj, count, ObjectPerminanceFrames))

                    keys_list = list(new_objects.keys())
                    for obj in keys_list:
                        phonicName = PeoplePhonics.get(obj, "none")
                        if obj not in EverDetected and obj in PeopleInfo:
                            new_objects[f"{phonicName} is {PeopleInfo[obj]}"] = new_objects.pop(obj)
                            EverDetected[obj] = 1
                        elif phonicName != "none":
                            new_objects[phonicName] = new_objects.pop(obj)

                    if not passive:
                        continue

                    speech_text = " " + ", ".join(
                        [f"{count} {o}" if count > 1 else o for o, count in new_objects.items()]
                    )
                    print("New detections, speaking out:", speech_text)
                    engine.say(speech_text)
                    engine.runAndWait()
                else:
                    print("No new objects detected.")
                # --- FPS Calculation Logic ---
                frame_count += 1
                # Calculate and print FPS every 30 frames
                if frame_count % 30 == 0:
                    end_time = time.time()
                    # Time elapsed for 30 frames
                    elapsed_time = end_time - start_time
                    # Calculate average FPS over that period
                    current_fps = frame_count / elapsed_time
                    print(f"------------------------------------")
                    print(f"Average FPS over last 30 frames: {current_fps:.2f}")
                    print(f"------------------------------------")
                    # Reset counters for the next batch
                    frame_count = 0
                    start_time = time.time()
    time.sleep(0.01)
