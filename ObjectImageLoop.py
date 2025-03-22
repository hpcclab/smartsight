import cv2 as cv
import os
import time

import face_recognition
import pickle

from ultralytics import YOLO
import pyttsx3 
from collections import Counter

# for handling input
import threading
import keyboard

# for the mllm api calls
import json
import base64
import os
import pyttsx3
import nest_asyncio
from openai import OpenAI
from nemoguardrails import LLMRails, RailsConfig

# For recording audio
import keyboard  # For detecting keypresses
import pyaudio   # For audio recording
import wave      # For saving the audio file

# For stt
import whisper

# Load whisper model
whisperModel = whisper.load_model("base")



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
minConf = 0.73

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

def MLLMAnalyzeImage(UserRequest):
    # Initialize Hive AI client
    client = OpenAI(
        base_url="https://api.thehive.ai/api/v3/",  # Hive AI's endpoint
        api_key="HwDF5vDdbekdQbWsjcrsAXfsZo53N2v7"  # Replace with your API key
    )

    # Set up NeMo Guardrails
    NVIDIA_API_KEY = "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"
    nest_asyncio.apply()
    os.environ["NVIDIA_API_KEY"] = NVIDIA_API_KEY
    config = RailsConfig.from_path("./config")
    rails = LLMRails(config)

    # Initialize Text-to-Speech
    global engine

    def get_completion(prompt, image_path, model="meta-llama/llama-3.2-11b-vision-instruct"):
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": base64_image}}
                ]}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content

    def nemo(text):
        completion = rails.generate(messages=[{"role": "user", "content": text}])
        return completion["content"]

    output_file = "results.txt"

    with open(output_file, "w") as result_file:
        image_name = "image.jpg"
        prompt = UserRequest + " Ensure response is reasonable and brief."
        image_path = image_name # os.path.join(test_folder, image_name)
        
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            final_response = "No Response because image not found."
        else:
            print(f"Processing {image_name}...")
            
            # Get AI response
            ai_response = get_completion(prompt, image_path)
            
            # Apply NeMo Guardrails
            final_response = ai_response
            # nemo(ai_response)
        
        # Write to output file
        result_file.write(f"Image: {image_name}\n")
        result_file.write(f"Prompt: {prompt}\n")
        result_file.write(f"Response: {final_response}\n\n")
        
        # Convert to speech
        engine.say(final_response)
        engine.runAndWait()
        
        print(f"Completed: {image_name}\n")
    global passive
    passive = True




# Audio configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100  # Sampling rate
CHUNK = 1024  # Size of each audio chunk
# RECORD_SECONDS = 5  # Duration of recording if fixed duration needed
WAVE_OUTPUT_FILENAME = "recording.wav"

audio = pyaudio.PyAudio()

# Function to record audio
def record_audio():
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

    print("Recording... Press 'spacebar' again to stop.")
    frames = []

    while True:
        data = stream.read(CHUNK)
        frames.append(data)
        if keyboard.is_pressed("space"):  # Stop recording when spacebar is pressed again
            break

    print("Recording stopped.")
    stream.stop_stream()
    stream.close()

    # Save the recording
    with wave.open(WAVE_OUTPUT_FILENAME, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    print(f"Audio saved as {WAVE_OUTPUT_FILENAME}")




##### Split into a thread for passive and one listening for input.
def check_input():
    global passive
    global Recording
    while True:
        # user_input = input("Type 'exit' to stop the script: ")
        if keyboard.is_pressed('space'):
            passive = False
            if not Recording:
                print("Recording started by user")
                Recording = True
                print("Started recording...")
                # keyboard.wait("space")  # Wait for the first spacebar press
                record_audio()

                # Cleanup
                audio.terminate()
                
                # textToSpeech
                transcription = whisperModel.transcribe("recording.wav")

                print(transcription["text"])
                global RecordingTranscription
                RecordingTranscription = transcription["text"]
                Recording = False

                # This will be removed when we want to run the mllm
                # passive = True
                break
            print("Script interrupted by user.")
            break
        time.sleep(0.1)

passive = True
input_thread = threading.Thread(target=check_input)
input_thread.start()
Recording = False
RecordingTranscription = "Transcription not found."

    # if First:
    #     input_thread = threading.Thread(target=check_input)
    #     input_thread.start()
    #     First = False
        

while True:
    if not passive:
        # mllm_thread = threading.Thread(target=MLLMAnalyzeImage)
        # mllm_thread.start()
        # Wait until recording is ready.
        while Recording == True:
            time.sleep(0.1)
        
        
        # Extract text from recording.
        UserRequest = RecordingTranscription
        
        MLLMAnalyzeImage(UserRequest)
        time.sleep(0.2)
        # input_thread = threading.Thread(target=check_input)
        # input_thread.start()
        while not passive:
            print("paused")
            time.sleep(0.5)
        time.sleep(0.2)
        input_thread = threading.Thread(target=check_input)  
        input_thread.start()
    # Your main script logic here
    # print("Script is passive...")
    # time.sleep(1)
    print(f"passive: {passive}")
    # Your main script logic here
    print("Passive perception is passive...")
    # while time.time() < MaxTime:
    if os.path.exists(file) and os.access(file, os.R_OK):
        try:
            img = cv.imread(file)
            # os.remove(file)
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
            if not passive:
                continue
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

                if not passive:
                    continue

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