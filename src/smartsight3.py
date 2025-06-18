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
# import operations.optical_flow

import socket
import struct
import cv2
import numpy as np
import sys
import queue


# --- Configuration ---
SERVER_IP = '0.0.0.0'
SERVER_PORT = 8000
HEADER_SIZE = struct.calcsize('<L') # Size of the header (4 bytes for unsigned long)

# --- Shared Resources (Queues and Events) ---
# Queue for raw image data from receiver to processor. Max size 1 to ensure latest frame.
raw_frame_queue = queue.Queue(maxsize=1)
# Queue for processed image data from processor to display. Max size 1 to ensure latest processed frame.
processed_frame_queue = queue.Queue(maxsize=1)
# Queue for the passive mode to use.
passive_frame_queue = queue.Queue(maxsize=1)
# Event to signal all threads to stop
stop_event = threading.Event()
# Frame taken when active mode is activated
active_frame = None
passive = True
Recording = False
RecordingTranscription = "Transcription not found."
# Object holding some active-mode functions and models.
active_mode = None

# --- FPS Benchmarking variables ---
frame_count = 0
start_time = time.time()



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
# Threads
# ----------------------------------------------------------------------------

# Input thread for active-mode trigger
def check_input():
    global passive, Recording, TimeKeyPressed, RecordingTranscription, active_frame
    while True:
        if keyboard.is_pressed('space'):
            if not Recording:
                TimeKeyPressed = time.time()
                passive = False
                Recording = True
                print("Recording started by user")
                # Try to get the latest processed frame for display
                while True:
                    try:
                        active_frame = None
                        active_frame = passive_frame_queue.get(timeout=0.01) # Small timeout
                        if active_frame != None:
                            break
                    except:
                        print("failed to get active frame. Trying again.")
                print("Started recording...")
                audio = pyaudio.PyAudio()
                active_mode.record_audio(audio)
                audio.terminate()
                RecordingTranscription = active_mode.recognize_speech()
                Recording = False
                break
        time.sleep(0.05)

def image_receiver_thread(connection):
    """
    Thread function to continuously receive raw image data from the socket.
    It puts the received raw image data into `raw_frame_queue`.
    """
    data_buffer = b''
    payload_size = 0

    print("Receiver thread started.")
    try:
        while not stop_event.is_set():
            # --- Step 1: Read the header (4 bytes) to get the image size ---
            # Continue receiving chunks until the header is complete
            while len(data_buffer) < HEADER_SIZE:
                try:
                    chunk = connection.recv(4096)
                    if not chunk:
                        print("Client disconnected or no more data in receiver thread.")
                        stop_event.set() # Signal other threads to stop
                        return
                    data_buffer += chunk
                except socket.error as e:
                    if stop_event.is_set(): # Check if we're shutting down gracefully
                        return
                    print(f"Socket error in receiver: {e}")
                    stop_event.set()
                    return

            # Once we have enough data for the header, unpack the payload size
            payload_size = struct.unpack('<L', data_buffer[:HEADER_SIZE])[0]
            data_buffer = data_buffer[HEADER_SIZE:] # Remove the header from the buffer

            # --- Step 2: Read the image data based on the payload size ---
            # Continue receiving chunks until the full image data is in the buffer
            while len(data_buffer) < payload_size:
                try:
                    chunk = connection.recv(4096)
                    if not chunk:
                        print("Client disconnected unexpectedly while receiving image data in receiver thread.")
                        stop_event.set()
                        return
                    data_buffer += chunk
                except socket.error as e:
                    if stop_event.is_set():
                        return
                    print(f"Socket error during image data reception: {e}")
                    stop_event.set()
                    return

            # Extract the complete image data
            image_data = data_buffer[:payload_size]
            data_buffer = data_buffer[payload_size:] # Keep any remaining data for the next frame

            # Put the raw image data into the queue for the processing thread
            # Use put_nowait to avoid blocking if the processing thread is slow,
            # ensuring we always have the latest frame.
            try:
                raw_frame_queue.put_nowait(image_data)
            except queue.Full:
                # If the queue is full, the processing thread hasn't picked up the last frame yet.
                # We discard the old one and put the new one. This effectively implements a
                # 'ping-pong' like behavior for the latest frame.
                try:
                    raw_frame_queue.get_nowait() # Remove the old frame
                except queue.Empty:
                    pass # Should not happen if put_nowait raised Full
                raw_frame_queue.put_nowait(image_data) # Add the new frame

    except Exception as e:
        print(f"Unexpected error in receiver thread: {e}")
        stop_event.set()
    finally:
        print("Receiver thread stopping.")



# Passive thread
def image_processing_thread():
    """
    Thread function to continuously get raw image data from `raw_frame_queue`,
    decode it, perform image processing, and put the processed frame into
    `processed_frame_queue`.
    """
    print("Processing thread started.")
    try:
        image_data = []
        image_data_np = []
        frames = []
        old = 0
        # Parameters for Shi-Tomasi corner detection (for initial feature points)
        feature_params = dict(maxCorners=100,
                            qualityLevel=0.3,
                            minDistance=7,
                            blockSize=7)

        # Parameters for Lucas-Kanade optical flow
        lk_params = dict(winSize=(15, 15),
                        maxLevel=2,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        buffering = True
        while not stop_event.is_set():
            try:
                # Get raw image data from the queue. Timeout prevents infinite blocking on shutdown.
                # Always replace the old frame and update it to the new frame.
                if (len(image_data) - 1) < old:
                    image_data.append(raw_frame_queue.get(timeout=0.1))
                    # Convert the byte data to a NumPy array and decode it
                    image_data_np.append(np.frombuffer(image_data[old], dtype=np.uint8))
                    frames.append(cv2.imdecode(image_data_np[old], cv2.IMREAD_COLOR))
                else:
                    image_data[old] = raw_frame_queue.get(timeout=0.1)
                    # Convert the byte data to a NumPy array and decode it
                    image_data_np[old] = np.frombuffer(image_data[old], dtype=np.uint8)
                    frames[old] = cv2.imdecode(image_data_np[old], cv2.IMREAD_COLOR)
                    buffering = False
                new = old
                old = (old + 1) % 2
                if buffering:
                    continue
                # image_data2 = raw_frame_queue.get(timeout=0.1)

                # if (len(image_data) - 1) >= old:
                #     np_arr = np.frombuffer(image_data[old], dtype=np.uint8)
                # else:
                #     continue
                # frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if frames[0] is not None and frames[0].size > 0 and frames[1] is not None and frames[1].size > 0:
                    # --- Perform image processing here ---
                    # --- frames[new] is your new frame ---
                    # processed_frame = OpticalFlow.frame_subtract(frames[old], frames[new])
                    # Example: Apply a simple blur and convert to grayscale (replace with your desired processing)
                    # processed_frame = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
                    # processed_frame = cv2.GaussianBlur(processed_frame, (5, 5), 0)

                    # Rotate image
                    processed_frame = cv2.rotate(frames[new], cv2.ROTATE_90_COUNTERCLOCKWISE)
                    
                     

                    # Put the processed frame into the queue for the display thread
                    try:
                        processed_frame_queue.put_nowait(processed_frame)
                    except queue.Full:
                        # If display thread is slow, drop old processed frame and add new one
                        try:
                            processed_frame_queue.get_nowait()
                        except queue.Empty:
                            pass
                        processed_frame_queue.put_nowait(processed_frame)
                    # Put the processed frame into the queue for the display thread
                    try:
                        passive_frame_queue.put_nowait(processed_frame)
                    except queue.Full:
                        # If passive thread is slow, drop old processed frame and add new one
                        try:
                            passive_frame_queue.get_nowait()
                        except queue.Empty:
                            pass
                        passive_frame_queue.put_nowait(processed_frame)
                else:
                    print("Processing thread received blank or invalid raw frame.")

            except queue.Empty:
                # No new raw frame available, continue checking
                pass
            except cv2.error as e:
                print(f"OpenCV error in processing thread: {e}")
            except Exception as e:
                print(f"Error in processing thread: {e}")
                stop_event.set() # Signal main thread to stop if critical error

    except Exception as e:
        print(f"Unexpected error in processing thread: {e}")
        stop_event.set()
    finally:
        print("Processing thread stopping.")

# ----------------------------------------------------------------------------
# Main loop: alternate between active and passive modes
# ----------------------------------------------------------------------------
def active_passive_mode():
    # ----------------------------------------------------------------------------
    # Setup path
    # ----------------------------------------------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Facial recognition files
    encodings_path = os.path.join(script_dir, 'models', 'FaceDetection', 'encodings.pickle')
    face_cascade_path = os.path.join(script_dir, 'models', 'FaceDetection', 'haarcascade_frontalface_default.xml')

    data = pickle.loads(open(encodings_path, "rb").read())
    detector = cv2.CascadeClassifier()
    detector.load(face_cascade_path)

    # YOLO model
    yolo_model_path = os.path.join(script_dir, 'models', 'yolov8s.pt')
    model = YOLO(yolo_model_path)

    # --- Consolidated OCR Engine ---
    # Initialize the full OCR engine once to be shared across modules.
    print("Initializing OCR engine...")
    ocr_engine = None # build_ocr()
    print("OCR engine initialized.")
    # -----------------------------

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
    # Global Variables
    # ----------------------------------------------------------------------------

    global active_mode, active_frame, RecordingTranscription, Recording, passive
    # ----------------------------------------------------------------------------
    # Instantiate modular operation classes (active mode, passive mode, etc)
    # ----------------------------------------------------------------------------
    object_detector = ObjectDetection(model, minConf)
    face_perceiver  = FacePerception(detector, data)
    text_detector = TextDetector(ocr_engine, iou_threshold=0.02)
    active_mode = ActiveMode(ocr_engine)

    while True:
        # ---------------------- Active Mode ----------------------
        if not passive:
            while Recording:
                time.sleep(0.05)
            print("Finished getting Active transcription")

            UserRequest = RecordingTranscription
            active_mode.MLLMAnalyzeImage(UserRequest, active_frame)
            passive = True 

            time.sleep(0.2)
            while not passive:
                print("paused")
                time.sleep(0.5)
            time.sleep(0.2)

            input_thread = threading.Thread(target=check_input)
            input_thread.daemon = True
            input_thread.start()

        # --------------------- Passive Mode ---------------------
        else:
            frame_start_time = time.time()
            try:
                frame = passive_frame_queue.get(timeout=0.1)
                print("Passive perception is active...")
                if frame is not None and frame.size > 0:
                    print("Detecting objects.")
                    yolo_results = object_detector.detect(frame)

                    object_counts = Counter()
                    text_announcements = {}

                    if yolo_results:
                        # TODO: RE-ENABLE TEXT DETECTION
                        # text_announcements = text_detector.analyze_frame(yolo_results, model.names, minConf)
                        
                        for box in yolo_results[0].boxes:
                            class_id = int(box.cls[0])
                            class_name = model.names[class_id]
                            conf = box.conf
                            print(f"Detected {class_name}. Confidence: {conf}")
                            if conf >= minConf:
                                if f"{class_name} with text" not in text_announcements:
                                    object_counts[class_name] += 1
                    object_counts.update(text_announcements)
                    object_counts = face_perceiver.recognize(frame, object_counts, passive)
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
            except queue.Empty:
                print("Passive queue empty. Trying again.")
            except Exception as e:
                print(f"An unexpected error occurred in the passive mode: {e}")
        time.sleep(0.01)


def run_server():
    global passive_frame_queue
    """
    Main server function to set up socket, accept connection, and manage threads.
    It handles the display of processed frames.
    """
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    connection = None # Initialize connection to None

    try:
        server_socket.bind((SERVER_IP, SERVER_PORT))
        server_socket.listen(1)
        print(f"Server listening on {SERVER_IP}:{SERVER_PORT}")
        print("Waiting for client connection...")

        # Accept a connection from a client
        connection, client_address = server_socket.accept()
        print(f"Connected to client: {client_address}")

        # Start the receiver thread
        receiver_t = threading.Thread(target=image_receiver_thread, args=(connection,))
        receiver_t.daemon = True # Allow main program to exit even if thread is running
        receiver_t.start()

        # Start the processing thread
        processor_t = threading.Thread(target=image_processing_thread)
        processor_t.daemon = True
        processor_t.start()

        # Start the input listener thread
        input_thread = threading.Thread(target=check_input)
        input_thread.daemon = True
        input_thread.start()
        
        # Start the input listener thread
        main_thread = threading.Thread(target=active_passive_mode)
        main_thread.daemon = True
        main_thread.start()

        # # Start the tts thread
        # processor_t = threading.Thread(target=image_processing_thread)
        # processor_t.daemon = True
        # processor_t.start()
        
        # # Start the user input thread
        # processor_t = threading.Thread(target=image_processing_thread)
        # processor_t.daemon = True
        # processor_t.start()

        print("Main display loop started. Press 'Q' to quit.")
        while not stop_event.is_set():
            try:
                # Try to get the latest processed frame for display
                frame_to_display = processed_frame_queue.get(timeout=0.01) # Small timeout

                if frame_to_display is not None and frame_to_display.size > 0:
                    cv2.imshow('Live Processed Stream (Press Q to quit)', frame_to_display)
                    # passive_frame_queue.put_nowait(frame_to_display)
                else:
                    # This might happen if queue.get returns None after timeout or empty
                    pass # Just continue, no frame available yet

                # Wait for 1ms and check for 'q' key press to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("User quit requested. Signaling threads to stop.")
                    stop_event.set() # Signal threads to stop
                    break # Exit display loop

            except queue.Empty:
                # No processed frame available yet, or processing is slower than display loop
                pass
            except cv2.error as e:
                print(f"OpenCV error in display loop: {e}")
                stop_event.set()
            except Exception as e:
                print(f"Error in main display loop: {e}")
                stop_event.set()
                break

    except socket.error as e:
        print(f"Socket error in main server: {e}")
    except Exception as e:
        print(f"An unexpected error occurred in main server: {e}")
    finally:
        print("Closing connections and resources.")
        stop_event.set() # Ensure all threads are signaled to stop
        # Give threads a moment to finish before joining
        receiver_t.join(timeout=1)
        processor_t.join(timeout=1)
        if connection:
            connection.close()
        server_socket.close()
        cv2.destroyAllWindows()
        print("Server shutdown complete.")


if __name__ == '__main__':
    run_server()
