import threading
import queue
import pyttsx3
import time
import sys
import os
import keyboard
# from operations.TestingThread import TestingThread
# Priority definitions
UrgentPassive = -10
ActiveThread = 0
PassiveThread = 1

class TTSThread(threading.Thread):
    """
    Non-blocking TTS thread using pyttsx3 event loop.
    Speaks messages asynchronously and respects priority.
    """
    def __init__(self, name="SmartSight-TTS", daemon=True):
        super().__init__(name=name, daemon=daemon)
        self.message_queue = queue.PriorityQueue()
        self.stop_running = threading.Event()
        self.engine = None 
        self.current_priority = PassiveThread + 1
        print(f"[{self.name}] Initialized.")

    def initialize_engine(self):
        try:
            self.engine = pyttsx3.init()
            voices = self.engine.getProperty('voices')
            self.engine.setProperty('voice', voices[1].id)
            self.engine.setProperty('volume', 1.0)
            self.engine.setProperty('rate', 200)
            print("Engine Initialized")

        except Exception as e:
            print(f"[{self.name}] Failed to initialize engine: {e}")
            self.engine = None

    def run(self):
        # Ensure engine is ready
        if self.engine is None:
            self.initialize_engine()
        while not self.stop_running.is_set():
            try:
                priority, message = self.message_queue.get(block=True,timeout=0.1)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[{self.name}] Queue error: {e}")
                continue
            try:
                print(f"{self.name} saying Priority {priority} Message: {message}")
                self.engine.say(message)
                self.engine.runAndWait()
            except Exception as e:
                print(f"[{self.name}] TTS error while speaking: {e}")
            finally:
                self.message_queue.task_done()

    def add_message(self, message, priority=PassiveThread):
        while self.engine and self.engine.isBusy():
            #if new message has higher current priority then we interupted the engine by initializing stop function
            if priority > self.current_priority:
                self.engine.stop()
                print(f"{self.name} interuptted due to a higher priority event")
        self.message_queue.put((priority, message))

    def stop(self):
        self.stop_running.set()
        print(f"{self.name} has stopped succesfully!")
        try:
            if self.engine is not None:
                self.engine.stop()
        except Exception:
            pass
if __name__ == "__main__":
    print("--- TTS Thread Test Started ---")
    # 2. Instantiate and start the TTS thread
    tts_thread = TTSThread(name="SmartSight-TTS")
    tts_thread.start()
    # Testing = TestingThread(callback=tts_thread.add_message)
     # Passive and active messages
    passive = [
        "The ambient temperature is seventy-two degrees Fahrenheit.",
        "You have a new message from the lab assistant.",
        "Your current location is the main laboratory.",        
        "The next scheduled task is to check the power supply.",
        "It has been one hour since your last break.",
        "Battery level is at eighty-five percent.",
        "Wi-Fi connection is stable.",
        "You are currently moving at a walking pace.",
        "A new data file has been saved.",
        "The nearest fire exit is to your left."
    ]
    print("Press 'p' to start TestingThread (higher-priority messages).")
    print("Passive messages will keep looping until interrupted.")

    try:
        while True:
            # Continuously feed passive messages
            for msg in passive:
                tts_thread.add_message(msg, priority=PassiveThread)
                time.sleep(1.5)

                # # Check for user input while passive loop runs
                # if keyboard.is_pressed("p"):
                #     if not Testing.is_alive():
                #         print("[MAIN] Starting TestingThread → injecting HIGH priority messages...")
                #         Testing.start()
                #     else:
                #         print("[MAIN] TestingThread already running.")

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("[MAIN] Exiting...")
        tts_thread.stop()
        # if Testing.is_alive():
        #     Testing.stop()
        #     Testing.join()
        tts_thread.join()