import threading
import queue
import pyttsx3
import time
from operations.TestingThread import TestingThread
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
    Testing = TestingThread(callback=tts_thread.add_message)
    Testing.start()
    tts_thread.add_message("Keyboard", PassiveThread)
    tts_thread.add_message("Mouse", ActiveThread)
    tts_thread.add_message("Person 1", UrgentPassive)
    # keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        tts_thread.stop()
        Testing.stop()
        tts_thread.join()
        Testing.join()