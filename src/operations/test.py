import threading
import queue
import pyttsx3
import time
from enum import Enum

class Priority(Enum):
    URGENT = 0
    NORMAL = 1

class AddMessageInterruptTTS(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.engine = pyttsx3.init()
        self.queue = queue.PriorityQueue()
        self.stop_flag = threading.Event()
        self.lock = threading.Lock()
        self.current_priority = None
        self.is_speaking = False

    def run(self):
        while not self.stop_flag.is_set():
            try:
                priority_value, message = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue

            with self.lock:
                self.current_priority = priority_value
                self.is_speaking = True

            print(f"Speaking [{Priority(priority_value).name}]: {message}")
            self.engine.say(message)
            self.engine.runAndWait()

            with self.lock:
                self.is_speaking = False
                self.current_priority = None

    def add_message(self, message, priority=Priority.NORMAL):
        with self.lock:
            # Check if new message has higher priority than current speech
            if self.is_speaking and self.current_priority is not None and priority.value < self.current_priority:
                print(f"Interrupting {Priority(self.current_priority).name} speech with {priority.name} message")
                self.engine.stop()  # Interrupt ongoing speech

        # Add message to queue
        self.queue.put((priority.value, message))

    def stop(self):
        self.stop_flag.set()
        self.engine.stop()

if __name__ == "__main__":
    tts = AddMessageInterruptTTS()
    tts.start()

    # Add a normal message that takes time to speak
    tts.add_message("This is a normal message taking some time.", Priority.NORMAL)
    time.sleep(1)  # Allow speech to start

    # Add an urgent message that interrupts
    tts.add_message("Urgent message interrupting!", Priority.URGENT)

    
