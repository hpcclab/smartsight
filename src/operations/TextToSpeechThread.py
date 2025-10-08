import threading
import queue
import pyttsx3
import time


class TTSThread(threading.Thread):

    def __init__(self, name="SmartSight-TTS", daemon=True):
        super().__init__(name=name, daemon=daemon)

        # Single priority queue
        self.queue = queue.PriorityQueue()

        self.stop_running = threading.Event()
        self.speaking_lock = threading.Lock()

        self.engine = None
        self.current_priority = None  # "urgent" | "active" | "passive"
        self.is_speaking = False

        print(f"[{self.name}] Initialized.")

    def initialize_engine(self):
        """Initialize pyttsx3 engine with default properties."""
        try:
            self.engine = pyttsx3.init()
            voices = self.engine.getProperty('voices')
            if len(voices) > 1:
                self.engine.setProperty('voice', voices[1].id)
            self.engine.setProperty('volume', 1.0)
            self.engine.setProperty('rate', 150)
            print(f"[{self.name}] Engine Initialized")
        except Exception as e:
            print(f"[{self.name}] Failed to initialize engine: {e}")
            self.engine = None

    def run(self):
        if self.engine is None:
            self.initialize_engine()

        # Main loop
        while not self.stop_running.is_set():
            priority, message = self.get_next_message()

            if message is None:
                time.sleep(0.05)  # idle briefly
                continue

            # If a higher priority arrives mid-speech, interrupt
            if self.current_priority is not None:
                if priority < self.priority_value(self.current_priority):
                    print("Run Log")
                    print(f"COMPARING: CURRENT PRIORITY {self.priority_value(self.current_priority)} AND NEW PRIORITY {priority}\n")
                    self.engine.stop()

            # Speak
            try:
                with self.speaking_lock:
                    self.is_speaking = True
                    self.current_priority = self.priority_name(priority)  # Convert back to name for logs
                    print(f"CURRENT PRIORITY: {self.current_priority}\n")

                print(f"[{self.name}] Speaking ({self.current_priority}): {message[:40]}...")
                self.engine.say(message)
                self.engine.runAndWait()

            except Exception as e:
                print(f"[{self.name}] Speech error: {e}")

            finally:
                with self.speaking_lock:
                    self.is_speaking = False
                    self.current_priority = None

    def get_next_message(self):
        """Get next message from priority queue (lowest number = highest priority)."""
        try:
            priority, message = self.queue.get_nowait()
            return priority, message
        except queue.Empty:
            return None, None

    def add_message(self, message, priority="passive"):
        """Add message to the single priority queue."""
        priority_val = self.priority_value(priority)
        self.queue.put((priority_val, message))
        print(f"[{self.name}] Queued ({priority}): {message[:40]}...")

        # Interrupt if necessary
        with self.speaking_lock:
            if self.current_priority is not None:
                current_val = self.priority_value(self.current_priority)
                if priority_val < current_val:
                    print("Add Message Log")
                    print(f"COMPARING: CURRENT PRIORITY {self.current_priority} AND NEW PRIORITY {priority}\n")
                    self.engine.stop()

    @staticmethod
    def priority_value(priority):
        """Helper to compare priorities numerically (lower = higher)."""
        if priority == "urgent":
            return 0
        elif priority == "active":
            return 1
        else:
            return 2

    @staticmethod
    def priority_name(value):
        """Convert priority value back to name."""
        if value == 0:
            return "urgent"
        elif value == 1:
            return "active"
        else:
            return "passive"

    def stop(self):
        """Stop the TTS thread and engine."""
        print(f"[{self.name}] Stopping...")
        self.stop_running.set()
        try:
            if self.engine is not None:
                self.engine.stop()
        except Exception as e:
            print(f"[{self.name}] Error stopping engine: {e}")
        print(f"[{self.name}] Stopped successfully!")
