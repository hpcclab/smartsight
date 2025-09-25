import threading
import queue
import pyttsx3
import time


class TTSThread(threading.Thread):
    """
    Non-blocking TTS thread using pyttsx3 event loop.
    Supports 3 priority queues: urgent, active, passive.
    Automatically interrupts lower-priority speech when higher-priority arrives.
    """

    def __init__(self, name="SmartSight-TTS", daemon=True):
        super().__init__(name=name, daemon=daemon)

        # Separate queues by priority level
        self.urgent_queue = queue.Queue()
        self.active_queue = queue.Queue()
        self.passive_queue = queue.Queue()

        self.stop_running = threading.Event()
        self.interrupt_event = threading.Event()
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
            message, priority = self.get_next_message()

            if message is None:
                time.sleep(0.05)  # idle briefly
                continue

            # If a higher priority arrives mid-speech, interrupt
            if self.current_priority is not None:
                if self.priority_value(priority) < self.priority_value(self.current_priority):
                    print(f"[{self.name}] Interrupting {self.current_priority} for {priority}")
                    self.engine.stop()

            # Speak
            try:
                with self.speaking_lock:
                    self.is_speaking = True
                    self.current_priority = priority

                print(f"[{self.name}] Speaking ({priority}): {message[:40]}...")
                self.engine.say(message)
                self.engine.runAndWait()

            except Exception as e:
                print(f"[{self.name}] Speech error: {e}")

            finally:
                with self.speaking_lock:
                    self.is_speaking = False
                    self.current_priority = None

    def get_next_message(self):
        """Check queues in priority order: urgent > active > passive."""
        try:
            if not self.urgent_queue.empty():
                return self.urgent_queue.get_nowait(), "urgent"
            elif not self.active_queue.empty():
                return self.active_queue.get_nowait(), "active"
            elif not self.passive_queue.empty():
                return self.passive_queue.get_nowait(), "passive"
            else:
                return None, None
        except queue.Empty:
            return None, None

    def add_message(self, message, priority="passive"):
        """Add message to a specific queue."""
        if priority == "urgent":
            self.urgent_queue.put(message)
        elif priority == "active":
            self.active_queue.put(message)
        else:
            self.passive_queue.put(message)

        print(f"[{self.name}] Queued ({priority}): {message[:40]}...")

        # Interrupt if necessary
        with self.speaking_lock:
            if self.current_priority is not None:
                if self.priority_value(priority) < self.priority_value(self.current_priority):
                    print(f"[{self.name}] Higher priority ({priority}) queued, interrupting...")
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


# ------------------------
# Example usage
# ------------------------
if __name__ == "__main__":
    tts = TTSThread()
    tts.start()

    # Queue some messages
    tts.add_message("This is a passive message. It will run last.", priority="passive")
    tts.add_message("This is an active message. It should interrupt passive.", priority="active")
    tts.add_message("URGENT! This interrupts everything immediately.", priority="urgent")

    # Let it run for a bit
    time.sleep(10)

    tts.stop()
