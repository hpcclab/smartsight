import threading
import queue
import pyttsx3
import time

# Priority definitions
UrgentPassive = -10
ActiveThread = 0
PassiveThread = 1

class TTSThread(threading.Thread):
    """
    Non-blocking TTS thread using pyttsx3 event loop.
    Speaks messages asynchronously and respects priority.
    """
    def __init__(self, message_queue: queue.PriorityQueue, active_mode_event: threading.Event, name="SmartSight-TTS", daemon=True):
        super().__init__(name=name, daemon=daemon)
        self.message_queue = message_queue
        self.active_mode_event = active_mode_event
        self.running = False
        self.engine = None
        print(f"[{self.name}] Initialized.")

    def initialize_engine(self):
        try:
            self.engine = pyttsx3.init()
            voices = self.engine.getProperty('voices')
            self.engine.setProperty('voice', voices[1].id if len(voices) > 1 else voices[0].id)
            self.engine.setProperty('volume', 1.0)
            self.engine.setProperty('rate', 250)
            print(f"[{self.name}] Engine initialized.")
            self.engine.startLoop(False)  # Non-blocking loop
        except Exception as e:
            print(f"[{self.name}] Failed to initialize engine: {e}")
            self.engine = None

    def run(self):
        self.initialize_engine()
        if not self.engine:
            return
        self.running = True

        while self.running:
            try:
                # Check for messages
                try:
                    priority, message = self.message_queue.get(block=True, timeout=0.1)
                except queue.Empty:
                    priority, message = None, None

                if message:
                    if message == "SHUTDOWN_SIGNAL":
                        self.running = False
                    elif self.active_mode_event.is_set() and priority >= ActiveThread:
                        # Skip non-urgent messages during active mode
                        self.message_queue.task_done()
                        continue
                    else:
                        if priority <= UrgentPassive:
                            self.engine.stop()  # interrupt for urgent messages
                        print(f"[{self.name}] Saying message: {message}")
                        self.engine.say(message)
                    self.message_queue.task_done()

                # Non-blocking iteration of pyttsx3 loop
                self.engine.iterate()
                time.sleep(0.01)

            except Exception as e:
                print(f"[{self.name}] Error: {e}")

        self.cleanup()

    def add_message(self, message: str, priority=PassiveThread):
        self.message_queue.put((priority, message))

    def stop(self):
        self.running = False
        self.message_queue.put((UrgentPassive - 1, "SHUTDOWN_SIGNAL"))

    def cleanup(self):
        if self.engine:
            try:
                self.engine.endLoop()
                self.engine.stop()
                self.engine = None
            except Exception as e:
                print(f"[{self.name}] Cleanup failed: {e}")

if __name__ == "__main__":
    print("--- TTS Thread Test Started ---")

    # 1. Create a shared priority queue and the active mode event
    tts_queue = queue.PriorityQueue()
    active_mode_event = threading.Event()

    # 2. Instantiate and start the TTS thread
    tts_thread = TTSThread(tts_queue, active_mode_event, name="SmartSight-TTS")
    tts_thread.start()

    # Give the engine a moment to initialize
    time.sleep(5)

    # --- Phase 1: Normal passive messages ---
    print("\n--- Phase 1: Passive Messages ---")
    tts_thread.add_message("A person is detected to your left.", priority=PassiveThread)
    tts_thread.add_message("Right now it is 4:30 PM.", priority=PassiveThread)

    # Wait a bit to let these messages speak
    time.sleep(5)

    # --- Phase 2: Urgent message (interrupt) ---
    print("\n--- Phase 2: Urgent Message ---")
    tts_thread.add_message("Danger! Workzone ahead!", priority=UrgentPassive)

    # Let the urgent message speak
    time.sleep(3)

    # --- Phase 3: Mixed messages ---
    print("\n--- Phase 3: Mixed Messages ---")
    tts_thread.add_message("This is a normal status update.", priority=PassiveThread)
    tts_thread.add_message("Emergency! Please stop immediately!", priority=UrgentPassive)
    tts_thread.add_message("All systems operating normally.", priority=PassiveThread)

    # Allow all queued messages to play
    time.sleep(8)

    # --- Phase 4: Shutdown ---
    print("\n--- Phase 4: Shutdown ---")
    tts_thread.stop()
    tts_thread.join(timeout=5)

    if tts_thread.is_alive():
        print("TTS thread did not terminate gracefully.")
    else:
        print("TTS thread stopped successfully.")

    print(f"Is TTS queue empty at shutdown? {tts_queue.empty()}")
    print("--- TTS Thread Test Finished ---")
