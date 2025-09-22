import threading
import queue
import pyttsx3
import time

# Priority definitions (lower number = higher priority)
UrgentPassive = 0
ActiveThread = 2
PassiveThread = 4

class TTSThread(threading.Thread):
    """
    Non-blocking TTS thread using pyttsx3 event loop.
    Speaks messages asynchronously and respects priority with interruption support.
    """
    def __init__(self, name="SmartSight-TTS", daemon=True):
        super().__init__(name=name, daemon=daemon)
        self.message_queue = queue.PriorityQueue()
        self.stop_running = threading.Event()
        self.engine = None 
        self.current_priority = None
        self.interrupt_event = threading.Event()
        self.speaking_lock = threading.Lock()
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
                priority, message = self.message_queue.get(block=True, timeout=0.1)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[{self.name}] Queue error: {e}")
                continue
            
            # Check if we should interrupt current speech
            if self.current_priority is not None and priority < self.current_priority:
                #interrupting current speech for higher priority message
                self.interrupt_event.set()
                # Wait a moment for current speech to stop
                time.sleep(0.1)
            
            # Speak the message with interruption support
            self._speak_with_interruption(priority, message)
            self.message_queue.task_done()

    def _speak_with_interruption(self, priority, message):
        """Speak a message with support for interruption by higher priority messages."""
        with self.speaking_lock:
            self.current_priority = priority
            self.interrupt_event.clear()
            
            try:
                print(f"{self.name} saying Priority {priority} Message: {message}")
                self.engine.say(message)
                
                # Use a custom runAndWait that can be interrupted
                self._run_and_wait_with_interruption()
                
            except Exception as e:
                print(f"[{self.name}] TTS error while speaking: {e}")
            finally:
                self.current_priority = None

    def _run_and_wait_with_interruption(self):
        """Custom runAndWait that can be interrupted by higher priority messages."""
        if self.engine is None:
            return
            
        # Start the speech
        self.engine.runAndWait()
        
        # Check for interruption during speech
        while self.engine.isBusy():
            if self.interrupt_event.is_set():
                print(f"[{self.name}] Speech interrupted by higher priority message")
                self.engine.stop()
                break
            time.sleep(0.01)  # Small delay to prevent busy waiting

    def add_message(self, message, priority=PassiveThread):
        """Add a message to the TTS queue with optional priority."""
        self.message_queue.put((priority, message))
        
        # If this is a higher priority message and we're currently speaking,
        # signal an interruption
        if (self.current_priority is not None and 
            priority < self.current_priority and 
            self.engine is not None and 
            self.engine.isBusy()):
            print(f"[{self.name}] Higher priority message queued, will interrupt current speech")
            self.interrupt_event.set()

    def stop(self):
        """Stop the TTS thread and any current speech."""
        self.stop_running.set()
        self.interrupt_event.set()  # Signal to stop any current speech
        print(f"{self.name} has stopped successfully!")
        try:
            if self.engine is not None:
                self.engine.stop()
        except Exception:
            pass
if __name__ == "__main__":
    print("--- TTS Thread Test Started ---")
    
    # Create and start the TTS thread
    tts_thread = TTSThread(name="SmartSight-TTS")
    tts_thread.start()
    
    # Test interruption mechanism
    print("Adding messages to test interruption...")
    tts_thread.add_message("This is a long passive message that should be interrupted", PassiveThread)
    time.sleep(1)
    # Wait a moment for speech to start, then interrupt with higher priority
    tts_thread.add_message("URGENT: This urgent message should interrupt!", UrgentPassive)
    
    # Add more messages to test priority ordering
    time.sleep(1)
    tts_thread.add_message("Active mode message", ActiveThread)
    tts_thread.add_message("Another passive message", PassiveThread)
    
    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping TTS thread...")
        tts_thread.stop()
        tts_thread.join()
        print("TTS thread stopped.")