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
        
        # Flag to indicate interruption occurred
        self.interrupted = threading.Event()
        # Flag to track if engine needs reset
        self.needs_reset = threading.Event()

        self.engine = None
        self.current_priority = None  # "urgent" | "active" | "passive"
        self.is_speaking = False

        print(f"[{self.name}] Initialized.")

    def initialize_engine(self):
        """Initialize pyttsx3 engine with default properties."""
        try:
            # Clean up old engine first if it exists
            if self.engine is not None:
                try:
                    del self.engine
                except:
                    pass
            
            self.engine = pyttsx3.init()
            voices = self.engine.getProperty('voices')
            if len(voices) > 1:
                self.engine.setProperty('voice', voices[1].id)
            self.engine.setProperty('volume', 1.0)
            self.engine.setProperty('rate', 150)
            self.needs_reset.clear()
            print(f"[{self.name}] Engine Initialized Successfully")
        except Exception as e:
            print(f"[{self.name}] Failed to initialize engine: {e}")
            import traceback
            traceback.print_exc()
            self.engine = None

    def run(self):
        if self.engine is None:
            self.initialize_engine()

        # Main loop
        while not self.stop_running.is_set():
            try:
                priority, message = self.get_next_message()

                if message is None:
                    time.sleep(0.01)  # shorter idle time for better responsiveness
                    continue

                # Only reinitialize if flagged as needed
                if self.needs_reset.is_set() or self.engine is None:
                    print(f"[{self.name}] Engine needs reset, reinitializing...")
                    self.initialize_engine()
                    if self.engine is None:
                        print(f"[{self.name}] Failed to initialize engine, skipping message")
                        continue

                # Reset interruption flag at start of new message
                self.interrupted.clear()

                # Speak
                try:
                    with self.speaking_lock:
                        self.is_speaking = True
                        self.current_priority = self.priority_name(priority)

                    print(f"[{self.name}] Speaking ({self.current_priority}): {message[:40]}...")
                    
                    self.engine.say(message)
                    self.engine.runAndWait()
                    
                    # Check if we were interrupted during speech
                    if self.interrupted.is_set():
                        print(f"[{self.name}] Message was interrupted")
                        # Mark engine for reset only after interruption
                        self.needs_reset.set()
                    else:
                        print(f"[{self.name}] Finished speaking: {message[:40]}...")
                        # Engine is fine, no reset needed

                except RuntimeError as e:
                    # RuntimeError often happens after engine.stop() is called
                    if "run loop already started" in str(e).lower() or self.interrupted.is_set():
                        print(f"[{self.name}] Expected interruption error, engine will reset")
                    else:
                        print(f"[{self.name}] RuntimeError: {e}")
                    self.needs_reset.set()
                    
                except Exception as e:
                    if self.interrupted.is_set():
                        print(f"[{self.name}] Speech interrupted (expected)")
                    else:
                        print(f"[{self.name}] Speech error: {e}")
                        import traceback
                        traceback.print_exc()
                    self.needs_reset.set()

                finally:
                    with self.speaking_lock:
                        self.is_speaking = False
                        self.current_priority = None
            
            except Exception as e:
                print(f"[{self.name}] Fatal error in run loop: {e}")
                import traceback
                traceback.print_exc()
                self.needs_reset.set()
                time.sleep(0.01)

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
            if self.is_speaking and self.current_priority is not None:
                current_val = self.priority_value(self.current_priority)
                if priority_val < current_val:
                    print(f"[{self.name}] INTERRUPTING! {self.current_priority} -> {priority}")
                    
                    # Set interruption flag
                    self.interrupted.set()
                    
                    # Stop the engine to interrupt current speech
                    try:
                        if self.engine is not None:
                            self.engine.stop()
                            # Mark for reset after interruption
                            self.needs_reset.set()
                            print(f"[{self.name}] Engine stopped, marked for reset")
                    except Exception as e:
                        print(f"[{self.name}] Error stopping engine: {e}")
                        self.needs_reset.set()

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
        
        # Clear the queue
        cleared = 0
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
                cleared += 1
            except queue.Empty:
                break
        if cleared > 0:
            print(f"[{self.name}] Cleared {cleared} messages from queue")
        
        try:
            if self.engine is not None:
                self.engine.stop()
                del self.engine
                self.engine = None
        except Exception as e:
            print(f"[{self.name}] Error stopping engine: {e}")
        print(f"[{self.name}] Stopped successfully!")


# ============================================
# TEST CODE
# ============================================

def test_speed_and_interruption():
    """Test speed and responsiveness of the TTS system."""
    
    tts = TTSThread()
    tts.start()
    
    print("\n=== TEST 1: Rapid Sequential Messages (Speed Test) ===")
    start = time.time()
    tts.add_message("Message one.", priority="passive")
    tts.add_message("Message two.", priority="passive")
    tts.add_message("Message three.", priority="passive")
    time.sleep(8)
    elapsed = time.time() - start
    print(f"Time for 3 messages: {elapsed:.2f}s")
    
    print("\n=== TEST 2: Interruption with Quick Recovery ===")
    tts.add_message("This is a long passive message that will be interrupted in two seconds.", priority="passive")
    time.sleep(2)
    tts.add_message("URGENT!", priority="urgent")
    time.sleep(2)
    tts.add_message("Back to normal passive message.", priority="passive")
    time.sleep(4)
    
    print("\n=== TEST 3: Multiple Rapid Interruptions ===")
    tts.add_message("Long passive background message that keeps getting interrupted.", priority="passive")
    time.sleep(1)
    tts.add_message("First urgent interruption!", priority="urgent")
    time.sleep(0.5)
    tts.add_message("Second urgent interruption!", priority="urgent")
    time.sleep(0.5)
    tts.add_message("Third urgent interruption!", priority="urgent")
    time.sleep(5)
    
    # Cleanup
    print("\n=== Stopping TTS Thread ===")
    tts.stop()
    tts.join(timeout=3)
    print("Test complete!")


if __name__ == "__main__":
    test_speed_and_interruption()
