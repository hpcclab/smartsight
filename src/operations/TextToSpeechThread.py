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

        self.engine = None
        self.current_priority = None  # "urgent" | "active" | "passive"
        self.is_speaking = False

        print(f"[{self.name}] Initialized.")

    def initialize_engine(self):
        """Initialize pyttsx3 engine with default properties."""
        try:
            # Clean up old engine first
            if self.engine is not None:
                try:
                    self.engine.stop()
                    del self.engine
                except:
                    pass
            
            self.engine = pyttsx3.init()
            voices = self.engine.getProperty('voices')
            if len(voices) > 1:
                self.engine.setProperty('voice', voices[1].id)
            self.engine.setProperty('volume', 1.0)
            self.engine.setProperty('rate', 150)
            print(f"[{self.name}] Engine Initialized Successfully")
        except Exception as e:
            print(f"[{self.name}] Failed to initialize engine: {e}")
            self.engine = None

    def run(self):
        if self.engine is None:
            self.initialize_engine()

        # Main loop
        while not self.stop_running.is_set():
            try:
                priority, message = self.get_next_message()

                if message is None:
                    time.sleep(0.05)  # idle briefly
                    continue

                # Ensure engine is ready - REINITIALIZE IF NEEDED
                if self.engine is None:
                    print(f"[{self.name}] Engine is None, reinitializing...")
                    self.initialize_engine()
                    # Give engine a moment to fully initialize
                    time.sleep(0.1)

                # Reset interruption flag at start of new message
                self.interrupted.clear()

                # Speak
                try:
                    with self.speaking_lock:
                        self.is_speaking = True
                        self.current_priority = self.priority_name(priority)
                        print(f"[{self.name}] CURRENT PRIORITY: {self.current_priority}")

                    print(f"[{self.name}] Speaking ({self.current_priority}): {message[:40]}...")
                    
                    self.engine.say(message)
                    self.engine.runAndWait()
                    
                    # Check if we were interrupted during speech
                    if self.interrupted.is_set():
                        print(f"[{self.name}] Message was interrupted - resetting engine")
                        # Force engine reset after interruption
                        try:
                            self.engine.stop()
                        except:
                            pass
                        self.engine = None
                    else:
                        print(f"[{self.name}] Finished speaking: {message[:40]}...")
                        # Even after successful speech, reinitialize to ensure clean state
                        print(f"[{self.name}] Reinitializing engine for next message...")
                        self.engine = None

                except Exception as e:
                    # Check if error was due to interruption
                    if self.interrupted.is_set():
                        print(f"[{self.name}] Speech interrupted (expected error)")
                    else:
                        print(f"[{self.name}] Speech error: {e}")
                        import traceback
                        traceback.print_exc()
                    
                    # ALWAYS reset engine after error or interruption
                    try:
                        if self.engine is not None:
                            self.engine.stop()
                    except:
                        pass
                    self.engine = None
                    print(f"[{self.name}] Engine reset - will reinitialize on next message")

                finally:
                    with self.speaking_lock:
                        self.is_speaking = False
                        self.current_priority = None
                        print(f"[{self.name}] Reset state - is_speaking=False, current_priority=None")
                        print(f"[{self.name}] DEBUG: Messages remaining in queue: {self.queue.qsize()}")
            
            except Exception as e:
                print(f"[{self.name}] Fatal error in run loop: {e}")
                import traceback
                traceback.print_exc()
                self.engine = None  # Reset engine on fatal error
                time.sleep(0.1)  # Prevent tight error loop

    def get_next_message(self):
        """Get next message from priority queue (lowest number = highest priority)."""
        try:
            priority, message = self.queue.get_nowait()
            print(f"[{self.name}] Retrieved message from queue: priority={priority}, msg='{message[:30]}...'")
            return priority, message
        except queue.Empty:
            return None, None

    def add_message(self, message, priority="passive"):
        """Add message to the single priority queue."""
        priority_val = self.priority_value(priority)
        self.queue.put((priority_val, message))
        print(f"[{self.name}] Queued ({priority}): {message[:40]}... [Queue size: {self.queue.qsize()}]")

        # Interrupt if necessary
        with self.speaking_lock:
            if self.current_priority is not None:
                current_val = self.priority_value(self.current_priority)
                if priority_val < current_val:
                    print(f"[{self.name}] INTERRUPTING! {self.current_priority} -> {priority}")
                    print(f"[{self.name}] Stopping current speech to announce higher priority message")
                    
                    # Set interruption flag
                    self.interrupted.set()
                    
                    # Stop the engine to interrupt current speech
                    try:
                        if self.engine is not None:
                            self.engine.stop()
                            print(f"[{self.name}] Engine stopped.")
                            # Force reinitialization on next message
                            self.engine = None
                            print(f"[{self.name}] Engine set to None - will reinitialize")
                    except Exception as e:
                        print(f"[{self.name}] Error stopping engine during interruption: {e}")
                        self.engine = None

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
        print(f"[{self.name}] Cleared {cleared} messages from queue")
        
        try:
            if self.engine is not None:
                self.engine.stop()
        except Exception as e:
            print(f"[{self.name}] Error stopping engine: {e}")
        print(f"[{self.name}] Stopped successfully!")


# ============================================
# TEST CODE
# ============================================

def test_interruption():
    """Test the TTS interruption system with different priority levels."""
    
    # Create and start the TTS thread
    tts = TTSThread()
    tts.start()
    
    print("\n=== TEST 1: Simple Interruption Test ===")
    tts.add_message("This is a very long passive message that should be interrupted by an urgent message in about two seconds. I will keep talking to give you enough time to interrupt me.", priority="passive")
    time.sleep(2)  # Let it start speaking
    tts.add_message("URGENT! Critical interruption!", priority="urgent")
    time.sleep(5)  # Wait longer to see if it continues
    
    print(f"\nDEBUG: Queue size after test 1: {tts.queue.qsize()}")
    print(f"DEBUG: Is speaking: {tts.is_speaking}")
    
    print("\n=== TEST 2: Three Messages in Sequence ===")
    tts.add_message("First message.", priority="passive")
    time.sleep(3)
    tts.add_message("Second message.", priority="passive")
    time.sleep(3)
    tts.add_message("Third message.", priority="passive")
    time.sleep(5)
    
    print(f"\nDEBUG: Final queue size: {tts.queue.qsize()}")
    
    # Cleanup
    print("\n=== Stopping TTS Thread ===")
    tts.stop()
    tts.join(timeout=3)
    print("Test complete!")


if __name__ == "__main__":
    test_interruption()
