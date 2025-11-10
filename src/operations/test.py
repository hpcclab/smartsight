import threading
import queue
import pyttsx3
import time


class SpeakerThread(threading.Thread):
    """Dedicated thread that only handles speaking."""
    
    def __init__(self, name="Speaker"):
        super().__init__(name=name, daemon=True)
        self.message_queue = queue.Queue(maxsize=1)  # Only holds current message
        self.stop_flag = threading.Event()
        self.speaking_event = threading.Event()
        self.engine = None
        
    def run(self):
        """Main loop - just speak messages."""
        self.engine = pyttsx3.init()
        voices = self.engine.getProperty('voices')
        if len(voices) > 1:
            self.engine.setProperty('voice', voices[1].id)
        self.engine.setProperty('volume', 1.0)
        self.engine.setProperty('rate', 150)
        
        print(f"[{self.name}] Speaker ready")
        
        while not self.stop_flag.is_set():
            try:
                # Wait for a message (blocking with timeout)
                message = self.message_queue.get(timeout=0.1)
                
                self.speaking_event.set()
                print(f"[{self.name}] Speaking: {message[:40]}...")
                
                try:
                    self.engine.say(message)
                    self.engine.runAndWait()
                    print(f"[{self.name}] Finished: {message[:40]}...")
                except Exception as e:
                    print(f"[{self.name}] Speech error: {e}")
                finally:
                    self.speaking_event.clear()
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[{self.name}] Error: {e}")
                self.speaking_event.clear()
    
    def speak(self, message):
        """Add message to speak (non-blocking)."""
        try:
            self.message_queue.put_nowait(message)
            return True
        except queue.Full:
            # Already speaking something
            return False
    
    def interrupt(self):
        """Stop current speech immediately."""
        if self.engine:
            try:
                self.engine.stop()
                # Clear the message queue
                try:
                    self.message_queue.get_nowait()
                except queue.Empty:
                    pass
                print(f"[{self.name}] Interrupted")
                return True
            except Exception as e:
                print(f"[{self.name}] Interrupt error: {e}")
                return False
        return False
    
    def is_speaking(self):
        """Check if currently speaking."""
        return self.speaking_event.is_set()
    
    def stop(self):
        """Stop the speaker thread."""
        self.stop_flag.set()
        if self.engine:
            try:
                self.engine.stop()
            except:
                pass


class TTSManager:
    """Main thread logic for managing TTS with priorities."""
    
    def __init__(self):
        self.priority_queue = queue.PriorityQueue()
        self.speaker = SpeakerThread()
        self.current_priority = None
        self.processing_thread = None
        self.stop_flag = threading.Event()
        
        print("[TTSManager] Initialized")
    
    def start(self):
        """Start the TTS system."""
        self.speaker.start()
        self.processing_thread = threading.Thread(
            target=self._process_queue, 
            daemon=True,
            name="QueueProcessor"
        )
        self.processing_thread.start()
        print("[TTSManager] Started")
    
    def _process_queue(self):
        """Background thread that processes the priority queue."""
        while not self.stop_flag.is_set():
            try:
                # Get highest priority message (non-blocking)
                priority, message = self.priority_queue.get(timeout=0.1)
                
                priority_name = self.priority_name(priority)
                print(f"[QueueProcessor] Processing ({priority_name}): {message[:40]}...")
                
                # Check if we need to interrupt
                if self.current_priority is not None:
                    if priority < self.current_priority:
                        print(f"[QueueProcessor] INTERRUPTING! {self.priority_name(self.current_priority)} -> {priority_name}")
                        self.speaker.interrupt()
                        # Give interruption a moment to take effect
                        time.sleep(0.05)
                
                # Wait for speaker to be ready
                while self.speaker.is_speaking():
                    time.sleep(0.01)
                
                # Send to speaker
                self.current_priority = priority
                success = self.speaker.speak(message)
                
                if not success:
                    # Speaker busy, re-queue with slight delay
                    time.sleep(0.05)
                    self.priority_queue.put((priority, message))
                    continue
                
                # Wait for speech to complete
                while self.speaker.is_speaking():
                    time.sleep(0.01)
                
                self.current_priority = None
                print(f"[QueueProcessor] Completed ({priority_name})")
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[QueueProcessor] Error: {e}")
                import traceback
                traceback.print_exc()
    
    def add_message(self, message, priority="passive"):
        """Add a message to the queue."""
        priority_val = self.priority_value(priority)
        self.priority_queue.put((priority_val, message))
        print(f"[TTSManager] Queued ({priority}): {message[:40]}...")
    
    @staticmethod
    def priority_value(priority):
        """Convert priority name to numeric value."""
        if priority == "urgent":
            return 0
        elif priority == "active":
            return 1
        else:
            return 2
    
    @staticmethod
    def priority_name(value):
        """Convert priority value to name."""
        if value == 0:
            return "urgent"
        elif value == 1:
            return "active"
        else:
            return "passive"
    
    def stop(self):
        """Stop the TTS system."""
        print("[TTSManager] Stopping...")
        self.stop_flag.set()
        
        # Clear queue
        cleared = 0
        while not self.priority_queue.empty():
            try:
                self.priority_queue.get_nowait()
                cleared += 1
            except queue.Empty:
                break
        
        if cleared > 0:
            print(f"[TTSManager] Cleared {cleared} messages")
        
        self.speaker.stop()
        
        if self.processing_thread:
            self.processing_thread.join(timeout=2)
        
        self.speaker.join(timeout=2)
        print("[TTSManager] Stopped")


# ============================================
# TEST CODE
# ============================================

def test_tts_manager():
    """Test the new architecture."""
    
    tts = TTSManager()
    tts.start()
    
    print("\n=== TEST 1: Sequential Messages ===")
    tts.add_message("First message.", priority="passive")
    tts.add_message("Second message.", priority="passive")
    tts.add_message("Third message.", priority="passive")
    time.sleep(8)
    
    print("\n=== TEST 2: Interruption ===")
    tts.add_message("This is a long passive message that will be interrupted.", priority="passive")
    time.sleep(2)
    tts.add_message("URGENT INTERRUPTION!", priority="urgent")
    time.sleep(4)
    
    print("\n=== TEST 3: Multiple Priorities ===")
    tts.add_message("Passive background message.", priority="passive")
    time.sleep(0.5)
    tts.add_message("Active message coming through.", priority="active")
    time.sleep(0.5)
    tts.add_message("URGENT! Critical alert!", priority="urgent")
    time.sleep(8)
    
    print("\n=== Cleanup ===")
    tts.stop()
    print("Test complete!")


if __name__ == "__main__":
    test_tts_manager()
