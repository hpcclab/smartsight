import threading
import queue
import pyttsx3
import time
import traceback


class TTSThread(threading.Thread):
    """Text-to-Speech thread with priority-based message queuing and interruption support."""
    
    # Priority mappings as class constants
    PRIORITIES = {"urgent": 0, "active": 1, "passive": 2}
    PRIORITY_NAMES = {v: k for k, v in PRIORITIES.items()}
    
    def __init__(self, name="SmartSight-TTS", daemon=True):
        super().__init__(name=name, daemon=daemon)
        
        self.queue = queue.PriorityQueue()
        self.stop_running = threading.Event()
        self.interrupted = threading.Event()
        self.speaking_lock = threading.Lock()
        
        self.engine = None
        self.current_priority = None
        self.is_speaking = False
        
        self._log("Initialized.")

    def _log(self, message):
        """Centralized logging with thread name prefix."""
        print(f"[{self.name}] {message}")

    def _reset_engine(self):
        """Safely stop and reset the TTS engine."""
        if self.engine is not None:
            try:
                self.engine.stop()
            except Exception:
                pass
            self.engine = None

    def _initialize_engine(self):
        """Initialize pyttsx3 engine with default properties."""
        self._reset_engine()
        
        try:
            self.engine = pyttsx3.init()
            voices = self.engine.getProperty('voices')
            if len(voices) > 1:
                self.engine.setProperty('voice', voices[1].id)
            self.engine.setProperty('volume', 1.0)
            self.engine.setProperty('rate', 150)
            self._log("Engine initialized successfully.")
        except Exception as e:
            self._log(f"Failed to initialize engine: {e}")
            self.engine = None

    def _ensure_engine(self):
        """Ensure engine is ready, reinitializing if needed."""
        if self.engine is None:
            self._log("Engine is None, reinitializing...")
            self._initialize_engine()
            time.sleep(0.1)

    def _get_next_message(self):
        """Get next message from priority queue (non-blocking)."""
        try:
            priority, message = self.queue.get_nowait()
            self._log(f"Retrieved: priority={priority}, msg='{message[:30]}...'")
            return priority, message
        except queue.Empty:
            return None, None

    def _speak_message(self, priority, message):
        """Handle speaking a single message with proper state management."""
        self.interrupted.clear()
        
        with self.speaking_lock:
            self.is_speaking = True
            self.current_priority = self.PRIORITY_NAMES.get(priority, "passive")
        
        self._log(f"Speaking ({self.current_priority}): {message[:40]}...")
        
        try:
            self.engine.say(message)
            self.engine.runAndWait()
            
            if self.interrupted.is_set():
                self._log("Message was interrupted - resetting engine")
                self._reset_engine()
            else:
                self._log(f"Finished speaking: {message[:40]}...")
                
        except Exception as e:
            if self.interrupted.is_set():
                self._log("Speech interrupted (expected error)")
            else:
                self._log(f"Speech error: {e}")
                traceback.print_exc()
            self._reset_engine()
            self._log("Engine reset - will reinitialize on next message")
        
        finally:
            with self.speaking_lock:
                self.is_speaking = False
                self.current_priority = None
            self._log(f"State reset. Queue size: {self.queue.qsize()}")

    def run(self):
        """Main thread loop - process messages from the queue."""
        self._ensure_engine()
        
        while not self.stop_running.is_set():
            try:
                priority, message = self._get_next_message()
                
                if message is None:
                    time.sleep(0.05)
                    continue
                
                self._ensure_engine()
                
                if self.engine is not None:
                    self._speak_message(priority, message)
                    
            except Exception as e:
                self._log(f"Fatal error in run loop: {e}")
                traceback.print_exc()
                self._reset_engine()
                time.sleep(0.1)

    def add_message(self, message, priority="passive"):
        """Add message to queue, interrupting lower-priority speech if needed."""
        priority_val = self.PRIORITIES.get(priority, 2)
        self.queue.put((priority_val, message))
        self._log(f"Queued ({priority}): {message[:40]}... [Queue: {self.queue.qsize()}]")
        
        with self.speaking_lock:
            if self.current_priority is not None:
                current_val = self.PRIORITIES.get(self.current_priority, 2)
                if priority_val < current_val:
                    self._log(f"INTERRUPTING: {self.current_priority} -> {priority}")
                    self.interrupted.set()
                    self._reset_engine()

    def stop(self):
        """Stop the TTS thread and clear all pending messages."""
        self._log("Stopping...")
        self.stop_running.set()
        
        cleared = 0
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
                cleared += 1
            except queue.Empty:
                break
        
        self._log(f"Cleared {cleared} messages from queue")
        self._reset_engine()
        self._log("Stopped successfully!")
