import threading
import queue
import pyttsx3
import time
import os

class Console:
    """Handles ANSI colors and centralized logging."""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    CYAN = '\033[36m'     # Info
    GREEN = '\033[32m'    # Success
    YELLOW = '\033[33m'   # Warning
    RED = '\033[31m'      # Error
    MAGENTA = '\033[35m'  # Urgent
    BLUE = '\033[34m'     # Active
    
    @staticmethod
    def log(source, message, color=CYAN):
        timestamp = time.strftime("%H:%M:%S")
        print(f"{color}[{timestamp}] [{source}] {message}{Console.RESET}")

class TTSThread(threading.Thread):
    PRIORITIES = {"urgent": 0, "active": 1, "passive": 2}
    PRIORITY_NAMES = {v: k for k, v in PRIORITIES.items()}
    PRIORITY_COLORS = {0: Console.MAGENTA, 1: Console.BLUE, 2: Console.RESET}

    def __init__(self):
        super().__init__(name="TTS-Thread", daemon=True)
        self.queue = queue.PriorityQueue()
        self.stop_event = threading.Event()
        self.interrupted_event = threading.Event()
        self.speaking_lock = threading.Lock()
        
        self.engine = None
        self.current_priority_val = None
        self.is_speaking = False

    def _initialize_engine(self):
        """Initializes engine and connects the 'started-word' callback."""
        try:
            if self.engine:
                self.engine.stop()
                del self.engine
        except: pass
            
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 160)
            self.engine.setProperty('volume', 1.0)
            
            # Hook the callback for word-level control
            self.engine.connect('started-word', self._on_word)
            
            # Select voice (standardize)
            voices = self.engine.getProperty('voices')
            if voices and len(voices) > 1:
                self.engine.setProperty('voice', voices[1].id)
                
            Console.log(self.name, "Engine initialized (Callback Mode).", Console.GREEN)
        except Exception as e:
            Console.log(self.name, f"Engine Init Failed: {e}", Console.RED)

    def _on_word(self, name, location, length):
        """
        Callback triggered by pyttsx3 before every word.
        """
        if self.interrupted_event.is_set():
            # Stop the engine immediately from within the event loop
            self.engine.stop()

    def run(self):
        self._initialize_engine()
        
        while not self.stop_event.is_set():
            try:
                # 1. Get message (Block until available)
                priority_val, message = self.queue.get(timeout=0.5)
                
                # 2. Update State
                with self.speaking_lock:
                    self.is_speaking = True
                    self.current_priority_val = priority_val
                    self.interrupted_event.clear() # Clear any old flags

                p_name = self.PRIORITY_NAMES.get(priority_val, "unknown")
                p_color = self.PRIORITY_COLORS.get(priority_val, Console.CYAN)
                Console.log(self.name, f"Speaking ({p_name}): {message[:40]}...", p_color)

                # 3. Speak
                if self.engine:
                    try:
                        self.engine.say(message)
                        self.engine.runAndWait() # Blocks here, but _on_word runs internally
                    except Exception as e:
                        Console.log(self.name, f"Playback Error: {e}", Console.RED)
                        self._initialize_engine()
                
                # 4. Handle Interruption Result
                if self.interrupted_event.is_set():
                    Console.log(self.name, ">> Interrupted successfully.", Console.YELLOW)

                # 5. Reset State
                with self.speaking_lock:
                    self.is_speaking = False
                    self.current_priority_val = None
                    
            except queue.Empty:
                continue
            except Exception as e:
                Console.log(self.name, f"Fatal Loop Error: {e}", Console.RED)
                time.sleep(1)

    def add_message(self, message, priority="passive"):
        p_val = self.PRIORITIES.get(priority, 2)
        
        # 1. Push to queue
        self.queue.put((p_val, message))
        
        # 2. Check logic
        with self.speaking_lock:
            if self.is_speaking and self.current_priority_val is not None:
                if p_val < self.current_priority_val:
                    Console.log(self.name, f"SIGNALING INTERRUPT: {self.current_priority_val} -> {p_val}", Console.MAGENTA)
                    self.interrupted_event.set()

    def stop(self):
        Console.log(self.name, "Stopping thread...", Console.YELLOW)
        self.stop_event.set()
        try:
            self.engine.stop()
        except: pass

    def clear_queue(self):
        with self.queue.mutex:
            self.queue.queue.clear()
        Console.log(self.name, "Queue Cleared.", Console.YELLOW)