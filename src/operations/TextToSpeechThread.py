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
        self.is_speaking = False
        print(f"[{self.name}] Initialized.")

    def initialize_engine(self):
        try:
            self.engine = pyttsx3.init()
            voices = self.engine.getProperty('voices')
            if len(voices) > 1:
                self.engine.setProperty('voice', voices[1].id)
            self.engine.setProperty('volume', 1.0)
            self.engine.setProperty('rate', 200)
            
            # Set up engine callbacks for interruption handling
            def on_start(name):
                with self.speaking_lock:
                    self.is_speaking = True
                    
            def on_end(name, completed):
                with self.speaking_lock:
                    self.is_speaking = False
                    self.current_priority = None
                    
            self.engine.connect('started-utterance', on_start)
            self.engine.connect('finished-utterance', on_end)
            
            print(f"[{self.name}] Engine Initialized")

        except Exception as e:
            print(f"[{self.name}] Failed to initialize engine: {e}")
            self.engine = None

    def run(self):
        # Ensure engine is ready
        if self.engine is None:
            self.initialize_engine()
            
        if self.engine is None:
            print(f"[{self.name}] Cannot run without engine")
            return
            
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
                print(f"[{self.name}] Interrupting current speech (priority {self.current_priority}) for higher priority message (priority {priority})")
                self._interrupt_speech()
            
            # Set current priority and speak the message
            with self.speaking_lock:
                self.current_priority = priority
                
            print(f"[{self.name}] Speaking (Priority {priority}): {message[:50]}{'...' if len(message) > 50 else ''}")
            
            # Clear interrupt event before speaking
            self.interrupt_event.clear()
            
            # Speak the message with interruption support
            try:
                self.engine.say(message)
                
                # Start a separate thread to monitor for interruptions during speech
                interrupt_monitor = threading.Thread(
                    target=self._monitor_interruption, 
                    daemon=True
                )
                interrupt_monitor.start()
                
                # This will block until speech is complete or interrupted
                self.engine.runAndWait()
                
            except Exception as e:
                print(f"[{self.name}] Speech error: {e}")
                
            finally:
                with self.speaking_lock:
                    self.is_speaking = False
                    self.current_priority = None
                    
            self.message_queue.task_done()

    def _monitor_interruption(self):
        """Monitor for interruption signals during speech"""
        while self.is_speaking and not self.interrupt_event.is_set():
            time.sleep(0.05)  # Check every 50ms
            
        if self.interrupt_event.is_set() and self.is_speaking:
            print(f"[{self.name}] Interruption detected during speech")
            self._interrupt_speech()

    def _interrupt_speech(self):
        """Actually interrupt the current speech"""
        if self.engine is not None:
            try:
                self.engine.stop()  # Stop current speech
                print(f"[{self.name}] Speech interrupted")
            except Exception as e:
                print(f"[{self.name}] Error stopping engine: {e}")

    def add_message(self, message, priority=PassiveThread):
        """Add a message to the TTS queue with optional priority."""
        self.message_queue.put((priority, message))
        print(f"[{self.name}] Message queued (Priority {priority}): {message[:30]}{'...' if len(message) > 30 else ''}")
        
        # If this is a higher priority message and we're currently speaking,
        # signal an interruption
        with self.speaking_lock:
            if (self.current_priority is not None and 
                priority < self.current_priority and 
                self.is_speaking):
                print(f"[{self.name}] Higher priority message queued, signaling interruption")
                self.interrupt_event.set()

    def stop(self):
        """Stop the TTS thread and any current speech."""
        print(f"[{self.name}] Stopping...")
        self.stop_running.set()
        self.interrupt_event.set()  # Signal to stop any current speech
        
        try:
            if self.engine is not None:
                self.engine.stop()
        except Exception as e:
            print(f"[{self.name}] Error stopping engine: {e}")
        
        print(f"[{self.name}] Stopped successfully!")
