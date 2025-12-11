import threading, queue, time, tempfile, os, wave
import sounddevice as sd
import soundfile as sf
from piper import PiperVoice  # from piper-tts package


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

class PiperTTSThread(threading.Thread):
    PRIORITIES = {"urgent": 0, "active": 1, "passive": 2}
    # PRIORITY_NAMES, PRIORITY_COLORS, Console same as before
    model_path = r"C:\Users\Crack\OneDrive\Documents\GitHub\smartsight\src\operations\TTS\PiperModel\en_US-lessac-low.onnx"
    def __init__(self):
        super().__init__(name="TTS-Thread", daemon=True)
        self.queue = queue.PriorityQueue()
        self.stop_event = threading.Event()
        self.interrupted_event = threading.Event()
        self.speaking_lock = threading.Lock()
        self.current_priority_val = None
        self.is_speaking = False
        self.voice = PiperVoice.load(self.model_path)  # load once [web:40]
        self.current_stream = None  # for manual playback stop

    def _synthesize_to_file(self, text: str) -> str:
        # Use Piper's built-in synthesize_wav method
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        
        try:
            with wave.open(path, 'wb') as wav_file:
                self.voice.synthesize_wav(text, wav_file)
        except Exception as e:
            if os.path.exists(path):
                os.remove(path)
            raise e
        
        return path

    def _play_with_interrupt(self, wav_path: str):
        data, samplerate = sf.read(wav_path, dtype="float32")
        # Play the entire audio, checking for interruption
        sd.play(data, samplerate)
        stream = sd.get_stream()

        # Check for interruption while playing
        while stream.active:
            if self.interrupted_event.is_set() or self.stop_event.is_set():
                sd.stop()
                break
            time.sleep(0.05)  # Check every 50ms

    def run(self):
        while not self.stop_event.is_set():
            try:
                priority_val, message = self.queue.get(timeout=0.5)

                with self.speaking_lock:
                    self.is_speaking = True
                    self.current_priority_val = priority_val
                    self.interrupted_event.clear()

                # log priority etc., same as before

                wav_path = self._synthesize_to_file(message)
                try:
                    if not self.interrupted_event.is_set():
                        self._play_with_interrupt(wav_path)
                finally:
                    if os.path.exists(wav_path):
                        os.remove(wav_path)

                with self.speaking_lock:
                    self.is_speaking = False
                    self.current_priority_val = None

            except queue.Empty:
                continue

    def add_message(self, message, priority="passive"):
        p_val = self.PRIORITIES.get(priority, 2)
        self.queue.put((p_val, message))
        with self.speaking_lock:
            if self.is_speaking and self.current_priority_val is not None:
                if p_val < self.current_priority_val:
                    self.interrupted_event.set()

    def stop(self):
        self.stop_event.set()
        self.interrupted_event.set()
        sd.stop()
