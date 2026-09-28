import threading
import queue
import logging
import time
import wave
from utilities.logging_setup import get_logger
from modules.TTS_module import TTSModule
from modules.ai_manager import AI_manager

try:
    import winsound
except ImportError:
    winsound = None
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class GlobalResponseModule:
    """
    Manages a priority queue of messages and synthesizes speech using the TTSModule.
    Playback can be interrupted and later resumed from the next queued message.
    """
    def __init__(self, default_priority: int = 10):
        self.logger = get_logger(self.__class__.__name__)
        self.message_queue = queue.PriorityQueue()
        self.running = False
        self.thread = None
        self.default_priority = default_priority
        self._playback_allowed = threading.Event()
        self._playback_allowed.set()
        self._state_lock = threading.Lock()
        self._current_entry = None
        self._play_token = 0
        self._last_expiry_check = time.monotonic()
        self._expiry_interval = 10.0

    def start(self):
        """Starts the global response module thread."""
        if not self.running:
            self.running = True
            self.logger.info("Starting GlobalResponseModule thread...")
            self.thread = threading.Thread(target=self._update_loop, daemon=True)
            self.thread.start()

    def stop(self):
        """Stops the global response module thread."""
        if self.running:
            self.running = False
            self.logger.info("Stopping GlobalResponseModule thread...")
            self._playback_allowed.set()
            with self._state_lock:
                self._play_token += 1
            # Unblock the queue. Empty text is never spoken.
            self.message_queue.put((0, "", 0.0, float("inf")))
            self._purge_playback()
            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=2.0)

    def add_message(self, message, message_time=None, message_expiration=1.0, *, priority=None):
        """Adds a message to the priority queue to be spoken.

        message_time is the unix time when the message became relevant.
        message_expiration is how many seconds after message_time it stays valid.
        """
        if message_time is None:
            message_time = time.time()
        if priority is None:
            priority = self.default_priority
        self.message_queue.put((priority, message, message_time, message_expiration))
        self.logger.info(f"Adding message: '{message}' with priority: {priority}, expires in {message_expiration - (time.time() - message_time)} seconds")
        self.logger.info(f"Current message queue size: {self.message_queue.qsize()}")

    def interrupt_playback(self):
        """Stop the utterance in progress and leave it on the priority queue."""
        with self._state_lock:
            self._playback_allowed.clear()
            self._play_token += 1
            entry = self._current_entry
            self._current_entry = None
            if entry is not None:
                self.message_queue.put(entry)
        self.logger.info("Interrupted playback.")
        self._purge_playback()

    def resume_playback(self):
        """Allow the worker to speak the next message in the priority queue."""
        self._playback_allowed.set()
        self.logger.info("Resuming playback.")

    def _purge_playback(self):
        if winsound is None:
            return
        try:
            winsound.PlaySound(None, winsound.SND_PURGE)
        except Exception as e:
            self.logger.error(f"Failed to stop playback: {e}")

    def _is_expired(self, message_time, message_expiration, now=None):
        if now is None:
            now = time.time()
        return now > message_time + message_expiration

    def _drop_expired_messages(self):
        """Drop expired messages. Runs from the worker at least every 10 seconds."""
        now_mono = time.monotonic()
        if now_mono - self._last_expiry_check < self._expiry_interval:
            return
        self._last_expiry_check = now_mono
        now = time.time()
        kept = []
        try:
            while True:
                item = self.message_queue.get_nowait()
                self.message_queue.task_done()
                _priority, message, message_time, message_expiration = item
                if message and self._is_expired(message_time, message_expiration, now):
                    self.logger.info(f"Dropping expired message: {message}")
                    continue
                kept.append(item)
        except queue.Empty:
            pass
        finally:
            for item in kept:
                self.message_queue.put(item)

    def _block_playback(self, entry, token):
        """Return True when this utterance must not start.

        Expired utterances are removed. Interrupted ones were already put back
        on the queue by interrupt_playback.
        """
        _priority, message, message_time, message_expiration = entry
        with self._state_lock:
            if token != self._play_token or not self._playback_allowed.is_set():
                return True
            if self._is_expired(message_time, message_expiration):
                if self._current_entry is entry:
                    self._current_entry = None
                self.logger.info(f"Dropping expired message before speech: {message}")
                return True
        return False

    def _update_loop(self):
        """Continuous loop to process the message queue and speak."""
        while self.running:
            try:
                self._drop_expired_messages()

                if not self._playback_allowed.is_set():
                    self._playback_allowed.wait(timeout=0.2)
                    continue

                try:
                    entry = self.message_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                if not self.running:
                    break

                priority, message, message_time, message_expiration = entry
                if not message:
                    self.message_queue.task_done()
                    continue

                if self._is_expired(message_time, message_expiration):
                    self.logger.info(f"Dropping expired message before speech: {message}")
                    self.message_queue.task_done()
                    continue

                with self._state_lock:
                    if not self._playback_allowed.is_set():
                        self.message_queue.put(entry)
                        self.message_queue.task_done()
                        continue
                    self._current_entry = entry
                    token = self._play_token

                try:
                    self._speak(entry, token, priority)
                finally:
                    with self._state_lock:
                        if self._current_entry is entry:
                            self._current_entry = None
                    self.message_queue.task_done()
            except Exception as e:
                self.logger.error(f"GlobalResponseModule loop error: {e}")

    def _speak(self, entry, token, priority):
        _priority, message, _message_time, _message_expiration = entry
        if self._block_playback(entry, token):
            return

        tts = AI_manager.get_module(lambda m: isinstance(m, TTSModule))
        if tts is None:
            self.logger.error("No TTS module available.")
            return
        if tts.model is None:
            tts.load_model()

        if self._block_playback(entry, token):
            return

        wav_path = tts.speak(message)
        self._play_wav(wav_path, token, entry, priority)

    def _play_wav(self, wav_path, token, entry, priority):
        if winsound is None or not wav_path:
            return
        try:
            with wave.open(wav_path, "rb") as wav_file:
                frame_rate = wav_file.getframerate() or 1
                duration = wav_file.getnframes() / float(frame_rate)
        except Exception as e:
            self.logger.error(f"Failed to read WAV for playback: {e}")
            return

        # Lifetime is checked here, immediately before sound starts.
        if self._block_playback(entry, token):
            return

        _priority, message, _message_time, _message_expiration = entry
        self.logger.info(f"Speaking (Priority {priority}): {message}")
        try:
            winsound.PlaySound(wav_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
        except Exception as e:
            self.logger.error(f"Playback error: {e}")
            return

        # Interrupt can land after the flag check and before PlaySound starts.
        with self._state_lock:
            interrupted = token != self._play_token or not self._playback_allowed.is_set()
        if interrupted:
            self._purge_playback()
            return

        deadline = time.monotonic() + duration
        while time.monotonic() < deadline and self.running:
            if token != self._play_token or not self._playback_allowed.is_set():
                self._purge_playback()
                return
            time.sleep(0.05)
