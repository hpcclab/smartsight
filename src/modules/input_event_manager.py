import os
import logging
import threading
import time
import keyboard
from modules.recording_manager import recording_manager_instance
from modules.ai_manager import AI_manager
# from modules.active_module_manager import active_module_manager_instance
from modules.speech_to_text_module import SpeechToTextModule

logger = logging.getLogger(__name__)

class InputEventManager:
    """
    Manager responsible for orchestrating user input events (e.g., voice commands).
    """

    def __init__(self, temp_dir: str = "temp"):
        self.temp_dir = temp_dir
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
            
        self._stop_event = threading.Event()
        self._listening_thread = None

    def start(self):
        """Starts the background listening thread."""
        if self._listening_thread is None or not self._listening_thread.is_alive():
            self._stop_event.clear()
            self._listening_thread = threading.Thread(target=self._listen_for_input, daemon=True)
            self._listening_thread.start()
            logger.info("InputEventManager listening thread started.")

    def stop(self):
        """Stops the background listening thread."""
        self._stop_event.set()
        if self._listening_thread:
            self._listening_thread.join()
            logger.info("InputEventManager listening thread stopped.")

    def _listen_for_input(self):
        logger.info("Listening for space bar press to trigger recording...")
        while not self._stop_event.is_set():
            if keyboard.is_pressed('space'):
                self.process_voice_command()
                # Wait for the spacebar to be released before continuing
                while keyboard.is_pressed('space') and not self._stop_event.is_set():
                    time.sleep(0.1)
            time.sleep(0.05)

    def process_voice_command(self) -> str:
        """
        Triggers the RecordingManager to capture audio while the spacebar is held,
        then delegates the audio to SpeechToTextModule via the AIManager for transcription.

        Returns:
            str: The transcribed text.
        """
        temp_audio_path = os.path.join(self.temp_dir, "last_voice_command.wav")

        logger.info("Space bar pressed! Initiating voice command processing...")

        # 1. Record Audio
        try:
            recording_manager_instance.record_audio(
                temp_audio_path, 
                is_recording_func=lambda: keyboard.is_pressed('space') and not self._stop_event.is_set()
            )
            logger.info("Voice command recorded.")
        except Exception as e:
            logger.error(f"Failed to record voice command: {e}")
            return ""

        # 2. Transcribe Audio via AIManager
        logger.info("Sending recorded audio for transcription...")
        try:
            transcribed_text = AI_manager.execute_module(
                condition=lambda m: isinstance(m, SpeechToTextModule),
                input_key="voice_command",
                input_data=temp_audio_path
            )

            logger.info(f"Transcription complete: '{transcribed_text}'")
            # Trigger active module.

            return transcribed_text
        except Exception as e:
            logger.error(f"Failed to transcribe voice command: {e}")
            return ""
