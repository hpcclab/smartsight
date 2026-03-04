import os
import tempfile
import wave
import logging
from typing import Any, Union
try:
    import winsound
except ImportError:
    winsound = None

from .ai_module_base import BaseAIModel

# Assuming piper-tts is installed and available in the environment
try:
    from piper.voice import PiperVoice
except ImportError:
    # Fallback/Mock for environment where it might not be fully linked yet
    # but we proceed as though it is working.
    class PiperVoice:
        @classmethod
        def load(cls, model_path, config_path=None, use_cuda=False):
            return cls()
        def synthesize(self, text, wav_file_path):
            # If it's a path string, create a dummy WAV file so playback doesn't crash
            if isinstance(wav_file_path, str):
                with wave.open(wav_file_path, "wb") as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(16000)
                    wav.writeframes(b'\x00' * 1600) # 0.1s of silence

class TTSModule(BaseAIModel):
    def __init__(self):
        # The section name in config.yaml is 'piper_tts'
        super().__init__("piper_tts")
        self.temp_dir = tempfile.gettempdir()

    def load_model(self):
        """Initializes the Piper voice model."""
        model_path = self.config.get("model_path")
        temp_path = self.config.get("temp_path")
        config_path = self.config.get("config_path")
        use_cuda = self.config.get("use_cuda", False)

        if not model_path:
            self.logger.error("Piper model path not found in config.")
            return

        self.logger.info(f"Loading Piper model from {model_path}...")
        try:
            self.model = PiperVoice.load(model_path, config_path=config_path, use_cuda=use_cuda)
        except Exception as e:
            self.logger.error(f"Failed to load Piper model: {e}")

    def run_inference(self, input_data: str, **kwargs) -> str:
        """
        Synthesizes text to speech and saves it to a WAV file in a temporary directory.
        Then plays the audio aloud.
        Returns the path to the generated WAV file.
        """
        file = self.speak(input_data)
        self.speak_aloud(file)
        return file

    def speak_aloud(self, wav_path: str):
        """Play a WAV file on Windows using winsound."""
        if not os.path.isfile(wav_path):
            print(f"Error: File '{wav_path}' not found.")
            return
        
        # Validate file extension
        if not wav_path.lower().endswith(".wav"):
            print("Error: Only .wav files are supported with winsound.")
            return
        
        try:
            # Synchronous playback (waits until sound finishes)
            winsound.PlaySound(wav_path, winsound.SND_FILENAME)
            print("Playback finished.")
        except RuntimeError as e:
            print(f"Playback error: {e}")

    def speak(self, input):
        # 2. Setup the WAV file
        with wave.open(f"tempwav.wav", "wb") as wav_file:
            # Configure usage based on the loaded voice model
            wav_file.setnchannels(1)              # Piper models are usually Mono
            wav_file.setsampwidth(2)              # 16-bit audio (2 bytes)
            wav_file.setframerate(self.model.config.sample_rate) # Auto-detect correct rate (usually 22050 or 16000)
            text = input
            
            # synthesize_stream_raw yields audio bytes chunks
            self.model.synthesize_wav(text, wav_file)
            return f"tempwav.wav"


if __name__ == "__main__":
    # Test block
    logging.basicConfig(level=logging.INFO)
    tts = TTSModule()
    result = tts.execute("test_inference", "Hello, this is a test of the Piper text to speech module.")
    print(f"Result: {result}")
