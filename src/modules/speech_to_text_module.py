import logging
import whisper
from config.config import get_config
from .ai_module_base import BaseAIModel

class SpeechToTextModule(BaseAIModel):
    """
    Module responsible for transcribing audio files to text using OpenAI Whisper.
    """
    def __init__(self):
        super().__init__("speech_to_text")
        self.model_size = self.config.get("model_size", "base")
        self.use_fp16 = self.config.get("use_fp16", False)

    def load_model(self):
        """Loads the Whisper STT model based on config."""
        self.logger.info(f"Loading Whisper STT Model ({self.model_size})...")
        try:
            self.model = whisper.load_model(self.model_size)
            self.logger.info("Whisper STT Model loaded successfully.")
        except Exception as e:
            self.logger.error(f"Failed to load Whisper STT Model: {e}")
            raise

    def run_inference(self, input_data: str, **kwargs) -> str:
        """
        Transcribes an audio file.

        Args:
            input_data (str): The file path to the audio (.wav) file.
            
        Returns:
            str: Transcribed text.
        """
        self.logger.debug(f"Transcribing audio from: {input_data}")
        try:
            # fp16=False if mostly running on CPU unless requested explicitly
            result = self.model.transcribe(input_data, fp16=self.use_fp16)
            transcribed_text = result.get("text", "").strip()
            self.logger.debug(f"Transcription result: {transcribed_text}")
            return transcribed_text
        except Exception as e:
            self.logger.error(f"Transcription failed: {e}")
            return ""
