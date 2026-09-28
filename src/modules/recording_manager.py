import pyaudio
import wave
import time
from utilities.logging_setup import get_logger
from config.config import get_config

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class RecordingManager:
    """
    Singleton class responsible for recording audio from the microphone.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RecordingManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initializes PyAudio settings from config."""
        self.logger = get_logger(self.__class__.__name__)
        self.config = get_config().get("recording", {})
        self.sample_rate = self.config.get("sample_rate", 16000)
        self.channels = self.config.get("channels", 1)
        self.chunk_size = self.config.get("chunk_size", 1024)
        self.audio = pyaudio.PyAudio()

    def record_audio(self, output_filepath: str, is_recording_func) -> str:
        """
        Records audio from the default microphone and saves it to a WAV file.
        
        Args:
            output_filepath (str): The desired file path for the saved audio (.wav).
            is_recording_func (callable): A function that returns True while recording should continue.
            
        Returns:
            str: The path to the saved audio file.
        """
        self.logger.info("Starting audio recording (waiting for release)...")

        try:
            stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )

            frames = []

            # Record while the condition is True
            while is_recording_func():
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                frames.append(data)

            self.logger.info("Recording finished.")

            stream.stop_stream()
            stream.close()

            # Save to WAV file
            with wave.open(output_filepath, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
                wf.setframerate(self.sample_rate)
                wf.writeframes(b''.join(frames))

            self.logger.info(f"Audio saved to {output_filepath}")
            return output_filepath

        except Exception as e:
            self.logger.error(f"Failed to record audio: {e}")
            raise

    def __del__(self):
        """Clean up PyAudio upon deletion."""
        if hasattr(self, 'audio'):
            self.audio.terminate()

# Global Singleton Instance Access
recording_manager_instance = RecordingManager()
