import time
import yaml
import logging
from abc import ABC, abstractmethod
from typing import Any, Generator, Union
from config.config import get_config

# Set up a unified logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class BaseAIModel(ABC):
    def __init__(self, section_name: str):
        self.section_name = section_name
        self.config = get_config().get(self.section_name, {})
        self.model = None
        self.logger = logging.getLogger(self.__class__.__name__)
        self.cache = {}

    @abstractmethod
    def load_model(self):
        """Child implements specific model loading."""
        pass

    @abstractmethod
    def run_inference(self, input_data: Any, **kwargs) -> Union[Any, Generator]:
        """Child implements core logic (can be return or yield)."""
        pass

    def execute(self, input_key: str, input_data: Any, **kwargs) -> Union[Any, Generator]:
        """
        The Orchestrator: Handles profiling, loading, and streaming detection.
        """
        if self.model is None:
            self.load_model()

        start_time = time.perf_counter()
        
        # We determine if the child is yielding (streaming) or returning
        result = self.run_inference(input_data, **kwargs)

        if isinstance(result, Generator):
            return self._stream_wrapper(result, start_time, input_key)
        else:
            end_time = time.perf_counter()
            self._log_profile(start_time, end_time)
            self.cache[input_key] = {"output": result, "time": end_time}
            return result

    def _stream_wrapper(self, gen, start_time, input_key):
        """Wraps a generator to profile it until the last chunk is yielded."""
        full_response = []
        for chunk in gen:
            full_response.append(str(chunk))
            yield chunk
        
        end_time = time.perf_counter()
        self._log_profile(start_time, end_time, is_stream=True)
        self.cache[input_key] = {"output": "".join(full_response), "time": end_time}

    def _log_profile(self, start, end, is_stream=False):
        duration = end - start
        stream_tag = "[STREAM]" if is_stream else "[STATIC]"
        self.logger.info(f"{stream_tag} Latency: {duration:.4f}s")