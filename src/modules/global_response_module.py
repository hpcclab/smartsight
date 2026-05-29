import threading
import queue
import logging
from typing import Union
from modules.TTS_module import TTSModule
from modules.ai_manager import AI_manager

class GlobalResponseModule:
    """
    Manages a priority queue of messages and synthesizes speech using the TTSModule.
    """
    def __init__(self, default_priority: int = 10):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.message_queue = queue.PriorityQueue()
        self.running = False
        self.thread = None
        self.default_priority = default_priority

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
            # Unblock the queue
            self.message_queue.put((0, ""))
            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=2.0)

    def add_message(self, text: Union[str, list], priority: int = None):
        """Adds a message to the priority queue to be spoken."""
        if priority is None:
            priority = self.default_priority
        self.message_queue.put((priority, text))

    def _update_loop(self):
        """Continuous loop to process the message queue and speak."""
        import string
        import time
        punctuation_and_space = set(string.punctuation + " \n\r\t")

        while self.running:
            try:
                # Block until a message is available
                priority, text = self.message_queue.get(timeout=1.0)
                
                if not self.running:
                    break
                    
                if text is not None:
                    if isinstance(text, list):
                        buffer = ""
                        read_idx = 0
                        while self.running:
                            if read_idx < len(text):
                                token = text[read_idx]
                                read_idx += 1
                                
                                if token is None:  # Sentinel indicating stream finished
                                    if buffer.strip():
                                        self.logger.info(f"Speaking remaining (Priority {priority}): {buffer}")
                                        AI_manager.execute_module(lambda m: isinstance(m, TTSModule), "global_response", buffer)
                                    break
                                
                                buffer += token
                                
                                # Search backwards for the last space or punctuation
                                last_idx = -1
                                for i in range(len(buffer) - 1, -1, -1):
                                    if buffer[i] in punctuation_and_space:
                                        last_idx = i
                                        break
                                
                                if last_idx != -1:
                                    speakable = buffer[:last_idx + 1]
                                    buffer = buffer[last_idx + 1:]
                                    if speakable.strip():
                                        self.logger.info(f"Speaking (Priority {priority}): {speakable}")
                                        AI_manager.execute_module(lambda m: isinstance(m, TTSModule), "global_response", speakable)
                            else:
                                # Wait briefly for more tokens
                                time.sleep(0.02)
                    elif text != "":
                        self.logger.info(f"Speaking (Priority {priority}): {text}")
                        AI_manager.execute_module(lambda m: isinstance(m, TTSModule), "global_response", text)
                
                self.message_queue.task_done()
            except queue.Empty:
                pass
            except Exception as e:
                self.logger.error(f"GlobalResponseModule loop error: {e}")
