import sys
import os
import time
import pytest
from unittest.mock import MagicMock

# Setup path so src can be imported
_script_dir = os.path.dirname(__file__)
_project_root = os.path.abspath(os.path.join(_script_dir, '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
_src_dir = os.path.join(_project_root, 'src')
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from src.modules.global_response_module import GlobalResponseModule
from src.modules.ai_manager import AIManager

def test_streaming_and_parsing():
    # Mock execute_module to capture spoken items
    spoken_items = []
    
    # We will import and mock the actual AI_manager that modules are using
    from modules.ai_manager import AI_manager
    original_execute = AI_manager.execute_module
    
    def mock_execute_module(condition, input_key, *args, **kwargs):
        if args:
            spoken_items.append(args[0])
        return "mock_wav.wav"

    AI_manager.execute_module = mock_execute_module

    try:
        # Initialize GlobalResponseModule
        module = GlobalResponseModule()
        module.start()

        # Add a streaming list of tokens
        token_list = []
        module.add_message(token_list, priority=5)

        # Append tokens with short sleeps to simulate real streaming
        tokens = ["He", "llo", ",", " this", " is", " a", " test", " of", " streaming", "."]
        for token in tokens:
            token_list.append(token)
            time.sleep(0.05)
            
        token_list.append(None) # Sentinel

        # Wait for the update loop to finish processing
        start_time = time.time()
        while len(spoken_items) < 6 and time.time() - start_time < 3.0:
            time.sleep(0.05)

        # Stop the module
        module.stop()

        # Check what was spoken
        print(f"Spoken items: {spoken_items}")
        # Expected spoken items should be words split by punctuation/space
        # "Hello," (space/punctuation after "llo,")
        # "this " (space after "this")
        # "is " (space after "is")
        # "a " (space after "a")
        # "test " (space after "test")
        # "of " (space after "of")
        # "streaming." (punctuation after "streaming")
        spoken_text = " ".join(spoken_items)
        assert "Hello" in spoken_text
        assert "this" in spoken_text
        assert "is" in spoken_text
        assert "a" in spoken_text
        assert "test" in spoken_text
        assert "streaming" in spoken_text
        
    finally:
        # Restore original execute_module
        AI_manager.execute_module = original_execute

if __name__ == "__main__":
    test_streaming_and_parsing()
