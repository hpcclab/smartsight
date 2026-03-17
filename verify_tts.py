import sys
import os
import logging

# Add src to path if needed (assuming running from root)
sys.path.append(os.path.join(os.getcwd(), 'src'))

from modules.TTS_module import TTSModule

def test_tts():
    logging.basicConfig(level=logging.INFO)
    print("Initializing TTSModule...")
    # Point to the config file (adjusted path from root)
    tts = TTSModule()
    
    test_text = "Verification successful. The Piper text to speech module is operational."
    print(f"Executing TTS with text: '{test_text}'")
    
    # execute() internally calls run_inference() and load_model()
    result = tts.execute("verification_test", test_text)
    
    print(f"Result (Output Path): {result}")
    
    if os.path.exists(result):
        print(f"SUCCESS: WAV file created at {result}")
    else:
        print(f"FAILURE: WAV file not found at {result}")

if __name__ == "__main__":
    test_tts()
