import logging
import sys
import os
import time

# Ensure the root of the project is in the python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.input_event_manager import InputEventManager
from modules.ai_manager import AI_manager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def main():
    print("Testing InputEventManager and Audio Pipeline.")
    print("Please make sure your microphone is enabled and ready.")
    # Start the Input Event Manager
    input_manager = InputEventManager()
    input_manager.start()
    time.sleep(18)
    input_manager.stop()
    

if __name__ == "__main__":
    main()
