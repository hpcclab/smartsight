import threading
import queue
import pyttsx3
import logging # Use logging instead of print for better control
import time 

class TTS(threading.Thread):
    #Constructor for the thread with daemon set to True 
    def __init__(self, message_queue: queue.PriorityQueue, name = 'TextToSpeech', daemon = True ):
        super().__init__(message_queue, name, daemon=daemon)

    #set up the TTS voice annoucment 
    def InitializeEngine(self):
        try:
            self.engine = pyttsx3.init()
            voices = self.engine.getProperty('voices')
            self.engine.setProperty('voice', voices[1].id)
            self.engine.setProperty('volume', 1.0)
            self.engine.setProperty('rate', 150)
            print("Initializing successfully...")
        except Exception as e:
            print(f"Fail to initialize TTS engine: {e}")
            self.engine = None 
    
    def run(self):
        pass 

    def stop(self):
        pass 

    def cleanup(self):
        pass 


if __name__ == "__main__":
    pass 

        
    








