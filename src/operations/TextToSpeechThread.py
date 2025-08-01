import threading
import queue
import pyttsx3
import logging # Use logging instead of print for better control
import time 

#Priority definition 
UrgentPassive = -10
ActiveThread = 0
PassiveThread = 1
class TTS(threading.Thread):
    #Constructor for the thread with daemon set to True 
    def __init__(self, message_queue: queue.PriorityQueue, name = 'TextToSpeech', daemon = True ):
        self.message_queue = message_queue
        self.name = name 
        self.daemon = True 
        self.running = False 
        print(f"{self.name} initialied successfully!")
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
        self.InitializeEngine()
        self.running = True 
        if not self.engine():
            print(f"{self.name} Engine not found ERROR")
            return  
        while self.running:
            if message == "SHUTDOWN_SIGNAL":
                print(f"Receiving Shutdown Signal.")
                self.running = False 
            try:
                priority, message = self.message_queue.get(block=True, timeout=0.1)
                print(f"Run priority {priority}, Say message: {message}")
                self.engine.say(message)
                self.engine.runAndWait()
                self.message_queue.all_tasks_done()
            except queue.Empty():
                pass
            except Exception as e:
                print(f"{self.name} received ERROR: {e}")
        self.cleanup()

    def stop(self):
        print(f"Receive Shutting down signal ")
        self.running = False 
        self.message_queue.put(UrgentPassive -1, "SHUTDOWN_SIGNAL")
        

    def cleanup(self):
        if self.engine: 
            try:
                self.engine.stop()
                self.engine.endLoop()
                self.engine = None 
            except Exception as e:
                print(f"{self.name} failed to perform clean up ERROR: {e}")



if __name__ == "__main__":
    pass 

        
    








