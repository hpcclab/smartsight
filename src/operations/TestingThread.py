import threading
import json
import queue 
import time
# from operations.TextToSpeechThread import TTSThread

class TestingThread(threading.Thread):
    def __init__(self, callback, daemon=True):
        super().__init__(daemon=daemon, name="TestingThread")
        self.callback = callback
        self.stop_running = threading.Event()
        print(f"{self.name} initialized!")

    def run(self):
        #Json Data source 
        data_file = r'src\operations\testData.json'
        try:
            with open(data_file,"r") as file:
                data = json.load(file)
            for item in data:
                #Adding messages and priority into the message queue priority (Urgent Passive) 
                self.callback(item['text'], item['priority'])
        except FileNotFoundError:
            print(f"File Not Found")
        except Exception as e:
            print(f"{self.name} Found Error Opening Json file | Error: {e}")
    def stop(self):
        self.stop_running.set()
        print(f"{self.name} has stopped successfully")

# class Receiver:
#     def handle_message(self, priority, text):
#         print(f"Got message [{priority}]: {text}")


# receiver = Receiver()
# thread = TestingThread(callback=receiver.handle_message)
# thread.start()
# thread.join()