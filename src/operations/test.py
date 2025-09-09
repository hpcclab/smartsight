import json 
import queue 
data_file = r'src\operations\testData.json'
with open(data_file,"r") as file:
    data = json.load(file)
msg_queue = queue.PriorityQueue()
for item in data:
    msg_queue.put((item['priority'], item['text']))

while not msg_queue.empty():
    priority, message = msg_queue.get()
    print(f"Priority {priority}, Message: {message}")