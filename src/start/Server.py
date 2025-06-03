import socket
import time
from pathlib import Path

HOST = "0.0.0.0"  # Listen on all interfaces
PORT = 8888  # Arbitrary port
iter = 111

script_dir = Path(__file__).resolve().parent
temp_dir   = (script_dir.parent.parent / 'temp').resolve()

def runServer():
    #Make sure temp directory exists
    temp_dir.mkdir(parents=True, exist_ok=True)
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((HOST, PORT))
            print("Ready to recieve")
            s.listen(1)
            conn, addr = s.accept()
            with conn:
                print('Connected by', addr)
                # Write incoming bytes into temp/image.jpg
                output_path = temp_dir/"image.jpg"
                with open(output_path, "wb") as file:
                    while True:
                        data = conn.recv(1024)
                        if not data: break
                        file.write(data)
            print("File received successfully")


# Single tcp connection

# def runServer():
#     while True:
#         with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
#             server_socket.bind((HOST, PORT))
#             server_socket.listen()
#             print(f"Listening on {HOST}:{PORT}...")
            
#             conn, addr = server_socket.accept()
#             with conn:
#                 print(f"Connected by {addr}")
#                 file = open("image.jpg", "wb")
#                 FileOpen = True
#                 while conn:
#                     data = conn.recv(1024)
#                     if data == b"ENDIMG" or data == None or data == "":
#                         if FileOpen:
#                             file.close()
#                             FileOpen = False
#                             print("File closed.")
#                     else:
#                         if not FileOpen:
#                             file = open("image.jpg", "wb")
#                             FileOpen = True
#                         file.write(data)
#                         print("File opened.")

# def runServer():
#     while True:
#         with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
#             server_socket.bind((HOST, PORT))
#             server_socket.listen()
#             print(f"Listening on {HOST}:{PORT}...")
            
#             conn, addr = server_socket.accept()
#             with conn:
#                 print(f"Connected by {addr}")
#                 with open("image.jpg", "wb") as file:
#                     while data := conn.recv(1024):
#                         file.write(data)
#                 print("File received successfully!")
