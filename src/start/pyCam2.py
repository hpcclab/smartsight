import io
import socket
import struct
import time
import os
from picamera2 import Picamera2
from picamera2.encoders import JpegEncoder
from picamera2.outputs import FileOutput

# --- Configuration ---
# IMPORTANT: Change this to your server's IP address and chosen port.
# Ensure the port matches the server's listening port.
IPFile = "LaptopIP.txt"
if os.path.exists(IPFile):
    with open(IPFile, "r") as f:
        lines = f.readlines()
        SERVER_IP = lines[0].strip()
else:
    # Set a default IP if the file doesn't exist, or handle the error
    SERVER_IP = '127.0.0.1' 
    print(f"Warning: '{IPFile}' not found. Using default IP: {SERVER_IP}")


SERVER_PORT = 8000
RESOLUTION = (640, 480) # Reduced resolution for faster transmission
FRAMERATE = 25 # Frames per second

def run_client():
    """
    Connects to the server, captures video frames using Picamera2,
    and streams them over the network as JPEGs.
    Each frame is sent with a preceding 4-byte length header.
    """
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print(f"Attempting to connect to {SERVER_IP}:{SERVER_PORT}...")

    try:
        client_socket.connect((SERVER_IP, SERVER_PORT))
        print(f"Successfully connected to {SERVER_IP}:{SERVER_PORT}")

        # Use Picamera2 in a 'with' statement for proper resource management
        with Picamera2() as picam2:
            # 1. Create a configuration object
            # The 'main' stream is used for stills and video
            config = picam2.create_video_configuration(main={"size": RESOLUTION})
            picam2.configure(config)

            # 2. Set the framerate (optional, but good for consistency)
            picam2.set_controls({"FrameRate": FRAMERATE})
            
            # 3. Start the camera. This handles the 'warm-up' time internally.
            picam2.start()
            print(f"Camera initialized with resolution {RESOLUTION} and framerate {FRAMERATE}.")
            
            # Create an in-memory stream to hold image data
            stream = io.BytesIO()

            print("Starting video capture and streaming...")
            # 4. Capture continuously in a loop
            while True:
                try:
                    # Capture a single frame to the stream as a JPEG
                    # This is the modern equivalent of the 'capture_continuous' loop's body
                    picam2.capture_file(stream, format='jpeg')

                    # Get the raw image data from the stream
                    image_data = stream.getvalue()
                    image_size = len(image_data)

                    # Pack the image size into a 4-byte unsigned long (little-endian)
                    # This tells the receiver how many bytes to expect for the image
                    size_bytes = struct.pack('<L', image_size)

                    # Send the size first, then the image data
                    client_socket.sendall(size_bytes)
                    client_socket.sendall(image_data)

                    # Reset the stream for the next frame
                    stream.seek(0)
                    stream.truncate()

                except BrokenPipeError:
                    print("Connection broken. Server disconnected or pipe closed.")
                    break
                except Exception as e:
                    print(f"Error during streaming: {e}")
                    break
                    
    except socket.error as e:
        print(f"Socket connection error: {e}")
    except Exception as e:
        # Catching a general Exception is more robust for Picamera2,
        # as its specific errors might differ from the old library.
        print(f"An unexpected error occurred: {e}. Make sure the camera is enabled and connected.")
    finally:
        print("Closing client socket.")
        client_socket.close()

if __name__ == '__main__':
    run_client()