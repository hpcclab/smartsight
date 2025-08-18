# Adding optical flow support.

import socket
import struct
import cv2
import numpy as np
import sys
import threading
import queue
import time # For simulating processing time
import OpticalFlow

# --- Configuration ---
SERVER_IP = '0.0.0.0'
SERVER_PORT = 8000
HEADER_SIZE = struct.calcsize('<L') # Size of the header (4 bytes for unsigned long)

# --- Shared Resources (Queues and Events) ---
# Queue for raw image data from receiver to processor. Max size 1 to ensure latest frame.
raw_frame_queue = queue.Queue(maxsize=1)
# Queue for processed image data from processor to display. Max size 1 to ensure latest processed frame.
processed_frame_queue = queue.Queue(maxsize=1)
# Event to signal all threads to stop
stop_event = threading.Event()

def image_receiver_thread(connection):
    """
    Thread function to continuously receive raw image data from the socket.
    It puts the received raw image data into `raw_frame_queue`.
    """
    data_buffer = b''
    payload_size = 0

    print("Receiver thread started.")
    try:
        while not stop_event.is_set():
            # --- Step 1: Read the header (4 bytes) to get the image size ---
            # Continue receiving chunks until the header is complete
            while len(data_buffer) < HEADER_SIZE:
                try:
                    chunk = connection.recv(4096)
                    if not chunk:
                        print("Client disconnected or no more data in receiver thread.")
                        stop_event.set() # Signal other threads to stop
                        return
                    data_buffer += chunk
                except socket.error as e:
                    if stop_event.is_set(): # Check if we're shutting down gracefully
                        return
                    print(f"Socket error in receiver: {e}")
                    stop_event.set()
                    return

            # Once we have enough data for the header, unpack the payload size
            payload_size = struct.unpack('<L', data_buffer[:HEADER_SIZE])[0]
            data_buffer = data_buffer[HEADER_SIZE:] # Remove the header from the buffer

            # --- Step 2: Read the image data based on the payload size ---
            # Continue receiving chunks until the full image data is in the buffer
            while len(data_buffer) < payload_size:
                try:
                    chunk = connection.recv(4096)
                    if not chunk:
                        print("Client disconnected unexpectedly while receiving image data in receiver thread.")
                        stop_event.set()
                        return
                    data_buffer += chunk
                except socket.error as e:
                    if stop_event.is_set():
                        return
                    print(f"Socket error during image data reception: {e}")
                    stop_event.set()
                    return

            # Extract the complete image data
            image_data = data_buffer[:payload_size]
            data_buffer = data_buffer[payload_size:] # Keep any remaining data for the next frame

            # Put the raw image data into the queue for the processing thread
            # Use put_nowait to avoid blocking if the processing thread is slow,
            # ensuring we always have the latest frame.
            try:
                raw_frame_queue.put_nowait(image_data)
            except queue.Full:
                # If the queue is full, the processing thread hasn't picked up the last frame yet.
                # We discard the old one and put the new one. This effectively implements a
                # 'ping-pong' like behavior for the latest frame.
                try:
                    raw_frame_queue.get_nowait() # Remove the old frame
                except queue.Empty:
                    pass # Should not happen if put_nowait raised Full
                raw_frame_queue.put_nowait(image_data) # Add the new frame

    except Exception as e:
        print(f"Unexpected error in receiver thread: {e}")
        stop_event.set()
    finally:
        print("Receiver thread stopping.")

def image_processing_thread():
    """
    Thread function to continuously get raw image data from `raw_frame_queue`,
    decode it, perform image processing, and put the processed frame into
    `processed_frame_queue`.
    """
    print("Processing thread started.")
    try:
        image_data = []
        image_data_np = []
        frames = []
        old = 0
        # Parameters for Shi-Tomasi corner detection (for initial feature points)
        feature_params = dict(maxCorners=100,
                            qualityLevel=0.3,
                            minDistance=7,
                            blockSize=7)

        # Parameters for Lucas-Kanade optical flow
        lk_params = dict(winSize=(15, 15),
                        maxLevel=2,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        buffering = True
        while not stop_event.is_set():
            try:
                # Get raw image data from the queue. Timeout prevents infinite blocking on shutdown.
                # Always replace the old frame and update it to the new frame.
                if (len(image_data) - 1) < old:
                    image_data.append(raw_frame_queue.get(timeout=0.1))
                    # Convert the byte data to a NumPy array and decode it
                    image_data_np.append(np.frombuffer(image_data[old], dtype=np.uint8))
                    frames.append(cv2.imdecode(image_data_np[old], cv2.IMREAD_COLOR))
                else:
                    image_data[old] = raw_frame_queue.get(timeout=0.1)
                    # Convert the byte data to a NumPy array and decode it
                    image_data_np[old] = np.frombuffer(image_data[old], dtype=np.uint8)
                    frames[old] = cv2.imdecode(image_data_np[old], cv2.IMREAD_COLOR)
                    buffering = False
                new = old
                old = (old + 1) % 2
                if buffering:
                    continue
                # image_data2 = raw_frame_queue.get(timeout=0.1)

                # if (len(image_data) - 1) >= old:
                #     np_arr = np.frombuffer(image_data[old], dtype=np.uint8)
                # else:
                #     continue
                # frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                processed_frame = OpticalFlow.optical_flow_lk(frames[old], frames[new])

                if frames[0] is not None and frames[0].size > 0 and frames[1] is not None and frames[1].size > 0:
                    # --- Perform image processing here ---
                    # Example: Convert to grayscale (replace with your desired processing)
                    # processed_frame = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
                    # Example: Apply a simple blur
                    # processed_frame = cv2.GaussianBlur(processed_frame, (5, 5), 0)

                    # Simulate some processing time (remove in production if not needed)
                    # time.sleep(0.05)

                    # Put the processed frame into the queue for the display thread
                    try:
                        processed_frame_queue.put_nowait(processed_frame)
                    except queue.Full:
                        # If display thread is slow, drop old processed frame and add new one
                        try:
                            processed_frame_queue.get_nowait()
                        except queue.Empty:
                            pass
                        processed_frame_queue.put_nowait(processed_frame)
                else:
                    print("Processing thread received blank or invalid raw frame.")

            except queue.Empty:
                # No new raw frame available, continue checking
                pass
            except cv2.error as e:
                print(f"OpenCV error in processing thread: {e}")
            except Exception as e:
                print(f"Error in processing thread: {e}")
                stop_event.set() # Signal main thread to stop if critical error

    except Exception as e:
        print(f"Unexpected error in processing thread: {e}")
        stop_event.set()
    finally:
        print("Processing thread stopping.")


def run_server():
    """
    Main server function to set up socket, accept connection, and manage threads.
    It handles the display of processed frames.
    """
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    connection = None # Initialize connection to None

    try:
        server_socket.bind((SERVER_IP, SERVER_PORT))
        server_socket.listen(1)
        print(f"Server listening on {SERVER_IP}:{SERVER_PORT}")
        print("Waiting for client connection...")

        # Accept a connection from a client
        connection, client_address = server_socket.accept()
        print(f"Connected to client: {client_address}")

        # Start the receiver thread
        receiver_t = threading.Thread(target=image_receiver_thread, args=(connection,))
        receiver_t.daemon = True # Allow main program to exit even if thread is running
        receiver_t.start()

        # Start the processing thread
        processor_t = threading.Thread(target=image_processing_thread)
        processor_t.daemon = True
        processor_t.start()

        print("Main display loop started. Press 'Q' to quit.")
        while not stop_event.is_set():
            try:
                # Try to get the latest processed frame for display
                frame_to_display = processed_frame_queue.get(timeout=0.01) # Small timeout

                if frame_to_display is not None and frame_to_display.size > 0:
                    cv2.imshow('Live Processed Stream (Press Q to quit)', frame_to_display)
                else:
                    # This might happen if queue.get returns None after timeout or empty
                    pass # Just continue, no frame available yet

                # Wait for 1ms and check for 'q' key press to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("User quit requested. Signaling threads to stop.")
                    stop_event.set() # Signal threads to stop
                    break # Exit display loop

            except queue.Empty:
                # No processed frame available yet, or processing is slower than display loop
                pass
            except cv2.error as e:
                print(f"OpenCV error in display loop: {e}")
                stop_event.set()
            except Exception as e:
                print(f"Error in main display loop: {e}")
                stop_event.set()
                break

    except socket.error as e:
        print(f"Socket error in main server: {e}")
    except Exception as e:
        print(f"An unexpected error occurred in main server: {e}")
    finally:
        print("Closing connections and resources.")
        stop_event.set() # Ensure all threads are signaled to stop
        # Give threads a moment to finish before joining
        receiver_t.join(timeout=1)
        processor_t.join(timeout=1)
        if connection:
            connection.close()
        server_socket.close()
        cv2.destroyAllWindows()
        print("Server shutdown complete.")


if __name__ == '__main__':
    run_server()
