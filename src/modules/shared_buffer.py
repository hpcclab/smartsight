from utilities.frame_buffer import PingPongBuffer

# Instantiate the buffer once. 
# Any module that imports this will access the same instance.
video_buffer = PingPongBuffer()