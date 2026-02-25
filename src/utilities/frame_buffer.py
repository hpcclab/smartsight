import numpy as np
import threading
import copy

class PingPongBuffer:
    def __init__(self):
        # Our two buffer slots: index 0 and index 1
        self._buffers = [None, None]
        
        # Points to the slot holding the most recently written frame
        self._current_write_idx = 0 
        
        # Lock to prevent read/write collisions during the swap
        self._lock = threading.Lock()

    def update(self, new_frame: np.ndarray):
        """
        Receives a new OpenCV Mat (numpy array) and stores it.
        The previously written frame automatically becomes the 'previous' frame.
        """
        with self._lock:
            # Toggle the index: 0 becomes 1, 1 becomes 0
            self._current_write_idx = 1 - self._current_write_idx
            
            # Write the new frame into the active write slot.
            # We use .copy() to ensure GStreamer doesn't overwrite the memory 
            # while OpenCV is trying to process it later.
            self._buffers[self._current_write_idx] = new_frame.copy()

    def retrieve_frame(self) -> np.ndarray:
        """
        Returns the most recent frame for computer vision tasks.
        """
        with self._lock:
            frame = self._buffers[self._current_write_idx]
            return frame.copy() if frame is not None else None

    def retrieve_both(self):
        """
        Returns both the current and previous frames.
        Highly useful for motion detection, optical flow, or background subtraction.
        Returns: (current_frame, previous_frame)
        """
        with self._lock:
            current_frame = self._buffers[self._current_write_idx]
            previous_frame = self._buffers[1 - self._current_write_idx]
            
            return (
                current_frame.copy() if current_frame is not None else None,
                previous_frame.copy() if previous_frame is not None else None
            )