import gi
import sys
import threading
import numpy as np

gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")  # Required for appsink
from gi.repository import Gst, GstApp, GLib

from modules.shared_buffer import video_buffer
from config.config import get_config
import cv2
import os
import time

class CameraStream(threading.Thread):
    def __init__(self, ip_address="raspberrypi.local"):
        super().__init__()
        self.daemon = True  # Ensures thread closes when main program exits
        self.ip_address = ip_address
        self.rtsp_uri = f"rtsp://{self.ip_address}:8554/stream"
        
        # Load configurations
        self.config = get_config().get("input", {})
        self.simulation_mode = self.config.get("simulation_mode_enabled", False)
        # self.looping_enabled = self.config.get("simulation_looping_enabled", True)
        self.recording_enabled = self.config.get("recording_enabled", False)
        self.video_name = self.config.get("video_name", "test_video.mp4")
        
        # Determine the absolute path to the Testing directory
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.testing_dir = os.path.join(self.base_dir, "Testing")
        if self.recording_enabled and not os.path.exists(self.testing_dir):
            os.makedirs(self.testing_dir, exist_ok=True)
            
        self.video_writer = None
        
        Gst.init(sys.argv)
        self.loop = GLib.MainLoop()
        self.pipeline = None

    def run(self):
        """This runs in the background thread when stream.start() is called."""
        # pipeline description:
        # videoconvert ! video/x-raw,format=BGR converts the hardware decoded frame into OpenCV's BGR format.
        # appsink name=sink drop=true max-buffers=1 ensures we only keep the absolute newest frame.
        
        if self.simulation_mode:
            video_path = os.path.join(self.testing_dir, self.video_name)
            # Use filesrc and sync=true for simulation playback at normal speed.
            # Using drop=true max-buffers=1 to align with appsink pattern.

            pipeline_str = (
                f"multifilesrc location={video_path.replace(chr(92), '/')} ! decodebin ! "
                "videoconvert ! video/x-raw,format=BGR ! "
                "appsink name=sink emit-signals=true max-buffers=1 drop=true sync=true"
            )

        else:
            pipeline_str = (
                f"rtspsrc location={self.rtsp_uri} latency=0 drop-on-latency=true ! "
                "rtph264depay ! h264parse ! decodebin ! "
                "videoconvert ! video/x-raw,format=BGR ! "
                "appsink name=sink emit-signals=true max-buffers=1 drop=true sync=false"
            )

        self.pipeline = Gst.parse_launch(pipeline_str)

        # Connect the bus to handle errors/EOS
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.on_message)

        # Get the appsink element and connect the 'new-sample' signal
        appsink = self.pipeline.get_by_name("sink")
        appsink.connect("new-sample", self.on_new_sample)

        print(f"Connecting to {self.rtsp_uri} in background thread...")
        self.pipeline.set_state(Gst.State.PLAYING)
        
        # This will block the background thread, keeping GStreamer alive
        self.loop.run()

    def stop(self):
        """Safely shuts down the pipeline and the thread."""
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
        if self.loop.is_running():
            self.loop.quit()
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

    def on_message(self, bus, message):
        """Handles pipeline bus messages."""
        t = message.type
        if t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"GStreamer Error: {err}, {debug}")
            self.stop()
        elif t == Gst.MessageType.EOS:
            print("End-Of-Stream reached.")
            self.stop()
        return True

    def on_new_sample(self, sink):
        """Callback triggered every time a new frame arrives at appsink."""
        sample = sink.emit("pull-sample")
        if not sample:
            return Gst.FlowReturn.ERROR

        # Extract the buffer and caps (metadata like width/height)
        buffer = sample.get_buffer()
        caps = sample.get_caps()
        
        # Extract width and height from caps
        structure = caps.get_structure(0)
        width = structure.get_value("width")
        height = structure.get_value("height")

        # Map the GStreamer buffer to read the raw memory
        success, map_info = buffer.map(Gst.MapFlags.READ)
        if success:
            # Convert the raw memory into a NumPy array for OpenCV (Height x Width x 3 channels for BGR)
            frame = np.ndarray(
                shape=(height, width, 3),
                dtype=np.uint8,
                buffer=map_info.data
            )
            
            # Push the OpenCV frame into our shared PingPongBuffer
            video_buffer.update(frame)
            
            if self.recording_enabled:
                if self.video_writer is None:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = os.path.join(self.testing_dir, f"recording_{timestamp}.mp4")
                    # Try to retrieve fps from caps, otherwise default to 30.
                    fps = 30.0
                    if structure.has_field("framerate"):
                        # 'framerate' is typically a Gst.Fraction
                        fraction = structure.get_value("framerate")
                        if fraction.denom > 0:
                            fps = fraction.num / fraction.denom
                    # fallback if extraction fails
                    if fps <= 0:
                        fps = 30.0
                    
                    self.video_writer = cv2.VideoWriter(
                        filename, 
                        cv2.VideoWriter_fourcc(*'mp4v'), 
                        fps, 
                        (width, height)
                    )
                self.video_writer.write(frame)
            
            # Clean up the memory mapping
            buffer.unmap(map_info)
            
        return Gst.FlowReturn.OK