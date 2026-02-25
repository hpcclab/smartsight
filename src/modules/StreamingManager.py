import gi
import sys
import threading
import numpy as np

gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")  # Required for appsink
from gi.repository import Gst, GstApp, GLib

# Import our shared buffer instance
from modules.shared_buffer import video_buffer

class CameraStream(threading.Thread):
    def __init__(self, ip_address="raspberrypi.local"):
        super().__init__()
        self.daemon = True  # Ensures thread closes when main program exits
        self.ip_address = ip_address
        self.rtsp_uri = f"rtsp://{self.ip_address}:8554/stream"
        
        Gst.init(sys.argv)
        self.loop = GLib.MainLoop()
        self.pipeline = None

    def run(self):
        """This runs in the background thread when stream.start() is called."""
        # pipeline description:
        # videoconvert ! video/x-raw,format=BGR converts the hardware decoded frame into OpenCV's BGR format.
        # appsink name=sink drop=true max-buffers=1 ensures we only keep the absolute newest frame.
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
            
            # Clean up the memory mapping
            buffer.unmap(map_info)
            
        return Gst.FlowReturn.OK