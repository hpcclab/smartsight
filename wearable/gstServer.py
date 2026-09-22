import gi
import sys

gi.require_version("Gst", "1.0")
gi.require_version("GstRtspServer", "1.0")
from gi.repository import Gst, GstRtspServer, GLib

# Initialize GStreamer
Gst.init(sys.argv)

class CameraRTSPServer:
    def __init__(self):
        # Create the RTSP Server
        self.server = GstRtspServer.RTSPServer()
        self.server.set_service("8554")  # Standard RTSP port

        # Create a Media Factory
        factory = GstRtspServer.RTSPMediaFactory()

        # The pipeline string
        # We use v4l2h264enc for hardware accelerated encoding, and rtph264pay to package it for RTSP.
        # The payloader MUST be named "pay0" for the RTSP server to find it.
        pipeline_str = (
            "libcamerasrc ! "
            "video/x-raw,width=1280,height=720,framerate=30/1,format=NV12,interlace-mode=progressive ! "
            "v4l2h264enc extra-controls=\"controls,repeat_sequence_header=1,video_bitrate=10000000\" ! "
            "video/x-h264,profile=high,level=(string)4 ! "
            "h264parse ! "
            "rtph264pay name=pay0 pt=96 config-interval=1"
        )
        #'videoconvert ! v4l2h264enc extra-controls="encode,h264_profile=4,h264_level=10" ! '
        factory.set_launch(pipeline_str)

        # Set to True so multiple clients can view the same stream without opening the camera multiple times
        factory.set_shared(True)

        # Attach the factory to a mount point (e.g., /stream)
        mounts = self.server.get_mount_points()
        mounts.add_factory("/stream", factory)

        # Attach the server to the default main context
        self.server.attach(None)

        print("RTSP Server is running!")
        print("Connect using: rtsp://<your_pi_ip_address>:8554/stream")

# Start the server and the main loop
server = CameraRTSPServer()
loop = GLib.MainLoop()

try:
    loop.run()
except KeyboardInterrupt:
    print("\nShutting down RTSP server...")