"""
Manual check for DollarDetectionAIModule against a webcam or a recording.

The real pipeline pulls frames over RTSP via GStreamer (see StreamingManager),
which needs the Pi and `gi`. This bypasses all of that and feeds frames straight
into the module, drawing the boxes so you can see what it is actually labelling.

    python Testing/dollarDetectionTest.py                                   # webcam
    python Testing/dollarDetectionTest.py --source Testing/dollarsRecording.mp4 --auto 0.5
    python Testing/dollarDetectionTest.py --backend local --min-conf 0.35

SPACE detects on the current frame, q quits. Detection is keypress-triggered by
default because the roboflow backend bills per call; --auto runs it on a timer,
which is what you want for a recording.

Boxes are green at or above min_conf (announced) and orange below it (found but
discarded) - the orange ones are how you tell a bad threshold from a bad model.
"""
import os
import sys
import time
import argparse

import cv2 as cv

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (os.path.join(_project_root, "src"), _project_root):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Model paths in config.yaml are relative to the repo root
os.chdir(_project_root)

from modules.dollar_detection_ai_module import DollarDetectionAIModule

FONT = cv.FONT_HERSHEY_SIMPLEX


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="0",
                        help="camera index or path to a video file (default: 0)")
    parser.add_argument("--backend", choices=["roboflow", "local"],
                        help="override dollar_detection.backend from config.yaml")
    parser.add_argument("--min-conf", type=float,
                        help="override dollar_detection.min_conf from config.yaml")
    parser.add_argument("--auto", type=float, metavar="SECONDS",
                        help="detect every SECONDS instead of waiting for SPACE")
    args = parser.parse_args()

    detector = DollarDetectionAIModule()
    if args.backend:
        detector.backend = args.backend
    # copy first: config is the shared dict handed out by get_config()
    detector.config = dict(detector.config)
    if args.min_conf is not None:
        detector.config["min_conf"] = args.min_conf
    min_conf = float(detector.config.get("min_conf", 0.5))
    # Detect below the threshold too, so the overlay can show what it discarded
    detector.config["min_conf"] = min(min_conf, 0.05)

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(f"Could not open source {args.source!r}.")

    print(f"Backend: {detector.backend} | min_conf: {min_conf} | source: {args.source}")
    print("SPACE to detect, q to quit." if not args.auto else f"Auto-detecting every {args.auto}s.")

    boxes, caption, last_run = [], "", 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("End of source.")
                break

            now = time.monotonic()
            key = cv.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" ") or (args.auto and now - last_run >= args.auto):
                last_run = now
                boxes = detector.detect_boxes(frame)
                announced = [b for b in boxes if b[1] >= min_conf]
                caption = detector.summarise(announced)
                pos = cap.get(cv.CAP_PROP_POS_MSEC) / 1000
                print(f"t={pos:6.1f}s  {caption}"
                      + (f"   [discarded: {_fmt(b for b in boxes if b[1] < min_conf)}]"
                         if len(boxes) != len(announced) else ""))

            for name, conf, (x1, y1, x2, y2) in boxes:
                colour = (0, 255, 0) if conf >= min_conf else (0, 165, 255)
                cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), colour, 3)
                cv.putText(frame, f"{name} {conf:.2f}", (int(x1), max(24, int(y1) - 8)),
                           FONT, 0.8, colour, 2)
            cv.putText(frame, caption or "SPACE to detect, q to quit",
                       (10, 30), FONT, 0.7, (0, 255, 0), 2)
            cv.imshow("Dollar Detection Test", frame)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv.destroyAllWindows()


def _fmt(dets):
    return ", ".join(f"{n} {c:.2f}" for n, c, _ in dets) or "none"


if __name__ == "__main__":
    main()
