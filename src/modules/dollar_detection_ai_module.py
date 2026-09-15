import os
import base64
import cv2 as cv
import requests
from collections import Counter
from .ai_module_base import BaseAIModel

class DollarDetectionAIModule(BaseAIModel):

    DENOMINATIONS = {
        "one": 1,
        "five": 5,
        "ten": 10,
        "twenty": 20,
        "fifty": 50,
    }

    ROBOFLOW_CLASS_MAP = {
        "1": "fifty",
        "2": "five",
        "3": "five",      # never observed in probing; by elimination
        "4": "one",
        "5": "one",
        "6": "ten",
        "7": "ten",
        "8": "twenty",
        "9": "twenty",
        "20": "fifty",    # fifty-back, despite the name
    }

    NO_DETECTION = "No dollar bills detected."
    ERROR = "An unexpected error occurred during dollar detection."

    def __init__(self):
        super().__init__("dollar_detection")
        self.backend = str(self.config.get("backend", "roboflow")).lower()
        self.class_map = {str(k): v for k, v in
                          self.config.get("roboflow_class_map", self.ROBOFLOW_CLASS_MAP).items()}
        self._session = None

    def _api_key(self):
        """Env var wins so the key can stay out of the config file entirely."""
        return os.environ.get("ROBOFLOW_API_KEY") or self.config.get("roboflow_api_key", "")

    def load_model(self):
        """Prepares whichever backend is configured."""
        if self.backend == "local":
            from ultralytics import YOLO
            model_path = self.config.get("model_path", "models/dollar_detection/best.pt")
            self.logger.info(f"Loading local dollar detection model from {model_path}...")
            self.model = YOLO(model_path)
            return

        if not self._api_key():
            self.logger.error(
                "No Roboflow API key. Set the ROBOFLOW_API_KEY environment variable "
                "or add dollar_detection.roboflow_api_key to configSensitive.yaml."
            )
        self._session = requests.Session()
        self.model = self._session
        self.logger.info(f"Dollar detection using hosted model {self.config.get('roboflow_model_id', 'dollar-bill-a5fkm/1')}")

    def _detect_roboflow(self, frame) -> list:
       
        api_key = self._api_key()
        if not api_key:
            raise RuntimeError("Roboflow API key is not configured.")

        quality = int(self.config.get("jpeg_quality", 80))
        ok, buf = cv.imencode(".jpg", frame, [int(cv.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            raise RuntimeError("Failed to JPEG-encode the frame.")

        model_id = self.config.get("roboflow_model_id", "dollar-bill-a5fkm/1")
        api_url = self.config.get("roboflow_api_url", "https://serverless.roboflow.com")
        min_conf = float(self.config.get("min_conf", 0.5))

        response = self._session.post(
            f"{api_url}/{model_id}",
            params={"confidence": min_conf},
            # Key goes in the Authorization header, never the query string
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data=base64.b64encode(buf.tobytes()).decode(),
            timeout=float(self.config.get("roboflow_timeout", 10)),
        )
        response.raise_for_status()

        detections = []
        for pred in response.json().get("predictions", []):
            conf = float(pred.get("confidence", 0.0))
            if conf < min_conf:
                continue
            class_name = str(pred.get("class", ""))
            denomination = self.class_map.get(class_name)
            if not denomination:
                self.logger.warning(f"Unmapped Roboflow class: {class_name!r}")
                continue
            # Roboflow returns center-x/center-y/width/height
            x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
            detections.append((denomination, conf,
                               (x - w / 2, y - h / 2, x + w / 2, y + h / 2)))
        return detections

    def _detect_local(self, frame) -> list:
        """Runs the bundled YOLO weights.

        Returns a list of (denomination, confidence, (x1, y1, x2, y2)).
        """
        min_conf = float(self.config.get("min_conf", 0.5))
        results = self.model(frame, verbose=False)

        detections = []
        if results:
            for box in results[0].boxes:
                conf = box.conf.item()
                if conf < min_conf:
                    continue
                class_name = self.model.names[int(box.cls[0])]
                denomination = class_name.split("-")[0]  # "twenty-front" -> "twenty"
                if denomination not in self.DENOMINATIONS:
                    self.logger.warning(f"Unrecognized class name: {class_name}")
                    continue
                detections.append((denomination, conf, tuple(box.xyxy[0].tolist())))
        return detections

    def detect_boxes(self, frame) -> list:
        """Raw detections as (denomination, confidence, (x1, y1, x2, y2)).

        run_inference() is the announcement path; this is for tooling that needs
        the boxes, e.g. Testing/dollarDetectionTest.py drawing an overlay.
        """
        if self.model is None:
            self.load_model()
        return self._detect_local(frame) if self.backend == "local" else self._detect_roboflow(frame)

    def summarise(self, detections) -> str:
        """Phrases detections the way object detection does: singular labels,
        count only when > 1, largest denomination first.

        The passive detector parses this back apart and re-pluralizes, so the
        labels stay singular here (see PassiveDetectorModule._parse_detections).
        """
        bill_counts = Counter(d[0] for d in detections)
        if len(bill_counts) < 1:
            return self.NO_DETECTION

        ordered = sorted(bill_counts.items(),
                         key=lambda item: self.DENOMINATIONS[item[0]], reverse=True)

        output_parts = []
        for denomination, count in ordered:
            label = f"{denomination} dollar bill"
            output_parts.append(f"{count} {label}" if count > 1 else label)
        return ", ".join(output_parts)

    def run_inference(self, input_data, **kwargs) -> str:
        """
        Runs dollar bill detection on the input image path or frame.
        Returns a string summary of detected bills.
        """
        try:
            if isinstance(input_data, str):
                frame = cv.imread(input_data)
            else:
                frame = input_data  # Assume it's already an image array

            if frame is None or frame.size == 0:
                self.logger.warning("Empty frame received or could not load image.")
                return self.NO_DETECTION

            if self.model is None:
                self.load_model()

            detections = (self._detect_local(frame) if self.backend == "local"
                          else self._detect_roboflow(frame))
            return self.summarise(detections)

        except requests.Timeout:
            self.logger.error("Roboflow request timed out.")
            return self.ERROR
        except Exception as e:
            self.logger.error(f"Error during inference: {e}")
            return self.ERROR
