import numpy as np
from paddleocr import PaddleOCR

def build_fast_ocr_detector():
    """
    Builds a lightweight PaddleOCR model optimized for fast text detection only.
    This is ideal for passive mode operation.
    """
    return PaddleOCR(
        lang="en",
        gpu=False,
        use_angle_cls=False,
        # Disable all recognition and enhancement parts for maximum speed
        text_recognition_model_name=None, 
        show_log=False,
        # Use the fast mobile detection model
        text_detection_model_name="PP-OCRv5_mobile_det"
    )

def calculate_iou(box_a, box_b):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    Boxes are in format [x_min, y_min, x_max, y_max].
    """
    # Determine the coordinates of the intersection rectangle
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    # Compute the area of intersection, return 0 if no overlap
    inter_area = max(0, x_b - x_a) * max(0, y_b - y_a)
    if inter_area == 0:
        return 0.0

    # Compute the area of both bounding boxes
    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    # Compute the IoU and return it
    iou = inter_area / float(box_a_area + box_b_area - inter_area)
    return iou

class TextDetector:
    def __init__(self, iou_threshold=0.02):
        """
        Initializes the text detector with a fast OCR model and IoU threshold.
        A low IoU threshold is used to associate text that is 'near' an object.
        """
        self.ocr_detector = build_fast_ocr_detector()
        self.iou_threshold = iou_threshold

    def analyze_frame(self, yolo_results, model_names, min_confidence):
        """
        Detects text and correlates it with YOLO objects.

        Args:
            yolo_results: The raw results object from the YOLO model.
            model_names: A list of class names from the YOLO model.
            min_confidence: The confidence threshold for considering a YOLO detection.

        Returns:
            A dictionary of announcements to be added to the object counts.
            e.g., {"person with text": 1, "text detected": 1}
        """
        if not yolo_results:
            return {}

        img = yolo_results[0].orig_img
        # Use PaddleOCR for text detection only (cls=False, rec=False)
        ocr_res = self.ocr_detector.ocr(img, cls=False, rec=False)

        if not ocr_res or not ocr_res[0]:
            return {}  # No text found in the frame

        # Get bounding boxes for all detected text regions
        text_boxes = []
        for line in ocr_res[0]:
            points = np.array(line).astype(int)
            x_min, y_min = np.min(points, axis=0)
            x_max, y_max = np.max(points, axis=0)
            text_boxes.append({'box': [x_min, y_min, x_max, y_max], 'associated': False})

        # Get bounding boxes for confident YOLO objects
        yolo_objects = []
        for box in yolo_results[0].boxes:
            if box.conf >= min_confidence:
                class_id = int(box.cls[0])
                class_name = model_names[class_id]
                yolo_objects.append({
                    'name': class_name,
                    'box': box.xyxy[0].cpu().numpy().astype(int),
                    'has_text': False
                })

        # Correlate text boxes with YOLO object boxes
        for text in text_boxes:
            for obj in yolo_objects:
                if calculate_iou(text['box'], obj['box']) > self.iou_threshold:
                    obj['has_text'] = True
                    text['associated'] = True
                    # A text box is associated with the first object it overlaps with
                    break
        
        announcements = {}
        # Prepare announcements for objects that have associated text
        for obj in yolo_objects:
            if obj['has_text']:
                announcement_key = f"{obj['name']} with text"
                announcements[announcement_key] = announcements.get(announcement_key, 0) + 1

        # If any text was found that was NOT on a detected object, add a general announcement
        if any(not text['associated'] for text in text_boxes):
            announcements["text detected"] = 1

        return announcements
