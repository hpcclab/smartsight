import numpy as np
from paddleocr import PaddleOCR

def calculate_iou(box_a, box_b):
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])
    inter_area = max(0, x_b - x_a) * max(0, y_b - y_a)
    if inter_area == 0:
        return 0.0
    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    iou = inter_area / float(box_a_area + box_b_area - inter_area)
    return iou

class TextDetector:
    # Accept the shared OCR engine during initialization
    def __init__(self, ocr_engine, iou_threshold=0.02):
        self.ocr_engine = ocr_engine
        self.iou_threshold = iou_threshold

    def analyze_frame(self, yolo_results, model_names, min_confidence):
        if not yolo_results:
            return {}

        img = yolo_results[0].orig_img
        
        # Use the shared engine, but with rec=False to only run detection
        ocr_res = self.ocr_engine.ocr(img, cls=False, rec=False)

        if not ocr_res or not ocr_res[0]:
            return {}  

        text_boxes = []
        for line in ocr_res[0]:
            points = np.array(line).astype(int)
            x_min, y_min = np.min(points, axis=0)
            x_max, y_max = np.max(points, axis=0)
            text_boxes.append({'box': [x_min, y_min, x_max, y_max], 'associated': False})

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

        for text in text_boxes:
            for obj in yolo_objects:
                if calculate_iou(text['box'], obj['box']) > self.iou_threshold:
                    obj['has_text'] = True
                    text['associated'] = True
                    break
        
        announcements = {}
        for obj in yolo_objects:
            if obj['has_text']:
                announcement_key = f"{obj['name']} with text"
                announcements[announcement_key] = announcements.get(announcement_key, 0) + 1
        if any(not text['associated'] for text in text_boxes):
            announcements["text detected"] = 1

        return announcements
