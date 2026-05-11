import os
import cv2 as cv
import numpy as np
import threading
import queue
from .ai_module_base import BaseAIModel

class OCRModule(BaseAIModel):
    conf_threshold = 0.6
    min_length = 2

    def __init__(self):
        super().__init__("ocr")
        self.request_queue = queue.PriorityQueue()
        self.request_counter = 0
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def load_model(self):
        """
        The model is loaded inside the worker thread. 
        We just signify that it's 'loaded' here so the base class passes.
        """
        self.model = True

    @staticmethod
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

    @staticmethod
    def reading_order(boxes):
        centres = np.array([np.mean(b, axis=0) for b in boxes])               
        return np.lexsort((centres[:, 0], centres[:, 1] * 4)).tolist()

    @staticmethod
    def _prep(img: np.ndarray) -> np.ndarray:
        """Light contrast boost"""
        img = cv.resize(img, None, fx=1.6, fy=1.6, interpolation=cv.INTER_CUBIC)
        lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
        l, a, b = cv.split(lab)
        l = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
        merged = cv.merge((l, a, b))
        return np.ascontiguousarray(merged)

    def _worker_loop(self):
        # Force single-thread mode for OpenMP to reduce conflict chance
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        os.environ["OMP_NUM_THREADS"] = "1"
        
        from paddleocr import PaddleOCR
        
        ocr = None

        def init_model():
            self.logger.info("Initializing PaddleOCR in worker thread...")
            return PaddleOCR(
                lang="en",
                gpu=False,
                enable_mkldnn=False, # Keep this False to minimize issues
                use_angle_cls=True,
                use_doc_orientation_classify=True,
                use_doc_unwarping=True,
                use_textline_orientation=True,
                text_detection_model_name="PP-OCRv5_server_det",
                text_recognition_model_name="PP-OCRv5_server_rec",
                show_log=False,
                det_db_box_thresh=0.30,
                det_db_unclip_ratio=2.0,
                rec_image_shape="3,64,640",
                rec_batch_num=8,
            )

        while True:
            # Block until we receive a frame
            try:
                priority, req_id, frame, simple_detection, res_q = self.request_queue.get()
            except Exception:
                continue
            
            if ocr is None:
                ocr = init_model()

            if frame is None:
                res_q.put("No text detected.")
                continue

            img = self._prep(frame)
            
            if simple_detection:
                try:
                    ocr_res = ocr.ocr(img, rec=False, cls=False)
                    if ocr_res and ocr_res[0]:
                        res_q.put("Text detected.")
                    else:
                        res_q.put("No text detected.")
                except Exception as e:
                    self.logger.error(f"Simple OCR FAILED: {e}")
                    res_q.put("Error reading text.")
                continue

            try:
                ocr_res = ocr.ocr(img, cls=True)
            except Exception as e:
                self.logger.error(f"Thread Conflict Detected ({e}). Reloading Model on current thread...")
                ocr = init_model()
                try:
                    ocr_res = ocr.ocr(img, cls=True)
                except Exception as e2:
                    self.logger.error(f"OCR FAILED FINAL: {e2}")
                    res_q.put("Error reading text.")
                    continue

            if not ocr_res or not ocr_res[0]:
                res_q.put("No text detected.")
                continue

            kept = [
                (box, txt.strip())
                for box, (txt, conf) in ocr_res[0]
                if conf >= self.conf_threshold and len(txt) >= self.min_length
            ]
            if not kept:
                res_q.put("No text detected.")
                continue

            order = self.reading_order([b for b, _ in kept])
            ordered_text = [kept[i][1].replace(".", "") for i in order]
            if not ordered_text:
                res_q.put("No text detected.")
                continue

            OCROut = " ".join(ordered_text)
            res_q.put(OCROut)

    def run_inference(self, input_data, **kwargs) -> str:
        """
        Runs object detection on the input image using the background thread.
        Returns a string summary of detected objects.
        """
        simple_text_detection = kwargs.get("simple_text_detection", False)
        priority = 1 if not simple_text_detection else 2
        
        # We only pass actual numpy arrays or try to read them
        frame = None
        if isinstance(input_data, str):
             frame = cv.imread(input_data)
        else:
             frame = input_data
             
        # Create a unique result queue for this request
        res_q = queue.Queue()
        
        # Dispatch
        self.request_counter += 1
        self.request_queue.put((priority, self.request_counter, frame, simple_text_detection, res_q))
        
        # Wait for result with timeout (to prevent total lockup)
        try:
            # 10 seconds is plenty for PaddleOCR CPU on typical frames. Adjust if needed.
            result = res_q.get(timeout=10.0)
            return result
        except queue.Empty:
            self.logger.warning("OCR inference timed out.")
            return "OCR Module Timeout"