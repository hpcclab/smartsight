# operations/commands.py  ──────────────────────────────────────────────
import cv2 as cv
import numpy as np
from paddleocr import PaddleOCR


# OCR settings
def build_ocr():
    """PP-OCR v5  • mobile detector  +  mobile recogniser (tiny & fast)."""
    return PaddleOCR(
        lang="en",
        gpu=False,                  
        use_angle_cls=True,            
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="PP-OCRv5_mobile_rec",
        show_log=False,
        rec_batch_num=4,               
    )
# Reading order helper
def reading_order(boxes):
    centres = np.array([np.mean(b, axis=0) for b in boxes])               
    return np.lexsort((centres[:, 0], centres[:, 1] * 4)).tolist()
class Commands:
    def __init__(self, ocr, conf_threshold=0.6, min_length=2):
        self.ocr = ocr
        self.conf_threshold = conf_threshold
        self.min_length = min_length

    @staticmethod
    def _prep(img: np.ndarray) -> np.ndarray:
        """Light contrast boost"""
        img = cv.resize(img, None, fx=1.6, fy=1.6,
                        interpolation=cv.INTER_CUBIC)
        lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
        l, a, b = cv.split(lab)
        l = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
        return cv.merge((l, a, b))

    def read_text(self, img_path: str):
        img = cv.imread(img_path)
        if img is None:
            return []

        img = self._prep(img)

        ocr_res = self.ocr.ocr(img, cls=True)
        if not ocr_res or not ocr_res[0]:
            return []

        # keep confident results
        kept = [
            (box, txt.strip())
            for box, (txt, conf) in ocr_res[0]
            if conf >= self.conf_threshold and len(txt) >= self.min_length
        ]
        if not kept:
            return []

        order = reading_order([b for b, _ in kept])
        ordered_text = [kept[i][1].rstrip(".") for i in order]  # drop stray dots
        return ordered_text
