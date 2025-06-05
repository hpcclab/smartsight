import cv2 as cv
import numpy as np
from paddleocr import PaddleOCR

def build_ocr():
    """
    PP-OCR v5  • server detector  +  server recogniser
    ≈ 62 MB total; best accuracy you can get without extra deps.
    """
    return PaddleOCR(
        lang="en",
        gpu=False,
        # preprocessing helpers
        use_angle_cls=True,
        use_doc_orientation_classify=True,
        use_doc_unwarping=True,
        use_textline_orientation=True,     # <-- enable line-angle fix
        # model choice
        text_detection_model_name="PP-OCRv5_server_det",   # <── changed
        text_recognition_model_name="PP-OCRv5_server_rec",
        # fine-tunes (slightly looser to catch skinny boxes)
        det_db_box_thresh=0.35,            # ↓ a bit
        det_db_unclip_ratio=2.0,           # ↑ a bit
        rec_image_shape="3,64,512",
        rec_batch_num=6,                  # batch more crops at once
        show_log=False
    )
    
def reading_order(boxes, y_tol_ratio=0.6):
    """Sort quadrilateral boxes line-by-line (top→bottom → left→right)."""
    if not boxes:
        return []
    centres = [np.mean(b, axis=0) for b in boxes]
    heights = [abs(b[0][1] - b[2][1]) for b in boxes]
    median_h = np.median(heights)

    groups, used = [], [False]*len(boxes)
    for i, c_i in enumerate(centres):
        if used[i]:
            continue
        row = [i]; used[i] = True
        for j, c_j in enumerate(centres):
            if not used[j] and abs(c_j[1]-c_i[1]) < median_h*y_tol_ratio:
                row.append(j); used[j] = True
        groups.append(sorted(row, key=lambda k: centres[k][0]))
    groups.sort(key=lambda g: centres[g[0]][1])
    return [boxes[k] for g in groups for k in g]


class Commands:
    def __init__(self, ocr, conf_threshold, min_length):
        self.ocr = ocr
        self.conf_threshold = conf_threshold
        self.min_length = min_length
        
    @staticmethod
    def _prep(img: np.ndarray) -> np.ndarray:
        """CLAHE contrast + bilateral blur -> cleaner crops for OCR."""
        img = cv.resize(img, None, fx=2.0, fy=2.0, interpolation=cv.INTER_CUBIC)
        lab   = cv.cvtColor(img, cv.COLOR_BGR2LAB)
        l, a, b = cv.split(lab)
        l      = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
        lab    = cv.merge((l, a, b))
        sharp  = cv.cvtColor(lab, cv.COLOR_LAB2BGR)
        return cv.bilateralFilter(sharp, d=5, sigmaColor=75, sigmaSpace=75)

    def read_text(self, image_path):
        """
        Run OCR on `image_path`, keep high-confidence words, impose a strict
        top-to-bottom / left-to-right reading order, and return the list of
        recognised text strings exactly as read.
        """
        img = cv.imread(image_path)
        if img is None:
            return []
        img = self._prep(img)

        # ── 1. OCR inference ──────────────────────────────────────────────
        result = self.ocr.ocr(img, cls=True)
        if not result or not result[0]:
            return []

        # ── 2. keep only confident, non-empty strings ────────────────────
        kept = [
            (bbox, txt.strip())
            for bbox, (txt, conf) in result[0]
            if conf >= self.conf_threshold and len(txt) >= self.min_length
        ]
        if not kept:
            return []

        # ── 3. enforce reading order (row-wise) ──────────────────────────
        ordered = reading_order([b for b, _ in kept], y_tol_ratio=0.4)      # helper above
        idx = {id(b): i for i, b in enumerate(ordered)}
        kept.sort(key=lambda x: idx[id(x[0])])

        # ── 4. return plain strings in order ─────────────────────────────
        return [txt for _, txt in kept]
