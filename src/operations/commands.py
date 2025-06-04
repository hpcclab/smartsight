import cv2 as cv

class Commands:
    def __init__(self, ocr, conf_threshold, min_length):
        self.ocr = ocr
        self.conf_threshold = conf_threshold
        self.min_length = min_length

    def read_text(self, image_path):
        """
        Read *all* text in top-to-bottom order and return a list of strings.
        """
        img = cv.imread(image_path)
        if img is None:
            return []

        # Run OCR – result is [[ [bbox, (text, conf)], … ]]
        results = self.ocr.ocr(img, cls=True)
        if not results or not results[0]:
            return []

        # Flatten, filter, and collect (top_y, line_text)
        lines = []
        for bbox, (text, conf) in results[0]:
            text = text.strip()
            if conf < self.conf_threshold or len(text) < self.min_length:
                continue
            # bbox is a list of four (x, y) points – take the smallest y as “top”
            top_y = min(pt[1] for pt in bbox)
            lines.append((top_y, text))

        if not lines:
            return []

        # Sort so we read from top to bottom
        lines.sort(key=lambda t: t[0])
        return [t for _, t in lines]
