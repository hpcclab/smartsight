import cv2 as cv
import face_recognition

class FacePerception:
    def __init__(self, detector, data):
        self.detector = detector
        self.data = data

    def recognize(self, img, object_counts, passive):
        if not passive:
            return object_counts
        if "person" in object_counts:
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

            # Detect faces in the grayscale image
            rects = self.detector.detectMultiScale(gray, scaleFactor=1.3, 
                                                   minNeighbors=6, minSize=(40, 40),
                                                   flags=cv.CASCADE_SCALE_IMAGE)
            boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]
            
            encodings = face_recognition.face_encodings(rgb, boxes)
            names = []
            tolerance = 0.5
            confidences = []
            currentname = "Unknown"
  
  
            for encoding in encodings:
                distances = face_recognition.face_distance(self.data["encodings"], encoding)
                
                matches = []
                for i in distances:
                    m = True
                    if i > tolerance:
                        m = False
                    matches.append(m)
                    if m:
                        confidences.append(1-i)
                
                name = "Unknown" 
                if True in matches:
                    matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                    counts = {}
                    for i in matchedIdxs:
                        name = self.data["names"][i]
                        counts[name] = counts.get(name, 0) + 1

                    name = max(counts, key=counts.get)

                    if currentname != name:
                        currentname = name
                names.append(name)
            for name in names:
                if name != "Unknown":
                    object_counts[name] = 1
        return object_counts 
