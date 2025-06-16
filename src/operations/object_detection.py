import cv2 as cv
from collections import Counter

class ObjectDetection:
    def __init__(self, model, minConf):
        self.model = model
        self.minConf = minConf

    def detect(self, img):
        results = self.model(img)
        return results # Returns EVERYTHING, including locations
 