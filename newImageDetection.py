import cv2 as cv
import os
import time

file = "image.jpg"

MaxTime = time.time() + 120*3

while time.time() < MaxTime:
    if os.path.exists(file):    
        img = cv.imread(file)
        img = cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE)
        if img is not None:
            os.remove(file)
            cv.imshow("image", img)
            cv.pollKey()
            # cv.waitKey(0)
    print("TimeRemaining: " + str(MaxTime - time.time()))

