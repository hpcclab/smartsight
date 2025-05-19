import cv2 as cv
import os
import time

file = "image.jpg"
simFolder = "simulate"

# MaxTime = time.time() + 120*3

imgNum = 0
choice = "n"
while choice != "1" and choice != "2":
    choice = input("1. save images\n2. run simulation\nEnter selection:")

if choice == "1":
    while True:
        if os.path.exists(file):    
            img = cv.imread(file)
            img = cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE)
            if img is not None:
                os.remove(file)
                cv.imwrite(f"{simFolder}/image{imgNum}.jpg", img)
                imgNum += 1
                # cv.imshow("image", img)
                # cv.pollKey()
                # cv.waitKey(0)
        # print("TimeRemaining: " + str(MaxTime - time.time()))

if choice == "2":
    while True:
        if os.path.exists(f"{simFolder}/image{imgNum}.jpg"):    
            img = cv.imread(f"{simFolder}/image{imgNum}.jpg")
            img = cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE)
            if img is not None:
                # os.remove(file)
                cv.imwrite(f"image.jpg", img)
                imgNum += 1
                time.sleep(1)
        else:
            imgNum = 0
