import os
import cv2 as cv
import time
from pathlib import Path

# compute absolute path to this script's folder
script_dir = os.path.dirname(os.path.abspath(__file__))

# point simFolder at that folder (no more "simulate/simulate" errors)
simFolder = script_dir

# compute and ensure temp directory exists
tempFolder = os.path.abspath(os.path.join(script_dir, '..', 'temp'))
os.makedirs(tempFolder, exist_ok=True)

file = "image.jpg"
imgNum = 0
choice = "n"
while choice not in ("1","2"):
    choice = input("1. save images\n2. run simulation\nEnter selection:")

if choice == "1":
    while True:
        input_path = os.path.join(simFolder, file)
        if os.path.exists(input_path):
            img = cv.imread(input_path)
            img = cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE)
            if img is not None:
                os.remove(input_path)
                cv.imwrite(os.path.join(simFolder, 'images', f"image{imgNum}.jpg"), img)
                imgNum += 1

if choice == "2":
    while True:
        # build full path to the next test image
        image_path = os.path.join(simFolder, 'images', f"image{imgNum}.jpg")
        if os.path.exists(image_path):
            img = cv.imread(image_path)
            #img = cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE)
            if img is not None:
                # write the rotated frame into our temp folder as image.jpg
                cv.imwrite(os.path.join(tempFolder, file), img)
                print(f"Saved image {imgNum}")
                imgNum += 1
                time.sleep(1)
            else:
                print("Reached End")
                imgNum = 0
        else:
            print("No more images to process, resetting counter")
            imgNum = 0 
