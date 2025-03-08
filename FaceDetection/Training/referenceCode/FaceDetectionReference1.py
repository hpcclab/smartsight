# import the necessary packages
import face_recognition
import pickle
import cv2
import time
import argparse
import os
import pandas as pd

def IDFace(image_path, output_csv, encodings_path, result_image_path):
    # Initialize 'currentname' to trigger only when a new person is identified
    currentname = "unknown"
    # Determine faces from encodings.pickle file model created from train_model.py
    encodings_path = "encodings.pickle"
    # Use this xml file for face detection with Haar cascades
    #https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
    face_cascade_path = "/home/caj0249/Documents/FacialRecognitionTest/haarcascade_frontalface_default.xml"

    # Load the known faces and embeddings along with OpenCV's Haar
    # cascade for face detection
    print("[INFO] loading encodings + face detector...")
    data = pickle.loads(open(encodings_path, "rb").read())
    detector = cv2.CascadeClassifier()
    detector.load(face_cascade_path)

    image = cv2.imread(cv2.samples.findFile(image_path))
    if image is None:
        print('Could not open or find the image ', image_path)
        exit(0)

    # Convert the image to grayscale for face detection and RGB for face recognition
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Detect faces in the grayscale image
    rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
                                      minNeighbors=5, minSize=(30, 30),
                                      flags=cv2.CASCADE_SCALE_IMAGE)

    # Convert bounding boxes from (x, y, w, h) to (top, right, bottom, left)
    boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]
    
    # Compute the facial embeddings for each face bounding box
    encodings = face_recognition.face_encodings(rgb, boxes)
    names = []

    # Loop over the facial embeddings
    for encoding in encodings:
        # Attempt to match each face in the input image to our known encodings
        matches = face_recognition.compare_faces(data["encodings"], encoding, tolerance=0.5)
        name = "Unknown"  # Default to "Unknown" if no match is found

        # Check if a match was found
        if True in matches:
            # Get the indices of all matched faces and count each match
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            
        # Count the number of times each face was matched
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            # Determine the recognized face with the most matches
            name = max(counts, key=counts.get)

            # If a new person is identified, print their name
            if currentname != name:
                currentname = name
                #print(currentname)

        # Add the name to the list of recognized names
        names.append(name)
        
        # Loop over the recognized faces and draw rectangles around them
    for ((top, right, bottom, left), name) in zip(boxes, names):
        # Draw the predicted face's name on the image
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 0, 0), 2)

    # Save the image to a file
    cv2.imwrite(result_image_path, image)

def folderScan(folder, output_csv_folder, encodings_path):
    if not os.path.exists(output_csv_folder):
        os.makedirs(output_csv_folder)
    # TODO Give file name for csv
    output_csv = output_csv_folder # os.path.join(output_csv_folder, "results.csv")
    if os.path.isfile(output_csv):
        os.remove(output_csv)
    for file in os.listdir(folder):
        filepath = os.path.join(folder, file)
        # output_csv = os.path.join(output_csv_folder, f"{os.path.splitext(file)[0]}_results.csv")
        IDFace(filepath, output_csv, encodings_path, output_csv_folder+"/"+file+"Result.png")
        
parser = argparse.ArgumentParser(description='read image data')
parser.add_argument('--images', help='Path to input image folder (images to be recognized).', default='/home/caj0249/Documents/FacialRecognitionTest/testImages')
parser.add_argument('--csvout', help='Path to csv file for results.', default='/home/caj0249/Documents/FacialRecognitionTest/results')
parser.add_argument('--encodings', help='Path to encodings file for processing faces.', default='/home/caj0249/Documents/FacialRecognitionTest/encodings/encodings.pickle')
args = parser.parse_args()
folderScan(args.images, args.csvout, args.encodings)