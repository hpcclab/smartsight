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
    encodings_path += "encodings.pickle"
    # Use this xml file for face detection with Haar cascades
    #https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
    face_cascade_path = "C:\\Users\\jacob\\Documents\\UNT\\SmartSight\\SmartSightProject\\FaceDetection\\FacialRecognitionTest\\haarcascade_frontalface_default.xml"

    # Load the known faces and embeddings along with OpenCV's Haar
    # cascade for face detection
    print("[INFO] loading encodings + face detector...")
    load_time_start =time.time()
    data = pickle.loads(open(encodings_path, "rb").read())
    detector = cv2.CascadeClassifier()
    detector.load(face_cascade_path)
    load_time_end = time.time()
    load_time = load_time_end - load_time_start
    image = cv2.imread(cv2.samples.findFile(image_path))
    if image is None:
        print('Could not open or find the image ', image_path)
        exit(0)

    # Convert the image to grayscale for face detection and RGB for face recognition
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    det_process_start_time = time.time()
    # Detect faces in the grayscale image
    rects = detector.detectMultiScale(gray, scaleFactor=1.3, 
                                      minNeighbors=6, minSize=(40, 40),
                                      flags=cv2.CASCADE_SCALE_IMAGE)

    # Convert bounding boxes from (x, y, w, h) to (top, right, bottom, left)
    boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]
    det_process_end_time = time.time()
    det_process_time = det_process_end_time - det_process_start_time


    rec_process_start_time = time.time()
    # Compute the facial embeddings for each face bounding box
    encodings = face_recognition.face_encodings(rgb, boxes)
    names = []
    
    tolerance = 0.5
    confidences = []
    # Loop over the facial embeddings
    
    for encoding in encodings:
        # Attempt to match each face in the input image to our known encodings
        distances = face_recognition.face_distance(data["encodings"], encoding)
        
        matches = []
        for i in distances:
            m = True
            if i > tolerance:
                m = False
            matches.append(m)
            # print(i, m)
            if m:
                confidences.append(1-i)
        
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
        
    # Record end time and calculate time for the recognition process.
    rec_process_end_time = time.time()
    rec_process_time = rec_process_end_time - rec_process_start_time

    # Loop over the recognized faces and draw rectangles around them
    for ((top, right, bottom, left), name) in zip(boxes, names):
        # Draw the predicted face's name on the image
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 0, 0), 2)

    # Save the image to a file
    cv2.imwrite(result_image_path, image)
    totcon = 0
    avgCon = 0
    if len(confidences) > 0:
        # Calculate average confidence
        for con in confidences:
            totcon = totcon + con
        avgCon = totcon / len(confidences)
    # print(avgCon)
    # save detection information
    faceRec_data = []
    faceRec_data.append({
        'task_type': 'facialRecognition',
        'data_size': str(image.shape[0]) + 'x' + str(image.shape[1]),
        'file': image_path,
        'average_confidence(%)': "{:.5f}".format(avgCon*100),
        'recognition_models_load_time (s)': "{:.5f}".format(load_time),
        'detection_execution_time (s)': "{:.5f}".format(det_process_time),
        'recognition_execution_time (s)': "{:.5f}".format(rec_process_time)

    })
    if not os.path.isfile(output_csv):
        df = pd.DataFrame(faceRec_data)
        df.to_csv(output_csv, index=False, mode='a', header=True)
    else:
        df = pd.DataFrame(faceRec_data)
        df.to_csv(output_csv, index=False, mode='a', header=False)
    print(f"Facial Recognition data and stats saved to {output_csv}")

def folderScan(folder, output_csv_folder, encodings_path):
    if not os.path.exists(output_csv_folder):
        os.makedirs(output_csv_folder)
    # TODO Give file name for csv
    output_csv = os.path.join(output_csv_folder, "results.csv")
    if os.path.isfile(output_csv):
        os.remove(output_csv)
    for file in os.listdir(folder):
        filepath = os.path.join(folder, file)
        # output_csv = os.path.join(output_csv_folder, f"{os.path.splitext(file)[0]}_results.csv")
        IDFace(filepath, output_csv, encodings_path, output_csv_folder+"\\"+file+"Result.png")
        
parser = argparse.ArgumentParser(description='read image data')
parser.add_argument('--images', help='Path to input image folder (images to be recognized).', default='C:\\Users\\jacob\\Documents\\UNT\\SmartSight\\SmartSightProject\\FaceDetection\\FacialRecognitionTest\\testImages')
parser.add_argument('--csvout', help='Path to csv file for results.', default='C:\\Users\\jacob\\Documents\\UNT\\SmartSight\\SmartSightProject\\FaceDetection\\FacialRecognitionTest\\results')
parser.add_argument('--encodings', help='Path to encodings file for processing faces.', default='C:\\Users\\jacob\\Documents\\UNT\\SmartSight\\SmartSightProject\\FaceDetection\\FacialRecognitionTest\\encodings\\')
args = parser.parse_args()
folderScan(args.images, args.csvout, args.encodings)