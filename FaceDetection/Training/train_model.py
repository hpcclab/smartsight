# import the necessary packages
import face_recognition
import pickle
import cv2
import os
import sys

# Project directory
project_dir = os.path.dirname(os.path.abspath(sys.argv[0]))

# Directory where images are located
dataset_dir = project_dir + "\\dataset"

# Where to put the pickled encodings
pickle_dir = project_dir + "\\encodings"

# Initialize lists to store known encodings and names
knownEncodings = []
knownNames = []
if False:
# Start processing faces
print("[INFO] start processing faces...")

# Loop over the folders in the dataset directory (each folder represents a person)
for name in os.listdir(dataset_dir):
    person_dir = os.path.join(dataset_dir, name)
    if not os.path.isdir(person_dir):
        continue

    # Loop over the images for each person
    for img_counter, img_name in enumerate(os.listdir(person_dir)):
        image_path = os.path.join(person_dir, img_name)
        print(f"[INFO] processing {image_path}...")

        # Load the input image using OpenCV
        image = cv2.imread(image_path)
        if image is None:
            continue

        # Convert the image from BGR to RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect the face locations
        boxes = face_recognition.face_locations(rgb, model="hog")

        # Compute the facial encodings
        encodings = face_recognition.face_encodings(rgb, boxes)

        # Add each encoding and name to the known lists
        for encoding in encodings:
            knownEncodings.append(encoding)
            knownNames.append(name)

# Serialize the facial encodings and names to disk
print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}
with open(pickle_dir + "/encodings.pickle", "wb") as f:
    f.write(pickle.dumps(data))