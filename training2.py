import cv2
import os
import numpy as np
from deepface import DeepFace

# Initialize the LBPH face recognizer and face detector
recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Set the dataset path
dataset_path = 'C:/Users/MSI`/Desktop/Version 2/new verison of face detetion/training_images'

faces = []
labels = []

# Iterate through each folder inside the dataset path
for label_folder in os.listdir(dataset_path):
    label_folder_path = os.path.join(dataset_path, label_folder)

    # Ensure that the label_folder_path is a directory
    if not os.path.isdir(label_folder_path):
        continue

    # Use the folder name as the label (assuming labels are integers)
    try:
        label = int(label_folder)
    except ValueError:
        print(f"Skipping folder {label_folder}, label must be an integer.")
        continue

    # Iterate over the images in the folder
    for image_name in os.listdir(label_folder_path):
        image_path = os.path.join(label_folder_path, image_name)
        print(f"Processing image: {image_path}")

        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Unable to read image: {image_path}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect face in the image
        faces_detected = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        print(f"Number of faces detected in {image_name}: {len(faces_detected)}")

        for (x, y, w, h) in faces_detected:
            face_region = gray[y:y + h, x:x + w]
            faces.append(face_region)
            labels.append(label)

            # Use DeepFace to analyze the detected face (optional)
            try:
                analysis = DeepFace.analyze(img_path=image_path, actions=['age', 'gender', 'emotion'], enforce_detection=False)
                print(f"DeepFace Analysis for {image_name}: {analysis}")
            except Exception as e:
                print(f"DeepFace analysis failed for {image_name}: {e}")

# Train the recognizer on the faces and labels
if len(faces) > 0:
    recognizer.train(faces, np.array(labels))

    # Save the trained recognizer to a file
    recognizer.write('training_data.yml')
    print("Training data saved successfully.")
else:
    print("No faces were found for training.")