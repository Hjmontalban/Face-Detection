import cv2
import os
import numpy as np
from deepface import DeepFace
import dlib

# Initialize the LBPH face recognizer and face detector
recognizer = cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=8, grid_x=8, grid_y=8)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Set the dataset path
dataset_path = 'C:/Users/MSI`/Desktop/Version 2/new verison of face detetion/training_images'

faces = []
labels = []


# Data augmentation function
def augment_image(image):
    rows, cols = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), 15, 1)  # Rotate by 15 degrees
    rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))
    flipped_image = cv2.flip(image, 1)  # Horizontal flip
    return [rotated_image, flipped_image]


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

        # Apply augmentation
        augmented_images = [img] + augment_image(img)

        for aug_img in augmented_images:
            gray = cv2.cvtColor(aug_img, cv2.COLOR_BGR2GRAY)

            # Detect face using Haar Cascade
            faces_detected = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Detect face using Dlib
            dlib_faces = detector(gray)

            # Process detected faces
            for (x, y, w, h) in faces_detected:
                # Align the face using Dlib landmarks
                rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
                shape = predictor(gray, rect)
                aligned_face = dlib.get_face_chip(aug_img, shape)
                gray_aligned = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2GRAY)

                faces.append(gray_aligned)
                labels.append(label)

                # Use DeepFace to analyze the detected face (optional)
                try:
                    analysis = DeepFace.analyze(img_path=image_path, actions=['age', 'gender', 'emotion'],
                                                enforce_detection=False)
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
