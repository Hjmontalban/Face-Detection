import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import time  # Added for delay
from scipy.ndimage import rotate  # Added for face rotation
from sklearn.model_selection import train_test_split  # Added for dataset splitting
from sklearn.preprocessing import StandardScaler  # Added for feature scaling
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc  # Added for evaluation metrics
from sklearn.preprocessing import label_binarize  # Added for multi-class ROC curve plotting

# Initialize the LBPH face recognizer and face detector
recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Create folders for training dataset
dataset_path = 'C:/Users/MSI`/Desktop/Version 2/new verison of face detetion/training_images'

# Create label folders (e.g., 0, 1)
labels_to_create = ['0', '1']
for label in labels_to_create:
    label_folder_path = os.path.join(dataset_path, label)
    os.makedirs(label_folder_path, exist_ok=True)

print(f"Created folders: {', '.join(labels_to_create)}")

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

        # Delay for better visualization of the processing flow
        time.sleep(0.5)  # Added delay of 0.5 seconds

        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Unable to read image: {image_path}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to the image for noise reduction
        gray_filtered = cv2.GaussianBlur(gray, (5, 5), 0)

        # Detect face in the image
        faces_detected = face_cascade.detectMultiScale(gray_filtered, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        print(f"Number of faces detected in {image_name}: {len(faces_detected)}")

        for (x, y, w, h) in faces_detected:
            face_roi = gray_filtered[y:y + h, x:x + w]

            # Resize the face to a fixed size (e.g., 100x100)
            resized_face = cv2.resize(face_roi, (100, 100))

            # Rotate the face for augmentation
            rotated_face = rotate(resized_face, angle=15, reshape=False)
            faces.append(rotated_face)
            labels.append(label)

            # Add original face as well
            faces.append(resized_face)
            labels.append(label)

# Convert face images to numpy arrays and flatten them
faces_flattened = [face.flatten() for face in faces]

# Standardize the dataset for better performance
scaler = StandardScaler()
faces_scaled = scaler.fit_transform(faces_flattened)

# Split the dataset into training and testing sets
faces_train, faces_test, labels_train, labels_test = train_test_split(faces_scaled, labels, test_size=0.2, random_state=42)

# Train the recognizer on the training faces and labels
if len(faces_train) > 0:
    recognizer.train([np.reshape(face, (100, 100)) for face in faces_train], np.array(labels_train))

    # Save the trained recognizer to a file
    recognizer.write('training_data4.yml')
    print("Training data saved successfully.")

    # Accuracy calculation on the test set
    predicted_labels = []
    for face in faces_test:
        face_reshaped = np.reshape(face, (100, 100))
        predicted_label, confidence = recognizer.predict(face_reshaped)
        predicted_labels.append(predicted_label)

    # Calculate evaluation metrics
    accuracy = accuracy_score(labels_test, predicted_labels) * 100
    precision = precision_score(labels_test, predicted_labels, average='macro') * 100
    recall = recall_score(labels_test, predicted_labels, average='macro') * 100
    f1 = f1_score(labels_test, predicted_labels, average='macro') * 100

    print(f"Recognition Accuracy on Test Set: {accuracy:.2f}%")
    print(f"Precision: {precision:.2f}%")
    print(f"Recall: {recall:.2f}%")
    print(f"F1 Score: {f1:.2f}%")

    # Plotting metrics as subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Face Recognition Metrics on Test Set', fontsize=16)

    # Accuracy plot
    axes[0, 0].plot(range(len(predicted_labels)), [accuracy] * len(predicted_labels), color='b')
    axes[0, 0].set_title('Accuracy')
    axes[0, 0].set_ylim(0, 100)
    axes[0, 0].set_ylabel('Percentage')
    axes[0, 0].grid(True)

    # Precision plot
    axes[0, 1].plot(range(len(predicted_labels)), [precision] * len(predicted_labels), color='g')
    axes[0, 1].set_title('Precision')
    axes[0, 1].set_ylim(0, 100)
    axes[0, 1].set_ylabel('Percentage')
    axes[0, 1].grid(True)

    # Recall plot
    axes[0, 2].plot(range(len(predicted_labels)), [recall] * len(predicted_labels), color='r')
    axes[0, 2].set_title('Recall')
    axes[0, 2].set_ylim(0, 100)
    axes[0, 2].set_ylabel('Percentage')
    axes[0, 2].grid(True)

    # F1 Score plot
    axes[1, 0].plot(range(len(predicted_labels)), [f1] * len(predicted_labels), color='c')
    axes[1, 0].set_title('F1 Score')
    axes[1, 0].set_ylim(0, 100)
    axes[1, 0].set_ylabel('Percentage')
    axes[1, 0].grid(True)

    # Multi-class ROC Curve plot
    labels_test_binarized = label_binarize(labels_test, classes=np.unique(labels))
    n_classes = labels_test_binarized.shape[1]

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(labels_test_binarized[:, i], np.array(predicted_labels) == i)
        roc_auc = auc(fpr, tpr)
        axes[1, 1].plot(fpr, tpr, lw=2, label=f'Class {i} (AUC = {roc_auc:.2f})')

    axes[1, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[1, 1].set_title('ROC Curves for All Classes')
    axes[1, 1].set_xlabel('False Positive Rate')
    axes[1, 1].set_ylabel('True Positive Rate')
    axes[1, 1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    axes[1, 1].grid(True)

    # Hide the last subplot (bottom-right corner)
    axes[1, 2].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
else:
    print("No faces were found for training.")
