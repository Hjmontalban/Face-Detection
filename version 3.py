import pandas as pd
import cv2
import os
import time
from deepface import DeepFace

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load the pre-trained face recognition model
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Check if the training data file exists
training_data_path = 'training_data.yml'
if not os.path.exists(training_data_path):
    print(f"Error: The training data file '{training_data_path}' does not exist.")
    exit()

# Load the training data (Ensure correct format for training_data.yml)
try:
    print(f"Loading training data from: {os.path.abspath(training_data_path)}")
    recognizer.read(training_data_path)
except cv2.error as e:
    print(f"Error loading training data: {e}")
    exit()

# Load person information from CSV file
csv_file_path = "face_recognition_labels.csv"
label_names = {}
try:
    df = pd.read_csv(csv_file_path)
    for _, row in df.iterrows():
        label_id = row["Label ID"]
        label_names[label_id] = {
            "name": row["name"],
            "age": row["age"],
            "department": row["department"],
            "student_number": row["student_number"],
        }
    print(f"Data loaded from {csv_file_path}")
except FileNotFoundError:
    print(f"Error: The CSV file '{csv_file_path}' does not exist.")
    exit()

# Initialize the video capture
video_capture = cv2.VideoCapture(0)

# Set up the window
display_window_name = 'Face Recognition'
cv2.namedWindow(display_window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(display_window_name, 800, 600)

# Create a folder for screenshots if it doesn't exist
screenshot_folder = 'screenshots'
os.makedirs(screenshot_folder, exist_ok=True)

while True:
    # Read the current frame from the video capture
    ret, frame = video_capture.read()

    if not ret:
        print("Error: Unable to capture video.")
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Iterate over each detected face
    if len(faces) > 0:
        # Add delay when a face is detected
        time.sleep(0.2)  # Adjust the delay as needed

    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Recognize the face
        label, confidence = recognizer.predict(gray[y:y + h, x:x + w])

        # Prepare information to be displayed
        info_texts = []

        # Display the information of the recognized person
        if confidence < 75:  # Adjust threshold as needed
            person_info = label_names.get(label, None)
            if person_info:
                name = person_info["name"]
                age = person_info["age"]
                department = person_info["department"]
                student_number = person_info["student_number"]

                # Display information vertically
                info_texts = [
                    f"Name: {name}",
                    f"Age: {age}",
                    f"Department: {department}",
                    f"Student Number: {student_number}"
                ]

                # Take one screenshot of the recognized face and save it to the folder
                screenshot_filename = os.path.join(screenshot_folder, f"screenshot_{name.replace(' ', '_')}_{int(time.time())}.png")
                cv2.imwrite(screenshot_filename, frame)
            else:
                info_texts = ["Unknown"]
        else:
            info_texts = ["Unknown"]

        # Analyze the face using DeepFace to detect the dominant emotion
        try:
            face_roi = frame[y:y + h, x:x + w]  # Region of interest containing the face
            analysis_result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

            # If analysis_result is a list, get the first result
            if isinstance(analysis_result, list):
                analysis_result = analysis_result[0]

            # Extract the dominant emotion
            dominant_emotion = analysis_result.get('dominant_emotion', 'Unknown')
            info_texts.append(f"Emotion: {dominant_emotion}")

        except ValueError as e:
            print(f"Emotion analysis error: {e}")

        # Show the information on the video frame (positioned above the face)
        y_offset = y - 10 if y - 10 > 20 else y + h + 20
        for i, info_text in enumerate(info_texts):
            cv2.putText(frame, info_text, (x, y_offset - i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow(display_window_name, frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
video_capture.release()
cv2.destroyAllWindows()
