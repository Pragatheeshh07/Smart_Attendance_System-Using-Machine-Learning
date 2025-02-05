import cv2
import dlib
import numpy as np
import pandas as pd
import sqlite3
import datetime
import torch
from facenet_pytorch import MTCNN
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize MTCNN for face detection
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(keep_all=True, device=device)

# Initialize dlib face encoder
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')
face_rec_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

# Load known faces from CSV
def load_known_faces(csv_path):
    known_face_encodings = []
    known_face_names = []
    data = pd.read_csv(csv_path, header=None)
    for i in range(data.shape[0]):
        name = data.iloc[i][0]
        encoding = np.array(data.iloc[i][1:], dtype=float)
        known_face_encodings.append(encoding)
        known_face_names.append(name)
    return known_face_encodings, known_face_names

# Record attendance in SQLite database
def record_attendance(name):
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.datetime.now().strftime("%H:%M:%S")
    cursor.execute("CREATE TABLE IF NOT EXISTS attendance (name TEXT, time TEXT, date DATE, UNIQUE(name, date, subject))")
    cursor.execute("INSERT OR IGNORE INTO attendance (name, time, date) VALUES (?, ?, ?)",
                   (name, current_time, current_date))
    conn.commit()
    conn.close()
    logging.info(f"{name} marked as present at {current_time} on {current_date}")

def main():
    # Load known faces
    known_face_encodings, known_face_names = load_known_faces("data/features_all.csv")

    # Open video file
    cap = cv2.VideoCapture("video.mp4")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces in the frame
        boxes, _ = mtcnn.detect(frame)
        if boxes is not None:
            for box in boxes:
                x, y, x1, y1 = map(int, box)
                face_img = frame[y:y1, x:x1]

                # Get the landmarks/parts for the face in box
                shape = predictor(frame, dlib.rectangle(x, y, x1, y1))

                # Get the 128-dimensional face descriptor
                face_encoding = np.array(face_rec_model.compute_face_descriptor(frame, shape))

                # Compare face encodings with known faces
                face_distances = np.linalg.norm(known_face_encodings - face_encoding, axis=1)

                # Get the best match index and check the distance threshold
                best_match_index = np.argmin(face_distances)
                logging.info(f"Face distance: {face_distances[best_match_index]} for {known_face_names[best_match_index]}")

                if face_distances[best_match_index] < 0.5:  # Adjust the threshold as needed
                    name = known_face_names[best_match_index]
                    rectangle_color = (0, 255, 0)  # Green for known faces
                else:
                    name = "Unknown"
                    rectangle_color = (0, 0, 255)  # Red for unknown faces

                # Record attendance
                if name != "Unknown":
                    record_attendance(name)
                    logging.info(f"Recognized: {name}")  # Log the name

                # Draw rectangle around the face and label it
                cv2.rectangle(frame, (x, y), (x1, y1), rectangle_color, 2)
                cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, rectangle_color, 2)

        # Resize the frame to fit the window
        frame = cv2.resize(frame, (800, 600))  # Resize to a fixed size (800x600) or any size you prefer

        # Display the resulting frame in a normalized window
        cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
