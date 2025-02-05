import os
import mtcnn
import csv
import numpy as np
import logging
import cv2
import face_recognition

# Path of cropped faces
path_images_from_camera = "data/data_faces_from_camera/"

# Use MTCNN face detector
detector = mtcnn.MTCNN()


# Return 128D features for single image
def return_128d_features(path_img):
    try:
        img_rd = cv2.imread(path_img)
        if img_rd is None:
            logging.warning("Image not read properly: %s", path_img)
            return np.zeros(128)

        faces = detector.detect_faces(img_rd)
        logging.info("Image with faces detected: %s", path_img)

        if len(faces) != 0:
            x, y, width, height = faces[0]['box']
            face = img_rd[y:y + height, x:x + width]
            if len(face_recognition.face_encodings(face)) > 0:
                face_descriptor = face_recognition.face_encodings(face)[0]
            else:
                logging.warning("No face encodings found in image: %s", path_img)
                face_descriptor = np.zeros(128)
        else:
            logging.warning("No faces detected in image: %s", path_img)
            face_descriptor = np.zeros(128)
    except Exception as e:
        logging.error("Error in return_128d_features: %s", str(e))
        face_descriptor = np.zeros(128)
    return face_descriptor


# Return the mean value of 128D face descriptor for person X
def return_features_mean_personX(path_face_personX):
    features_list_personX = []
    try:
        photos_list = os.listdir(path_face_personX)
        if photos_list:
            for i in range(len(photos_list)):
                logging.info("Reading image: %s", path_face_personX + "/" + photos_list[i])
                features_128d = return_128d_features(path_face_personX + "/" + photos_list[i])
                if np.all(features_128d == 0):
                    continue
                else:
                    features_list_personX.append(features_128d)
        else:
            logging.warning("Warning: No images in %s/", path_face_personX)
        if features_list_personX:
            features_mean_personX = np.array(features_list_personX, dtype=object).mean(axis=0)
        else:
            features_mean_personX = np.zeros(128, dtype=object, order='C')
    except Exception as e:
        logging.error("Error in return_features_mean_personX: %s", str(e))
        features_mean_personX = np.zeros(128, dtype=object, order='C')
    return features_mean_personX


def main():
    logging.basicConfig(level=logging.INFO)
    try:
        person_list = os.listdir("data/data_faces_from_camera/")
        person_list.sort()
        with open("data/features_all.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            for person in person_list:
                logging.info("Processing person: %s", person)
                features_mean_personX = return_features_mean_personX(path_images_from_camera + person)
                if len(person.split('_', 2)) == 2:
                    person_name = person
                else:
                    person_name = person.split('_', 2)[-1]
                features_mean_personX = np.insert(features_mean_personX, 0, person_name, axis=0)
                writer.writerow(features_mean_personX)
                logging.info('\n')
            logging.info("Save all the features of faces registered into: data/features_all.csv")
    except Exception as e:
        logging.error("Error in main: %s", str(e))


if __name__ == '__main__':
    main()
