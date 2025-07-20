# face_trainer.py
import cv2 as cv
import numpy as np
import os

TRAINING_DATA_DIR = 'training_data'
haar_cascade = cv.CascadeClassifier('haar_face.xml')

people = [person for person in os.listdir(TRAINING_DATA_DIR) if os.path.isdir(os.path.join(TRAINING_DATA_DIR, person))]
print("Found people:", people)

features = []
labels = []

for person in people:
    path = os.path.join(TRAINING_DATA_DIR, person)
    label = people.index(person)

    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        img_array = cv.imread(img_path)
        if img_array is None:
            continue

        gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
        faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

        for (x, y, w, h) in faces_rect:
            faces_region = gray[y:y+h, x:x+w]
            features.append(faces_region)
            labels.append(label)

print("==== Training data created ====")

features = np.array(features, dtype='object')
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.train(features, labels)

face_recognizer.save('face_trained.yml')
np.save('people.npy', people)

print("==== Training complete. Model saved as 'face_trained.yml' ====")
