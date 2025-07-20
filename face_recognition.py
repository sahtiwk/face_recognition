#face_recognition.py
import cv2 as cv
import numpy as np

#keep the path of the image you wanna train below.
IMAGE_TO_TEST = 'test_image.jpg'

haar_cascade = cv.CascadeClassifier('haar_face.xml')
people = np.load('people.npy', allow_pickle=True)
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

img = cv.imread(IMAGE_TO_TEST)

if img is None:
    print(f"Error: Could not find '{IMAGE_TO_TEST}'.")
    print("Please make sure the image is in the same folder as the script.")
else:
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    for (x, y, w, h) in faces_rect:
        faces_region = gray[y:y+h, x:x+w]
        label_index, confidence = face_recognizer.predict(faces_region)

        print(f'Detected: {people[label_index]} with a confidence of {confidence:.2f}')
        cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv.putText(img, str(people[label_index]), (x, y-10), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), thickness=2)

    cv.imshow('Detected Face', img)
    cv.waitKey(0)
