import cv2
import os
from deepface import DeepFace
cascPath= r'C:\Users\Admin\Desktop\haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frames = video_capture.read()

    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    result = DeepFace.analyze(frames, actions=['emotion'])
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.putText(frames, result[0]['dominant_emotion'], (x, y), cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 0, 255), 3)
        cv2.rectangle(frames, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame

    cv2.imshow('Video', frames)
    print(result)
    print(result[0]['dominant_emotion'])

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()