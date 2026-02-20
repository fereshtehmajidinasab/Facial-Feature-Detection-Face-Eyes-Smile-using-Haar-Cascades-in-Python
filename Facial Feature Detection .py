import cv2 as cv
import numpy as np


face_cascade = cv.CascadeClassifier('face.xml')
eye_cascade = cv.CascadeClassifier('eye.xml')
smile_cascade = cv.CascadeClassifier('smile.xml')


cap = cv.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert to grayscale
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Declare ROI (Region of Interest)
    roi_gray = None
    roi_color = None

    # Detect faces
    faces = face_cascade.detectMultiScale(frame_gray, 1.3, 5)
    print(faces)
    # If faces are found, proceed with further detection and drawing
    if len(faces) > 0:
        # Draw rectangles around the faces
        for (x, y, w, h) in faces:
            cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)

            # Declare ROIs for face, eyes, and smiles
            roi_gray = frame_gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]

            # Detect eyes and smiles in the ROI
            eyes = eye_cascade.detectMultiScale(roi_gray)
            smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)

            # Draw rectangles around eyes and smiles
            for (ex, ey, ew, eh) in eyes:
                cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

            for (sx, sy, sw, sh) in smiles:
                cv.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)

    # Display the resulting frame
    cv.imshow('frame', frame)

    # Break if the ESC key is pressed
    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
