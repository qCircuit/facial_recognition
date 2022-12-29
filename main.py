import cv2
import os
import shutil

# get cv2 cascade
face_сascade = cv2.CascadeClassifier(
    os.path.join(
        cv2.data.haarcascades,
        "haarcascade_frontalface_default.xml"
    )
)

# set video source to the default webcam
video_capture = cv2.VideoCapture(0)

''' frame by frame processing '''

while True:
    ret, frame = video_capture.read()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_сascade.detectMultiScale(
        img,
        scaleFactor=1.02, # compensates objects' diff distance
        minNeighbors=5, # objects detected near moving window
        minSize=(30, 30), # moving window size
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    # get (x, y, width, height) of the rectangle
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
    
    # display
    cv2.imshow("video", frame)

    # exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# clean up
video_capture.release()
cv2.destroyAllWindows()
