import cv2
from keras.models import load_model
import numpy as np
import os
from datetime import datetime, timedelta

blue = (255, 0, 0)

# Lấy đường dẫn thư mục của file đang chạy
script_folder = os.path.dirname(os.path.abspath(__file__))

video_capture = cv2.VideoCapture(0)

# load the haar cascade face detector
face_detector = cv2.CascadeClassifier(os.path.join(script_folder, "haar_cascade/haarcascade_frontalface_default.xml"))
# load our trained model
model = load_model(os.path.join(script_folder, "model"))

# create a folder to store captured images in the same directory as the script
output_folder = os.path.join(script_folder, "captured_images")
os.makedirs(output_folder, exist_ok=True)

# Initialize time variables
start_time = datetime.now()
capture_interval = 1.5  # in seconds

# loop over the frames
while True:
    # get the next frame from the video and convert it to grayscale
    _, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # apply our face detector to the grayscale frame
    faces = face_detector.detectMultiScale(gray, 1.1, 8)
    
    # go through the face bounding boxes 
    for (x, y, w, h) in faces:
        # draw a rectangle around the face on the frame
      #  cv2.rectangle(frame, (x, y), (x + w, y + h), blue, 2)
        # extract the face from the grayscale image
        roi = gray[y:y + h, x:x + w]

        # Applying CLAHE to the face ROI
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
        roi_clahe = clahe.apply(roi)
        
        roi = cv2.resize(roi_clahe, (32, 32))
        roi = roi / 255.0
        # add a new axis to the image
        # previous shape: (32, 32), new shape: (1, 32, 32)
        roi = roi[np.newaxis, ...]
        # apply the smile detector to the face roi
        prediction = model.predict(roi)[0]
      #  label = "Smiling" if prediction >= 0.5 else "Not Smiling"

       # cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
        #            0.75, blue, 2)

        # capture and save image if smiling and interval has passed
        if prediction >= 0.5:
            if (datetime.now() - start_time).total_seconds() >= capture_interval:
                # create a unique filename using timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(output_folder, f"smile_{timestamp}.png")
                cv2.imwrite(filename, frame)
                print(f"Saved image: {filename}")
                start_time = datetime.now()

    cv2.imshow('Frame', frame)
   
    # wait for 1 millisecond and if the q key is pressed, break the loop
    if cv2.waitKey(1) == ord('q'):
        break
    
# release the video capture and close all windows
video_capture.release()
cv2.destroyAllWindows()
