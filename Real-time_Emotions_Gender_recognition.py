# Real-time Gender & Emotion recognition


# import libraries
from keras.models import load_model
from time import sleep
from tensorflow.keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np

# load models
face_classifier = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
gender_model = load_model('./models/Gender_CNN_model.model')
emotion_model = load_model('./models/FER_CNN_model.h5')


# classes
emotion_labels = ['Angry', 'Disgust', 'Fear','Happy', 'Neutral', 'Sad', 'Surprise']
gender_labels = ['Female', 'Male']


# create a video object
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # convert each frame to a grag-scale image for face_classifier (CascadeClassifier)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_frame, 1.3, 5)

    for (x, y, w, h) in faces: # coordinates of x = top and y = left & length of w = width and h = hight
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) # draw a green rectangle 
        detected_face_region_gray = gray_frame[y:y+h, x:x+w]
        detected_face_region_gray = cv2.resize(detected_face_region_gray, (48, 48), interpolation=cv2.INTER_AREA) # resize the detected_face_region to 48X48 => because my emotion detection model has trained on 48x48 images

        detected_face_region = detected_face_region_gray.astype('float')/255.0  # rescale 0-1
        detected_face_region = img_to_array(detected_face_region)
        detected_face_region = np.expand_dims(detected_face_region, axis=0) # expand dimensions to (1, 48, 48, 1) (n,img_width,img_height,num_channels) => so that this array has the right dimensions for prediction

        # Emotion
        emotion_preds = emotion_model.predict(detected_face_region)[0] # the prediction is a one-hot-encoded array for 7 classes
        emotion_label = emotion_labels[emotion_preds.argmax()] # find the maximun prbability in the predicted array to find the label
        emotion_label_position = (x, y)
        cv2.putText(frame, emotion_label, emotion_label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0, 0), 2)

        # Gender
        detected_gender_region = frame[y:y+h, x:x+w]
        detected_gender_region = cv2.resize(detected_gender_region, (96, 96),interpolation=cv2.INTER_AREA) # resize the detected_gender_region to 96X96 => because my gender detection model has trained on 96X96 images
        gender_pred = gender_model.predict(np.array(detected_gender_region).reshape(-1, 96, 96, 3)) # the prediction is 0 or 1
        gender_pred = (gender_pred >= 0.5).astype(int)[:, 0]
        gender_label = gender_labels[gender_pred[0]]
        gender_label_position = (x, y+h+50)
        cv2.putText(frame, gender_label, gender_label_position,cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Emotion & Gender Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
