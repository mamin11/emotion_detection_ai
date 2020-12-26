import cv2
from model import FacialExpressionModel
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array

face_classifier=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
model = FacialExpressionModel("emotionModel.json", "EmotionModelv3.h5")
font = cv2.FONT_HERSHEY_SIMPLEX

class VideoCam(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        _,fr = self.video.read()
        gray_frame = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray_frame, 1.3, 5)

        for (x, y, w, h) in faces:
            fc = gray_frame[y:y+h, x:x+w]

            roi = cv2.resize(fc, (48, 48))
            numpy_img = img_to_array(roi)
            image_batch = np.expand_dims(numpy_img, axis=0)
            image_batch /= 255.
            pred = model.predict_emotion(image_batch)

            cv2.putText(fr, pred, (x,y), font, 1, (255, 255, 0), 2)
            cv2.rectangle(fr, (x,y), (x+w, y+h), (0, 250, 0), 2)

            _, jpeg = cv2.imencode('.jpg', fr)
            if jpeg is None:
                return False
            else:
                return jpeg.tobytes()