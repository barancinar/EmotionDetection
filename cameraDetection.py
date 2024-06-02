import sys
import cv2
import numpy as np
from PyQt5 import uic, QtGui, QtCore
from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import pyqtSlot, QTimer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array


class CameraDetection(QWidget):

    def __init__(self):
        super(CameraDetection, self).__init__()
        print("hello 1")

        self.Window = uic.loadUi('ui/CameraDetection.ui', self)
        self.setWindowTitle('Kamerada Tespit')

        self.setFixedSize(self.size())
        print("hello 2")

        self.logic = 0
        self.value = 1
        self.cap = None
        print("hello 3")
        self.btnShow.clicked.connect(self.captureClicked)
        print("hello 4")
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        print("hello 5")
        # Load face detection model and emotion detection model
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        print("hello 6")
        self.emotion_model = load_model('model.h5')
        print("hello 7")
        self.emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

    @pyqtSlot()
    def captureClicked(self):
        if self.logic == 0:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("Error: Could not open camera")
                return
            self.logic = 1
            self.timer.start(30)
            self.btnShow.setText("Stop Camera")
        else:
            self.timer.stop()
            if self.cap:
                self.cap.release()
                self.cap = None
            self.logic = 0
            self.btnShow.setText("Start Camera")
            self.imgLabel.clear()

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                roi_gray = gray[y:y + h, x:x + w]
                roi_gray = cv2.resize(roi_gray, (48, 48))
                roi_gray = roi_gray.astype('float') / 255.0
                roi_gray = img_to_array(roi_gray)
                roi_gray = np.expand_dims(roi_gray, axis=0)

                preds = self.emotion_model.predict(roi_gray)[0]
                emotion_probability = np.max(preds)
                label = self.emotions[preds.argmax()]

                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            self.displayImage(frame, 1)
        else:
            print("return not found")

    def displayImage(self, img, window=1):
        qformat = QImage.Format_Indexed8

        if len(img.shape) == 3:  # rows[0], cols[1], channels[2]
            if img.shape[2] == 4:
                qformat = QImage.Format_RGB888
            else:
                qformat = QImage.Format_RGB888

        img = QImage(img, img.shape[1], img.shape[0], qformat)
        img = img.rgbSwapped()
        self.imgLabel.setPixmap(QPixmap.fromImage(img))
        self.imgLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

    def closeEvent(self, event):
        self.timer.stop()
        if self.cap:
            self.cap.release()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CameraDetection()
    window.show()
    sys.exit(app.exec_())