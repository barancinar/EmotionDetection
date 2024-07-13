import sys
import cv2
import numpy as np
from PyQt5 import uic, QtWidgets, QtGui, QtCore
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import pyqtSlot, Qt, QThread, pyqtSignal
from tensorflow.keras.models import load_model


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, filename, face_cascade, emotion_model, emotions):
        super().__init__()
        self.filename = filename
        self.face_cascade = face_cascade
        self.emotion_model = emotion_model
        self.emotions = emotions
        self._run_flag = True

    def run(self):
        cap = cv2.VideoCapture(self.filename)

        while self._run_flag:
            ret, frame = cap.read()

            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                roi_gray = gray[y:y + h, x:x + w]
                roi_gray = cv2.resize(roi_gray, (48, 48))
                roi_gray = roi_gray.astype('float32') / 255.0
                roi_gray = np.expand_dims(roi_gray, axis=0)
                roi_gray = np.expand_dims(roi_gray, axis=-1)

                preds = self.emotion_model.predict(roi_gray)[0]
                preds = np.exp(preds) / np.sum(np.exp(preds))

                emotion_probability = np.max(preds)
                label_index = np.argmax(preds)
                label = self.emotions[label_index]

                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2.8, (255, 0, 0), 4)

            self.change_pixmap_signal.emit(frame)

            if cv2.waitKey(1) == ord('q'):
                break

        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()


class VideoDetection(QtWidgets.QWidget):
    btnKapatClicked = pyqtSignal()

    def __init__(self):
        super().__init__()

        self.Window = uic.loadUi('ui/VideoDetection.ui', self)

        self.setWindowTitle("Video ile Duygu Tespiti")
        self.setFixedSize(self.size())

        self.btnShow.clicked.connect(self.start_video)
        self.btnKapat.clicked.connect(self.kapatClicked)

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.emotion_model = load_model('model/emotion.h5')
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

        self.thread = None

    @pyqtSlot()
    def start_video(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Video Seç", "", "Video dosyaları (*.mp4 *.avi)")

        if filename:
            if self.thread is not None:
                self.thread.stop()

            self.thread = VideoThread(filename, self.face_cascade, self.emotion_model, self.emotions)
            self.thread.change_pixmap_signal.connect(self.update_image)
            self.thread.start()

    @pyqtSlot()
    def kapatClicked(self):
        self.close()
        self.btnKapatClicked.emit()

    def closeEvent(self, event):
        if self.thread is not None:
            self.thread.stop()
            event.accept()

    def update_image(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.imgLabel.setPixmap(qt_img.scaled(self.imgLabel.size(), Qt.KeepAspectRatio))

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(q_img)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = VideoDetection()
    window.show()
    sys.exit(app.exec_())
