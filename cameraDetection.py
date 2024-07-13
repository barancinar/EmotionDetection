import sys
import cv2
import numpy as np
from PyQt5 import uic, QtGui, QtCore
from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import pyqtSlot, Qt, QThread, pyqtSignal, QTimer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array


class CameraDetection(QWidget):
    btnKapatClicked = pyqtSignal()

    def __init__(self):
        super(CameraDetection, self).__init__()

        self.Window = uic.loadUi('ui/CameraDetection.ui', self)
        self.setWindowTitle('Kamerada Tespit')
        self.setFixedSize(self.size())

        self.logic = 0
        self.cap = None

        self.btnShow.clicked.connect(self.captureClicked)
        self.btnKapat.clicked.connect(self.kapatClicked)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # Yüz tanıma modeli ve duygu tanıma modelini yükle
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.emotion_model = load_model('model/emotion.h5')
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

    @pyqtSlot()
    def captureClicked(self):
        if self.logic == 0:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("Hata: Kamera açılamadı")
                return
            self.logic = 1
            self.timer.start(30)
            self.btnShow.setText("Kamerayı Durdur")
        else:
            self.timer.stop()
            if self.cap:
                self.cap.release()
                self.cap = None
            self.logic = 0
            self.btnShow.setText("Kamerayı Başlat")
            self.imgLabel.clear()

    @pyqtSlot()
    def kapatClicked(self):
        self.close()
        self.btnKapatClicked.emit()

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                roi_gray = gray[y:y + h, x:x + w]
                roi_gray = cv2.resize(roi_gray, (48, 48))
                roi_gray = roi_gray.astype('float32') / 255.0
                roi_gray = np.expand_dims(roi_gray, axis=0)
                roi_gray = np.expand_dims(roi_gray, axis=-1)  # Şekli (1, 48, 48, 1) yap

                # Tahminleri ve ön işlemden sonra veriyi yazdır
                print(f"Ön İşlem Görüntü Şekli: {roi_gray.shape}")

                preds = self.emotion_model.predict(roi_gray)[0]
                preds = np.exp(preds) / np.sum(np.exp(preds))

                # Tahmin sonuçlarını yazdır
                print(f"Tahminler: {preds}")

                emotion_probability = np.max(preds)
                label_index = np.argmax(preds)
                label = self.emotions[label_index]

                # Daha fazla bilgi ekleyelim
                print(f"Tahmin edilen duygu: {label} ({emotion_probability:.2f})")

                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            self.displayImage(frame)
        else:
            print("Hata: Çerçeve bulunamadı")

    def displayImage(self, img, window=1):
        qformat = QImage.Format_RGB888

        if len(img.shape) == 3:  # rows[0], cols[1], channels[2]
            if img.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        img = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
        img = img.rgbSwapped()
        self.imgLabel.setPixmap(QPixmap.fromImage(img))

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
