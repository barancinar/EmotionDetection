import sys
import cv2
import numpy as np
from PyQt5 import uic, QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QWidget, QApplication, QVBoxLayout, QPushButton, QLabel, QFileDialog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import pyqtSlot, Qt, QThread, pyqtSignal
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array


class ImageDetection(QWidget):
    btnKapatClicked = pyqtSignal()

    def __init__(self):
        super(ImageDetection, self).__init__()

        self.Window = uic.loadUi('ui/ImageDetection.ui', self)

        self.setWindowTitle("Görüntü ile Duygu Tespiti")
        self.setFixedSize(self.size())

        self.btnShow.clicked.connect(self.detectEmotion)
        self.btnKapat.clicked.connect(self.kapatClicked)

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.emotion_model = load_model('model/emotion.h5')
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

    @pyqtSlot()
    def detectEmotion(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Resim Seç", "", "Resim dosyaları (*.jpg *.png)")

        if filename:
            image = cv2.imread(filename)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

            if len(faces) == 0:
                print("Yüz bulunamadı.")
                return

            for (x, y, w, h) in faces:
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 3)

                # Yüzü kırpılmış gri tonlamalı resim olarak alın
                roi_gray = gray[y:y + h, x:x + w]
                # Yüzü yeniden boyutlandır
                roi_gray = cv2.resize(roi_gray, (48, 48))

                roi_gray = roi_gray.astype('float32') / 255.0
                roi_gray = np.expand_dims(roi_gray, axis=0)
                roi_gray = np.expand_dims(roi_gray, axis=-1)

                preds = self.emotion_model.predict(roi_gray)[0]
                preds = np.exp(preds) / np.sum(np.exp(preds))

                emotion_probability = np.max(preds)
                label_index = np.argmax(preds)
                label = self.emotions[label_index]

                # Duygu etiketini yazdırın
                cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 3)

            # Resmi QLabel içine yerleştir
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, ch = image_rgb.shape
            bytesPerLine = ch * w
            q_img = QImage(image_rgb.data, w, h, bytesPerLine, QImage.Format_RGB888)

            # QLabel boyutlarına uygun hale getir
            pixmap = QPixmap.fromImage(q_img)
            pixmap = pixmap.scaled(self.imgLabel.size(), Qt.KeepAspectRatio)
            self.imgLabel.setPixmap(pixmap)

    @pyqtSlot()
    def kapatClicked(self):
        self.close()
        self.btnKapatClicked.emit()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageDetection()
    window.show()
    sys.exit(app.exec_())
