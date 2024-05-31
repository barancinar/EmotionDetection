import sys
import cv2
from PyQt5 import uic, QtGui, QtCore
from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import pyqtSlot, QTimer


class CameraDetection(QWidget):

    def __init__(self):
        super(CameraDetection, self).__init__()

        self.Window = uic.loadUi('ui/CameraDetection.ui', self)
        self.setWindowTitle('Kamerada Tespit')

        self.setFixedSize(self.size())

        self.logic = 0
        self.value = 1
        self.cap = None

        self.btnShow.clicked.connect(self.captureClicked)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

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
