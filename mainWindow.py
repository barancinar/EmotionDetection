from PyQt5 import uic
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QMainWindow, QComboBox
from cameraDetection import CameraDetection
from imageDetection import ImageDetection
from videoDetection import VideoDetection


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        uic.loadUi('ui/MainWindow.ui', self)
        self.setWindowTitle('Duygu Tespiti')

        self.btnCamera.clicked.connect(self.cameraClicked)
        self.btnImage.clicked.connect(self.imageClicked)
        self.btnVideo.clicked.connect(self.videolicked)

        self.source = None

    @pyqtSlot()
    def cameraClicked(self):
        cameraDetection = CameraDetection()
        cameraDetection.btnKapatClicked.connect(self.showMainWindow)
        cameraDetection.show()
        self.hide()

    @pyqtSlot()
    def imageClicked(self):
        imageDetection = ImageDetection()
        imageDetection.btnKapatClicked.connect(self.showMainWindow)
        imageDetection.show()
        self.hide()

    @pyqtSlot()
    def videolicked(self):
        videoDetection = VideoDetection()
        videoDetection.btnKapatClicked.connect(self.showMainWindow)
        videoDetection.show()
        self.hide()

    @pyqtSlot()
    def showMainWindow(self):
        self.show()



