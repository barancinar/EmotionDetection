
from PyQt5 import uic
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QMainWindow, QComboBox
from cameraDetection import CameraDetection


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        uic.loadUi('ui/MainWindow.ui', self)
        self.setWindowTitle('Duygu Tespiti')

        self.source = None

    @pyqtSlot()
    def on_pushButton_clicked(self):
        cameraDetection = CameraDetection()

        cameraDetection.show()
        self.hide()



