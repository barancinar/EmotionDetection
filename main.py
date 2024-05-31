import sys
import time
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QSplashScreen
from PyQt5.QtGui import QPixmap
from mainWindow import MainWindow


def main():
    # Create app
    app = QApplication(sys.argv)

    # Splash Screen
    pix_splash = QPixmap('images/splash.png')
    splash = QSplashScreen(pix_splash, Qt.WindowStaysOnTopHint)
    splash.setMask(pix_splash.mask())
    splash.show()
    app.processEvents()

    # 3s wait
    time.sleep(3)

    # Display MainWindow
    mainWindow = MainWindow()
    mainWindow.show()

    splash.finish(mainWindow)

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
