# ==========================================
# STANDARD LIBRARY
# ==========================================
import sys
import ctypes
import time
import logging
import configparser

# ==========================================
# THIRD-PARTY
# ==========================================
import qdarkstyle
import vmbpy

# ==========================================
# PyQt5
# ==========================================
import PyQt5
import PyQt5.QtWidgets as qt
from PyQt5.QtGui import QIcon

# ==========================================
# LOCAL MODULES
# ==========================================
from camera import Alvium
from GUI_modules.control import Control
from GUI_modules.image_window import ImageWin

window_icon_name = 'FVEY_Rosette.ico'

# main class, parent of other classes
class CameraGUI(qt.QMainWindow):
    def __init__(self, app):
        super().__init__()
        time.sleep(0.1) #It appears this sleep is necessary to run on python 3.11.2. Not sure why
        
        self.setWindowIcon(QIcon(window_icon_name))

        self.setStyleSheet("QWidget{font: 10pt;}")
        # self.setStyleSheet("QToolTip{background-color: black; color: white; font: 10pt;}")
        self.app = app
        logging.getLogger().setLevel("INFO")

        # read default settings from a local .ini file
        self.defaults = configparser.ConfigParser()
        self.defaults.read('defaults.ini')

        self.setWindowTitle(f"Alvium - {self.defaults['camera']['id']}")

        # instantiate other classes
        self.device = Alvium(self.defaults["camera"]["id"])
        self.image_win = ImageWin(self)
        self.control = Control(self, self.image_win)

        # load latest settings
        self.control.load_settings(latest=True)

        self.splitter = qt.QSplitter()
        self.splitter.setOrientation(PyQt5.QtCore.Qt.Horizontal)
        self.setCentralWidget(self.splitter)
        self.splitter.addWidget(self.image_win)
        self.splitter.addWidget(self.control)

        self.resize(1600, 900)
        self.show()

    def closeEvent(self, event):
        self.control.stop_server_worker()
        time.sleep(0.03)  # give some time for the server thread to close

        if not self.control.active:
            self.control.save_settings(latest=True)
            super().closeEvent(event)

        else:
            # ask if continue to close
            ans = qt.QMessageBox.warning(self, 'Program warning',
                                'Warning: the program is running. Conitnue to close the program?',
                                qt.QMessageBox.Yes | qt.QMessageBox.No,
                                qt.QMessageBox.No)
            if ans == qt.QMessageBox.Yes:
                self.control.save_settings(latest=True)
                super().closeEvent(event)
            else:
                event.ignore()

    

if __name__ == '__main__':
    app = qt.QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    
    # This is for making the window's icon
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("frag.cameras.alvium")
    app.setWindowIcon(QIcon(window_icon_name))

    main_window = CameraGUI(app)

    try:
        # All interaction with the camera needs to do within the context manager of both the Vimba SDK and the camera.
        # Entering the camera context manager is slow as hell, so do it once at the top level of the program. Doing it
        # this way does mean that other programs (e.g. vimba viewer) are locked out of the camera while this program is
        # running, but that's a worthwhile tradeoff.
        with vmbpy.VmbSystem.get_instance(), main_window.device.cam:
            app.exec_()
            sys.exit(0)
    except SystemExit:
        print("\nApp is closing...")