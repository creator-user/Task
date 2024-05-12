# src/main.py
import sys
from PyQt5 import QtWidgets
from src.ui.app import MainWindow

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
