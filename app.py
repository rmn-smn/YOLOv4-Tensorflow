'''
Main Application running a QApplication using PyQT5 with QML
'''

import os
import sys

import PyQt5.QtQml
import PyQt5.QtCore
import PyQt5.QtWidgets

from bridge import YoloBridge

if __name__ == '__main__':
    # set style environment variables
    os.environ['QT_QUICK_CONTROLS_STYLE'] = 'Material'
    os.environ['QT_QUICK_CONTROLS_MATERIAL_THEME'] = 'Dark'
    os.environ['QT_QUICK_CONTROLS_MATERIAL_ACCENT'] = 'Blue'

    # set up application window and engine
    app = PyQt5.QtWidgets.QApplication(sys.argv)
    engine = PyQt5.QtQml.QQmlApplicationEngine()

    # set up bridge handling communication between frontend and backend
    bridges = {
        'yoloBridge': YoloBridge(),
    }
    
    # expose bridge to qml code
    for name in bridges:
        engine.rootContext().setContextProperty(name, bridges[name])

    # load qml file
    engine.load('frontend/main.qml')
    if not engine.rootObjects():
        sys.exit(-1)
    sys.exit(app.exec())
