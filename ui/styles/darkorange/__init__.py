from ui.styles.darkorange import darkorange
from PyQt5 import QtCore

def getStyleSheet():
    fileLoc = './ui/styles/darkorange/darkorange.qss'
    print(fileLoc)
    return open(fileLoc).read()