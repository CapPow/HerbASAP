# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui/imageDialogUI.ui'
#
# Created by: PyQt5 UI code generator 5.12.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(400, 227)
        self.gridLayout = QtWidgets.QGridLayout(Dialog)
        self.gridLayout.setObjectName("gridLayout")
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.No|QtWidgets.QDialogButtonBox.Yes)
        self.buttonBox.setObjectName("buttonBox")
        self.gridLayout.addWidget(self.buttonBox, 3, 0, 1, 1)
        self.label_Image = QtWidgets.QLabel(Dialog)
        self.label_Image.setText("")
        self.label_Image.setAlignment(QtCore.Qt.AlignCenter)
        self.label_Image.setObjectName("label_Image")
        self.gridLayout.addWidget(self.label_Image, 1, 0, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem, 1, 1, 1, 1)
        self.label_dialog = QtWidgets.QLabel(Dialog)
        self.label_dialog.setAlignment(QtCore.Qt.AlignCenter)
        self.label_dialog.setObjectName("label_dialog")
        self.gridLayout.addWidget(self.label_dialog, 2, 0, 1, 2)

        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(Dialog.accept)
        self.buttonBox.rejected.connect(Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Verify color reference"))
        self.label_dialog.setText(_translate("Dialog", "Is the most white portion of the color reference visible?"))


