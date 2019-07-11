# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui/imageDialogUI.ui'
#
# Created by: PyQt5 UI code generator 5.12.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog_image(object):
    def setupUi(self, Dialog_image):
        Dialog_image.setObjectName("Dialog_image")
        Dialog_image.setWindowModality(QtCore.Qt.ApplicationModal)
        Dialog_image.resize(400, 227)
        self.gridLayout = QtWidgets.QGridLayout(Dialog_image)
        self.gridLayout.setObjectName("gridLayout")
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog_image)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.No|QtWidgets.QDialogButtonBox.Yes)
        self.buttonBox.setObjectName("buttonBox")
        self.gridLayout.addWidget(self.buttonBox, 3, 0, 1, 1)
        self.label_Image = QtWidgets.QLabel(Dialog_image)
        self.label_Image.setText("")
        self.label_Image.setAlignment(QtCore.Qt.AlignCenter)
        self.label_Image.setObjectName("label_Image")
        self.gridLayout.addWidget(self.label_Image, 1, 0, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem, 1, 1, 1, 1)
        self.label_dialog = QtWidgets.QLabel(Dialog_image)
        self.label_dialog.setAlignment(QtCore.Qt.AlignCenter)
        self.label_dialog.setObjectName("label_dialog")
        self.gridLayout.addWidget(self.label_dialog, 2, 0, 1, 2)

        self.retranslateUi(Dialog_image)
        self.buttonBox.accepted.connect(Dialog_image.accept)
        self.buttonBox.rejected.connect(Dialog_image.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog_image)

    def retranslateUi(self, Dialog_image):
        _translate = QtCore.QCoreApplication.translate
        Dialog_image.setWindowTitle(_translate("Dialog_image", "Verify color reference"))
        self.label_dialog.setText(_translate("Dialog_image", "Is the most white portion of the color reference visible?"))


