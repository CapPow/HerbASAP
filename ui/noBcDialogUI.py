# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui/noBcDialogUI.ui'
#
# Created by: PyQt5 UI code generator 5.12.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog_noBc(object):
    def setupUi(self, Dialog_noBc):
        Dialog_noBc.setObjectName("Dialog_noBc")
        Dialog_noBc.setWindowModality(QtCore.Qt.ApplicationModal)
        Dialog_noBc.resize(463, 136)
        self.gridLayout = QtWidgets.QGridLayout(Dialog_noBc)
        self.gridLayout.setObjectName("gridLayout")
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog_noBc)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setCenterButtons(False)
        self.buttonBox.setObjectName("buttonBox")
        self.gridLayout.addWidget(self.buttonBox, 2, 0, 1, 1)
        self.lineEdit_userBC = QtWidgets.QLineEdit(Dialog_noBc)
        self.lineEdit_userBC.setObjectName("lineEdit_userBC")
        self.gridLayout.addWidget(self.lineEdit_userBC, 1, 0, 1, 1)
        self.label = QtWidgets.QLabel(Dialog_noBc)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)

        self.retranslateUi(Dialog_noBc)
        self.buttonBox.accepted.connect(Dialog_noBc.accept)
        self.buttonBox.rejected.connect(Dialog_noBc.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog_noBc)

    def retranslateUi(self, Dialog_noBc):
        _translate = QtCore.QCoreApplication.translate
        Dialog_noBc.setWindowTitle(_translate("Dialog_noBc", "No Barcode Found"))
        self.label.setText(_translate("Dialog_noBc", "No barcode found, please enter the catalog number."))


