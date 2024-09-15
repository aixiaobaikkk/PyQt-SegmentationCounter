# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Image1.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        # Dialog.setStyleSheet("background-color: #f0f0f0;")

        Dialog.resize(840, 535)
        self.groupBox = QtWidgets.QGroupBox(Dialog)
        self.groupBox.setGeometry(QtCore.QRect(160, 30, 311, 341))
        self.groupBox.setObjectName("groupBox")
        self.label = QtWidgets.QLabel(self.groupBox)
        self.label.setGeometry(QtCore.QRect(10, 20, 291, 311))
        self.label.setText("")
        self.label.setObjectName("label")
        self.label.setStyleSheet("background-color: white;")
        self.groupBox_3 = QtWidgets.QGroupBox(Dialog)
        self.groupBox_3.setGeometry(QtCore.QRect(480, 30, 311, 341))
        self.groupBox_3.setObjectName("groupBox_3")
        self.label_4 = QtWidgets.QLabel(self.groupBox_3)
        self.label_4.setGeometry(QtCore.QRect(10, 20, 291, 311))
        self.label_4.setText("")
        self.label_4.setObjectName("label_4")
        self.label.setStyleSheet("background-color: white;")
        self.pushButton_5 = QtWidgets.QPushButton(Dialog)
        self.pushButton_5.setGeometry(QtCore.QRect(20, 360, 121, 41))
        self.pushButton_5.setObjectName("pushButton_5")
        self.pushButton = QtWidgets.QPushButton(Dialog)
        self.pushButton.setGeometry(QtCore.QRect(20, 50, 121, 41))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_3 = QtWidgets.QPushButton(Dialog)
        self.pushButton_3.setGeometry(QtCore.QRect(20, 130, 121, 41))
        self.pushButton_3.setObjectName("pushButton_3")

        self.groupBox_2 = QtWidgets.QGroupBox(Dialog)
        self.groupBox_2.setGeometry(QtCore.QRect(160, 380, 631, 121))
        self.groupBox_2.setObjectName("groupBox_2")
        self.groupBox_2.setObjectName("background-color: white;")
        self.textBrowser = QtWidgets.QTextBrowser(self.groupBox_2)
        self.textBrowser.setGeometry(QtCore.QRect(10, 20, 600, 91))
        self.textBrowser.setObjectName("textBrowser")
        self.pushButton_6 = QtWidgets.QPushButton(Dialog)
        self.pushButton_6.setGeometry(QtCore.QRect(20, 210, 121, 41))
        self.pushButton_6.setObjectName("pushButton_6")
        self.pushButton_6.setObjectName("background-color: white;")

        self.pushButton_7 = QtWidgets.QPushButton(Dialog)
        self.pushButton_7.setGeometry(QtCore.QRect(20, 290, 121, 41))
        self.pushButton_7.setObjectName("pushButton_7")
        self.pushButton_7.setObjectName("background-color: white;")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "水藻检测系统"))
        self.groupBox.setTitle(_translate("Dialog", "原图像"))
        self.groupBox_3.setTitle(_translate("Dialog", "检测结果"))
        self.pushButton_5.setText(_translate("Dialog", "清空图像"))
        self.pushButton.setText(_translate("Dialog", "载入图像"))
        self.pushButton_3.setText(_translate("Dialog", "图像检测"))
        self.groupBox_2.setTitle(_translate("Dialog", "检测信息"))
        self.pushButton_6.setText(_translate("Dialog", "查看大图"))
        self.pushButton_7.setText(_translate("Dialog", "保存"))

