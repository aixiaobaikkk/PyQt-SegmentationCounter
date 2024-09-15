import sys
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import Image_format as Image1
import cv2
from PIL import Image
import PyQt5_stylesheets
from PyQt5 import QtGui
from PyQt5.QtWidgets import *
import os
from infer_all import infer_all
def cvImgtoQtImg(cvImg,isConvertToGray=False):
    if isConvertToGray:
        QtImgBuf = cv2.cvtColor(cvImg, cv2.COLOR_BGR2GRAY)
        QtImg = QtGui.QImage(QtImgBuf.data, QtImgBuf.shape[1], QtImgBuf.shape[0], QtGui.QImage.Format_Grayscale8)
    else:
        QtImgBuf = cv2.cvtColor(cvImg, cv2.COLOR_BGR2RGBA)
        QtImg = QtGui.QImage(QtImgBuf.data, QtImgBuf.shape[1], QtImgBuf.shape[0], QtGui.QImage.Format_RGBA8888 )
    return QtImg

class MainDialog(QMainWindow):
    def __init__(self, parent=None):
        super(MainDialog, self).__init__(parent)
        self.ui = Image1.Ui_Dialog()
        self.ui.setupUi(self)

        self.setWindowTitle('浮游生物检测系统')
        self.b = ""
        self.result = np.ndarray(())
        self.src = np.ndarray(())
        self.Detection_result = np.ndarray(())
        self.point =""
        self.data = ""
        self.ui.pushButton.clicked.connect(self.select_button_clicked)
        self.ui.pushButton_3.clicked.connect(self.img_segmentation)
        self.ui.pushButton_5.clicked.connect(self.close_img)
        self.ui.pushButton_6.clicked.connect(self.showlarge)
        self.ui.pushButton_7.clicked.connect(self.save_img)
        self.set_background_image()

    def showlarge(self):
        if self.Detection_result.size > 1:
            self.img_show = self.Detection_result
            cv2.namedWindow("结果查看", cv2.WINDOW_NORMAL)
            cv2.imshow('结果查看', self.img_show)
            cv2.waitKey(0)
        else:
            msg_box = QMessageBox(QMessageBox.Warning, '提示', '图像为空，请先选择图像 ')
            msg_box.exec_()

    def set_background_image(self):
        self.frame = QFrame(self)
        self.frame.resize(830, 517)
        self.frame.move(0, 0)
        self.frame.lower()
        self.frame.setStyleSheet(
            'background-image: url("./PyQt5_stylesheets/back.png"); background-repeat: no-repeat;')

    # 保存图像
    def save_img(self):
      if self.Detection_result.size > 1:
           fileName, tmp = QFileDialog.getSaveFileName(self, '保存图像', ' ', '*.png *.jpg *.bmp')
           cv2.imwrite(fileName, self.Detection_result)
      else:
        msg_box = QMessageBox(QMessageBox.Warning, '提示', '图像为空，无法保存  ')
        msg_box.exec_()

    # 选择文件清空图像
    def close_img(self):
        self.ui.label.setPixmap(QtGui.QPixmap(""))
        self.ui.label_4.setPixmap(QtGui.QPixmap(""))

    # 选择图像
    def select_button_clicked(self):
        self.fileName, tmp = QFileDialog.getOpenFileName(self, '打开图像', 'Image', '*.png *.jpg *.bmp *.jpeg')
        print(self.fileName)
        if self.fileName == '':
            return
        self.src = cv2.imread(self.fileName)
        QtImg = cvImgtoQtImg(self.src)
        jpg_out_gray = QtGui.QPixmap(QtImg).scaled(self.ui.label.width(), self.ui.label.height())  # 设置图片大小
        self.ui.label.setPixmap(jpg_out_gray)  # 设置图片显示

    def img_segmentation(self):
       if self.src.size > 1:

            infer_img, class_count = infer_all(path=self.fileName)

            self.Detection_result = cv2.imread(infer_img)
            QtImg1 = cvImgtoQtImg(self.Detection_result)
            jpg_out_gray1 = QtGui.QPixmap(QtImg1).scaled(self.ui.label_4.width(), self.ui.label_4.height())  # 设置图片大小
            self.ui.label_4.setPixmap(jpg_out_gray1)  # 设置图片显示
            # 显示检测结果信息
            self.ui.textBrowser.setText(f"检测到 {len(class_count)} 种目标")
            print(class_count)
            for class_name, count in class_count.items():
                self.ui.textBrowser.append(f"类别 {class_name} 的数量为 {count}")

            # for class_label, count in class_count.items():
            #     self.ui.textBrowser.setText(f"类别 {class_label} 的目标数目为: {count}")
            #

       else:
        msg_box = QMessageBox(QMessageBox.Warning, '提示', '图像为空，无法保存  ')
        msg_box.exec_()





if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(PyQt5_stylesheets.load_stylesheet_pyqt5(style="style_blue"))
    ui = MainDialog()
    ui.show()
    sys.exit(app.exec_())




