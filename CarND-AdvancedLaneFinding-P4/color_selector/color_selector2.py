import sys
from PyQt5 import uic
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5.QtCore import QDir, Qt
from PyQt5.QtGui import QImage, QPainter, QPalette, QPixmap
from PyQt5.QtWidgets import (QAction, QApplication, QFileDialog, QLabel,
        QMainWindow, QMenu, QMessageBox, QScrollArea, QSizePolicy)
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtWidgets import (QApplication, QBoxLayout, QCheckBox, QComboBox,
        QDial, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QScrollBar,
        QSlider, QSpinBox, QStackedWidget, QWidget)
from PyQt5.QtGui import *
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from Sobel import *



class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set up the user interface from Designer.
        uic.loadUi("gui.ui", self)
        self.img = None
        self.hls = None

        # Connect up the buttons.
        # self.actionImage.clicked.connect(self.BtnClck)
        self.set_prop_slider(self.min_hue_slider, 0, 180)
        self.set_prop_slider(self.max_hue_slider, 0, 180)
        self.set_prop_slider(self.min_sat_slider, 0, 255)
        self.set_prop_slider(self.max_sat_slider, 0, 255)
        self.set_prop_slider(self.min_lig_slider, 0, 255)
        self.set_prop_slider(self.max_lig_slider, 0, 255)
        self.set_prop_slider(self.min_sobel_x_slider, 0, 255)
        self.set_prop_slider(self.max_sobel_x_slider, 0, 255)
        self.set_prop_slider(self.min_sobel_y_slider, 0, 255)
        self.set_prop_slider(self.max_sobel_y_slider, 0, 255)
        self.set_prop_slider(self.min_mag_slider, 0, 255)
        self.set_prop_slider(self.max_mag_slider, 0, 255)
        self.set_prop_slider(self.min_dir_slider, 0, np.pi)
        self.set_prop_slider(self.max_dir_slider, 0, np.pi)
        self.update_slider_val()
        self.slider_changed()


        self.open_image_btn.clicked.connect(self.OpenImage)
        self.actionImage.triggered.connect(self.OpenImage)
        self.show()

    def set_prop_slider(self, slider, mini, maxi):
        slider.setMinimum(mini)
        slider.setMaximum(maxi)
        slider.setTickInterval(1)
        slider.sliderReleased.connect(self.update_slider_val)
        slider.valueChanged.connect(self.slider_changed)


    def slider_changed(self):
        self.min_hue_lbl.setText("Min H: " + str(self.min_hue_slider.value()))
        self.min_sat_lbl.setText("Min S: " + str(self.min_sat_slider.value()))
        self.min_lig_lbl.setText("Min L: " + str(self.min_lig_slider.value()))
        self.min_sobel_x_lbl.setText("Min Sobel X: " + str(self.min_sobel_x_slider.value()))
        self.min_sobel_y_lbl.setText("Min Sobel Y: " + str(self.min_sobel_y_slider.value()))
        self.min_mag_lbl.setText("Min Mag: " + str(self.min_mag_slider.value()))
        self.min_dir_lbl.setText("Min Dir: " + str(self.min_dir_slider.value()))

        self.max_hue_lbl.setText("Max H: " + str(self.max_hue_slider.value()))
        self.max_sat_lbl.setText("Max S: " + str(self.max_sat_slider.value()))
        self.max_lig_lbl.setText("Max L: " + str(self.max_lig_slider.value()))
        self.max_sobel_x_lbl.setText("Min Sobel X: " + str(self.max_sobel_x_slider.value()))
        self.max_sobel_y_lbl.setText("Min Sobel Y: " + str(self.max_sobel_y_slider.value()))
        self.max_mag_lbl.setText("Min Mag: " + str(self.max_mag_slider.value()))
        self.max_dir_lbl.setText("Min Dir: " + str(self.max_dir_slider.value()))


    def update_slider_val(self):
        self.slider_changed()
        if self.hls is not None:
            masked_data, color_mask = self.color_selection()
            height, width, bytesPerComponent = masked_data.shape
            bytesPerLine = 3 * width
            self.display_image(QImage(masked_data.data, width, height, bytesPerLine,QImage.Format_RGB888), defa=1)

            sobel_x_mask = abs_sobel_thresh(self.hls[:,:,2], orient='x', sobel_kernel=3,
                thresh=(self.min_sobel_x_slider.value(), self.max_sobel_x_slider.value()))
            sobel_x_img = cv2.bitwise_and(self.img, self.img, mask=sobel_x_mask)
            self.display_image(QImage(sobel_x_img.data, width, height, bytesPerLine,QImage.Format_RGB888), defa=2)

            sobel_y_mask = abs_sobel_thresh(self.hls[:,:,2], orient='y', sobel_kernel=3,
                thresh=(self.min_sobel_y_slider.value(), self.max_sobel_y_slider.value()))
            sobel_y_img = cv2.bitwise_and(self.img, self.img, mask=sobel_y_mask)
            self.display_image(QImage(sobel_y_img.data, width, height, bytesPerLine,QImage.Format_RGB888), defa=3)

            mag_mask = mag_thresh(self.hls[:,:,2], sobel_kernel=3,
                mag_thresh=(self.min_mag_slider.value(), self.max_mag_slider.value()))
            mag_img = cv2.bitwise_and(self.img, self.img, mask=mag_mask)
            self.display_image(QImage(mag_img.data, width, height, bytesPerLine,QImage.Format_RGB888), defa=4)

            # dir_mask = dir_threshold(self.img, sobel_kernel=3,
            #     thresh=(np.pi/2 - np.pi/4, np.pi/2 + np.pi/4))
            y, x = mag_mask.shape
            img_size = (1280, 720)
            vertices = [np.array([ [(x//2) - 30,(y//2)+65], [(x//2) + 60,(y//2)+50],[x-100,y], [130,y] ], np.int32)]
            roi_mask = self.region_of_interest(mag_mask|color_mask|sobel_x_mask|sobel_y_mask, vertices)

            dir_img = cv2.bitwise_and(self.img, self.img, mask=roi_mask)
            src = np.array([[200, 716], [1124, 716], [700, 450], [585, 450]], np.float32)
            dst = np.array([[200, 716], [1124, 716], [1124, 0], [200, 0]], np.float32)
            M = cv2.getPerspectiveTransform(src, dst)
            reverse_trans = cv2.getPerspectiveTransform(dst, src)
            img = cv2.warpPerspective(dir_img, M, img_size, flags=cv2.INTER_LINEAR)
            self.display_image(QImage(img.data, width, height, bytesPerLine,QImage.Format_RGB888), defa=5)

            # self.display_image(masked_data)

    def region_of_interest(self, current_mask, vertices):
        """
        Applies an image mask.

        Only keeps the region of the image defined by the polygon
        formed from `vertices`. The rest of the image is set to black.
        """
        #defining a blank mask to start with
        mask = np.zeros_like(current_mask)
        ignore_mask_color = 1
        cv2.fillPoly(mask, vertices, ignore_mask_color)

        return mask & current_mask


    def color_selection(self):
        h = self.hls[:,:,0]
        l = self.hls[:,:,1]
        s = self.hls[:,:,2]
        mask = np.zeros_like(s)
        mask[(h >= self.min_hue_slider.value()) & (h <= self.max_hue_slider.value()) &
            (s >= self.min_sat_slider.value()) & (s <= self.max_sat_slider.value()) &
            (l >= self.min_lig_slider.value()) & (l <= self.max_lig_slider.value())] = 1

        masked_data = cv2.bitwise_and(self.img, self.img, mask=mask)

        return masked_data,mask


    def OpenImage(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open File",
                QDir.currentPath())
        if fileName:
            image = QImage(fileName)
            if image.isNull():
                QMessageBox.information(self, "Image Viewer",
                        "Cannot load %s." % fileName)
                return
            print(fileName)
            self.image = Image.open(fileName)
            self.img = np.array(self.image).copy()
            self.hls = cv2.cvtColor(self.img, cv2.COLOR_RGB2HLS)
            # self.image.setPixmap(QPixmap.fromImage(image))
            # img = Image.open(fileName)
            self.display_image(self.image, defa=0)
            # self.scaleFactor = 1
            #
            # self.printAct.setEnabled(True)
            # self.fitToWindowAct.setEnabled(True)
            # self.updateActions()
            #
            # if not self.fitToWindowAct.isChecked():
            #     self.imageLabel.adjustSize()

    def image_to_pixmap(self, img):
        im = img.convert("RGBA")
        data = im.tobytes("raw","RGBA")
        qim = QImage(data, im.size[0], im.size[1], QImage.Format_RGBA8888)
        pix = QPixmap.fromImage(qim)

        return pix

    def display_image(self, image, defa=0):
        if defa == 0:
            pixMap = self.image_to_pixmap(image).scaled(self.original_image.width(), 720,Qt.KeepAspectRatio)
        else:
            pixMap = QPixmap.fromImage(image).scaled(self.image_holder.width(), 720,Qt.KeepAspectRatio)


        if defa == 0:
            lable = QLabel(self.original_image)
        elif defa == 1:
            lable = QLabel(self.image_holder)
        elif defa == 2:
            lable = QLabel(self.sobel_x_holder)
        elif defa == 3:
            lable = QLabel(self.sobel_y_holder)
        elif defa == 4:
            lable = QLabel(self.mag_holder)
        elif defa == 5:
            lable = QLabel(self.dir_holder)
        lable.setPixmap(pixMap)
        lable.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MainWindow()
    sys.exit(app.exec_())
