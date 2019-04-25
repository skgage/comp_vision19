import sys

from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog
from PyQt5.uic import loadUi
from PyQt5 import QtWidgets
from PyQt5.QtGui import QIcon, QPixmap
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import pickle
import shutil
import k_means_funcs as kmf
import time
import glob

def load_images_from_folder(folder):
    images = []
    for filename in folder:
        img = cv2.imread(filename)
        img = cv2.resize(img, (1000,1000))
        #print(np.shape(img))
        if img is not None:
            images.append(img)
    return images
# k = 4
class PhotoSorter(QDialog):
    def __init__(self):
        super(PhotoSorter,self).__init__()
        loadUi('photosorter.ui',self)
        self.setWindowTitle('PhotoSorter Gui')
        self.pushButton1.clicked.connect(self.loadFiles_new)
        self.pushButton2.clicked.connect(self.loadFiles_add)

        # # Code for photo viewer
        self.group_spinBox.valueChanged.connect(self.group_change)
        self.photo_spinBox.valueChanged.connect(self.display_image)
    @pyqtSlot()
    def on_pushButton_clicked(self):
        self.label1.setText('Welcome: '+self.lineEdit1.text()+ ' '+self.lineEdit2.text())
    def loadFiles_new(self):
        dir_name = "sorted_images"
        if os.path.isdir(dir_name):
            shutil.rmtree(dir_name)
        filter = "JPG (*.jpg)"
        file_name = QFileDialog()
        file_name.setFileMode(QFileDialog.ExistingFiles)
        names, _ = file_name.getOpenFileNames(self, "Open files", "C\\Desktop", filter)
        #print (names)
        # print(np.shape(load_images_from_folder(names)))

        if os.path.exists('init_images.pkl'):
            os.remove('init_images.pkl')
        with open('init_images.pkl','wb') as file:
            pickle.dump(names, file)
            self.images = names
        #then sort images and put into folders
        #when done maybe show text saying it's done
        #so user can then click button to show folders to see how photos were sorted
        #Given some k number of photo clusters, create k subfolders
        # Sort Photos, codes are the subfolder that each got sorted into. Second input is number of segments that are calculated for each image. k depends on how close data is. Final input is the relative size of the largest cluster

        # Sort Photos, codes are the subfolder that each got sorted into. Second input is number of segments that are calculated for each image. k depends on how close data is. Final input is the relative size of the largest cluster.
        file_idx, codes, means, k = kmf.bin_photos(self.images, 2, .3)
        print(file_idx)

        print(codes)

        subfolder_names = np.linspace(0, k-1, num=k, dtype='i').astype(str)
        print(subfolder_names)

        for subfolder_name in subfolder_names:
            os.makedirs(os.path.join('sorted_images', subfolder_name), exist_ok=True)
        print ('Subfolders created with initial images.', len(self.images), " images sorted.")
        #place images into their appropriate subfolders
        for i in range(len(file_idx)):
            src_img_path = self.images[i]
            fid = codes[i]
            #img = src_img_path.split("prack/",1)[1]
            des_loc = "sorted_images/"+str(fid)
            shutil.copy(src_img_path, des_loc)

        # Code for photo viewer
        self.group_spinBox.setMaximum(k-1)
        self.photo_spinBox.setMaximum(len(glob.glob(os.path.join("sorted_images/"+str(0)+'/', '*.jpg'))) - 1)
        self.display_image()
        # TODO: Do we need this return?
        return names

    def loadFiles_add(self):
        dir_name = "sorted_images"
        if os.path.isdir(dir_name):
            shutil.rmtree(dir_name)
        filter = "JPG (*.jpg)"
        file_name = QFileDialog()
        file_name.setFileMode(QFileDialog.ExistingFiles)
        names, _ = file_name.getOpenFileNames(self, "Open files", "C\\Desktop", filter)
        #print (names)
        #print(np.shape(load_images_from_folder(names)))

        with open('init_images.pkl', 'rb') as f:
            images = pickle.load(f)
            self.images = np.append(images, names)
        #print(self.images)
        #then sort images and put into folders
        #when done maybe show text saying it's done
        #so user can then click button to show folders to see how photos were sorted

        #Given some k number of photo clusters, create k subfolders
        file_idx, codes, means, k = kmf.bin_photos(self.images, 2, .3)
        print(codes)

        subfolder_names = np.linspace(0, k-1, num=k, dtype='i').astype(str)

        for subfolder_name in subfolder_names:
            os.makedirs(os.path.join('sorted_images', subfolder_name))

        for i in range(len(file_idx)):
            src_img_path = self.images[i]
            fid = codes[i]
            #img = src_img_path.split("prack/",1)[1]
            des_loc = "sorted_images/"+str(fid)
            shutil.copy(src_img_path, des_loc)
        print ('Subfolders created with additional images.', len(self.images), " images sorted.")
        #place images into their appropriate subfolders

    def group_change(self):
        group = self.group_spinBox.value()

        self.photo_spinBox.setMaximum(len(glob.glob(os.path.join("sorted_images/"+str(group)+'/', '*.jpg'))) - 1)

        self.photo_spinBox.setValue(0)

        self.display_image()

    def display_image(self):
        # Get index from spinBox
        group = self.group_spinBox.value()
        photo = self.photo_spinBox.value()

        folder = glob.glob(os.path.join("sorted_images/"+str(group)+'/', '*.jpg'))

        self.pix_map = QPixmap(folder[photo])
        self.image_viewer.setPixmap(self.pix_map.scaledToWidth(self.image_viewer.width()))

app = QApplication(sys.argv)
window = PhotoSorter()
window.show()
sys.exit(app.exec_())
