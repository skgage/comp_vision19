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
import test_surf as surf

def load_images_from_folder(folder):
    images = []
    for filename in folder:
        img = cv2.imread(filename)
        img = cv2.resize(img, (1000,1000))
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
        self.pushButton3.clicked.connect(self.findDuplicates)

        # # Code for photo viewer
        self.group_spinBox.valueChanged.connect(self.group_change)
        self.photo_spinBox.valueChanged.connect(self.display_image)
        self.spinBox3.valueChanged.connect(self.dup_group_change)
        self.spinBox4.valueChanged.connect(self.dup_display_image)
    @pyqtSlot()
    def findDuplicates(self):
        folder = str(QFileDialog.getExistingDirectory(self, "Select Directory", "sorted_images"))
        
        [dup_list, filenames] = surf.find_duplicates(folder) # eg "sorted_images/0"
        print ('Duplicate list check: {}'.format(dup_list))
        dir_name = "sorted_images/duplicates"
        if os.path.isdir(dir_name):
            shutil.rmtree(dir_name)
        dup_counter = 1
        for i in range(len(dup_list)):
            img = filenames[i]
            print (img)
            if (dup_list[i] != 0):
                if ~os.path.isdir("sorted_images/duplicates/"+str(int(dup_list[i]-1))):
                    os.makedirs(os.path.join('sorted_images/duplicates/', str(int(dup_list[i]-1))), exist_ok=True)
                des_loc = "sorted_images/duplicates/"+str(int(dup_list[i]-1))
                shutil.copy(img, des_loc)

        # Code for photo viewer
        self.spinBox3.setMaximum(max(dup_list)-1)
        self.spinBox4.setMaximum(len(glob.glob(os.path.join("sorted_images/duplicates/"+str(0)+'/', '*.jpg')))-1)
        self.dup_display_image()

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

        self.file_idx, codes, self.means, k = kmf.bin_photos(self.images, 2, .3)

        subfolder_names = np.linspace(0, k-1, num=k, dtype='i').astype(str)

        for subfolder_name in subfolder_names:
            os.makedirs(os.path.join('sorted_images', subfolder_name), exist_ok=True)
        print ('Subfolders created with initial images.', len(self.images), " images sorted.")

        #place images into their appropriate subfolders
        for i in range(len(self.file_idx)):
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
        #return names

    def loadFiles_add(self):
        dir_name = "sorted_images"
        if os.path.isdir(dir_name):
            shutil.rmtree(dir_name)
        filter = "JPG (*.jpg)"
        file_name = QFileDialog()
        file_name.setFileMode(QFileDialog.ExistingFiles)
        names, _ = file_name.getOpenFileNames(self, "Open files", "C\\Desktop", filter)

        with open('init_images.pkl', 'rb') as f:
            images = pickle.load(f)
            self.images = np.append(images, names)
        with open('init_images.pkl','wb') as file:
            pickle.dump(self.images, file)
        #then sort images and put into folders
        #when done maybe show text saying it's done
        #so user can then click button to show folders to see how photos were sorted

        #Given some k number of photo clusters, create k subfolders
        self.file_idx, codes, self.means, k = kmf.re_bin_photos(self.file_idx, self.means, .3, names[0])

        subfolder_names = np.linspace(0, k-1, num=k, dtype='i').astype(str)

        for subfolder_name in subfolder_names:
            os.makedirs(os.path.join('sorted_images', subfolder_name), exist_ok=True)

        for i in range(len(self.file_idx)):
            src_img_path = self.images[i]
            fid = codes[i]
            #img = src_img_path.split("prack/",1)[1]
            des_loc = "sorted_images/"+str(fid)
            shutil.copy(src_img_path, des_loc)
        print ('Subfolders created with additional images.', len(self.images), " images sorted.")
        #place images into their appropriate subfolders

        # Code for photo viewer
        self.group_spinBox.setMaximum(k-1)
        self.photo_spinBox.setMaximum(len(glob.glob(os.path.join("sorted_images/"+str(0)+'/', '*.jpg'))) - 1)
        self.display_image()

    def group_change(self):
        group = self.group_spinBox.value()

        self.photo_spinBox.setMaximum(len(glob.glob(os.path.join("sorted_images/"+str(group)+'/', '*.jpg'))) - 1)

        self.photo_spinBox.setValue(0)

        self.display_image()

    def display_image(self):
        # Get index from spinBox
        group = self.group_spinBox.value()
        photo = self.photo_spinBox.value()
        print ('photo :', photo)

        folder = glob.glob(os.path.join("sorted_images/"+str(group)+'/', '*.jpg'))

        if len(folder) > 0:
            self.pix_map = QPixmap(folder[photo])
            self.image_viewer.setPixmap(self.pix_map.scaledToWidth(self.image_viewer.width()))

    def dup_group_change(self):
        group = self.spinBox3.value()

        self.spinBox3.setMaximum(len(glob.glob(os.path.join("sorted_images/duplicates/"+str(group)+'/', '*.jpg'))) - 1)

        self.spinBox4.setValue(0)

        self.dup_display_image()

    def dup_display_image(self):
        # Get index from spinBox
        group = self.spinBox3.value()
        photo = self.spinBox4.value()

        folder = glob.glob(os.path.join("sorted_images/duplicates/"+str(group)+'/', '*.jpg'))

        if len(folder) > 0:
            self.pix_map = QPixmap(folder[photo])
            self.duplicate_viewer.setPixmap(self.pix_map.scaledToWidth(self.duplicate_viewer.width()))

app = QApplication(sys.argv)
window = PhotoSorter()
window.show()
sys.exit(app.exec_())
