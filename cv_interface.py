import sys
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog
from PyQt5.uic import loadUi
from PyQt5 import QtWidgets
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

def load_images_from_folder(folder):
    images = []
    for filename in folder:
        img = cv2.imread(filename)
        img = cv2.resize(img, (1000,1000))
        print(np.shape(img))
        if img is not None:
            images.append(img)
    return images

class PhotoSorter(QDialog):
	def __init__(self):
		super(PhotoSorter,self).__init__()
		loadUi('photosorter.ui',self)
		self.setWindowTitle('PhotoSorter Gui')
		self.pushButton.clicked.connect(self.loadFiles)
	@pyqtSlot()
	def on_pushButton_clicked(self):
		self.label1.setText('Welcome: '+self.lineEdit1.text()+ ' '+self.lineEdit2.text())
	def loadFiles(self):

	    filter = "JPG (*.jpg)"
	    file_name = QFileDialog()
	    file_name.setFileMode(QFileDialog.ExistingFiles)
	    names, _ = file_name.getOpenFileNames(self, "Open files", "C\\Desktop", filter)
	    print (names)
	    print(np.shape(load_images_from_folder(names)))
	    #then sort images and put into folders
	    #when done maybe show text saying it's done
	    #so user can then click button to show folders to see how photos were sorted

app = QApplication(sys.argv)
window = PhotoSorter()
window.show()
sys.exit(app.exec_())