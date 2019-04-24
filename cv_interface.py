import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog
from PyQt5.uic import loadUi
from PyQt5 import QtWidgets
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os, shutil

def load_images_from_folder(folder):
    images = []
    for filename in folder:
        img = cv2.imread(filename)
        img = cv2.resize(img, (1000,1000))
        print(np.shape(img))
        if img is not None:
            images.append(img)
    return images

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s

class PhotoSorter(QDialog):
	def __init__(self):
		super(PhotoSorter,self).__init__()
		loadUi('photosorter.ui',self)
		self.setWindowTitle('PhotoSorter Gui')
		names = self.pushButton1.clicked.connect(self.loadFiles)
		names2 = np.append(names, self.pushButton2.clicked.connect(self.loadFiles))
		print(names2)
	@pyqtSlot()
	def on_pushButton_clicked(self):
		self.label1.setText('Welcome: '+self.lineEdit1.text()+ ' '+self.lineEdit2.text())
	def loadFiles(self):
		#when we want to sort new images, sorted images folder should be removed
		dir_name = "/Users/sarahgage/cv19/sorted_images"
		if os.path.isdir(dir_name):
			shutil.rmtree(dir_name)
		filter = "JPG (*.jpg)"
		file_name = QFileDialog()
		file_name.setFileMode(QFileDialog.ExistingFiles)
		names, _ = file_name.getOpenFileNames(self, "Open files", "C\\Desktop", filter)
		print (names)
		print(np.shape(load_images_from_folder(names)))
		files=[]
		for file in os.listdir("/Users/sarahgage/Downloads/prack/"):
			if file.endswith(".jpg"):
				files.append(os.path.join(os.getcwd(), file))

		for x in files:

			item = QtWidgets.QListWidgetItem()
			icon = QtGui.QIcon()
			icon.addPixmap(QtGui.QPixmap(_fromUtf8(x)), QtGui.QIcon.Normal, QtGui.QIcon.Off)
			item.setIcon(icon)
			self.listWidget.addItem(item)
		return names
	    #then sort images and put into folders
	    #when done maybe show text saying it's done
	    #so user can then click button to show folders to see how photos were sorted
	    #Given some k number of photo clusters, create k subfolders

	    #k = 1
	    #subfolder_names = np.linspace(0,k-1,num=k).astype(str)
	    #for subfolder_name in subfolder_names:
	    	#os.makedirs(os.path.join('sorted_images', subfolder_name))'''
	    #place images into their appropriate subfolders



app = QApplication(sys.argv)
window = PhotoSorter()
window.show()
sys.exit(app.exec_())