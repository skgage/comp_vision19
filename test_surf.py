import numpy as np
import cv2
import os
import pickle
from matplotlib import pyplot as plt
from scipy.cluster.vq import *
from sklearn.cluster import KMeans
import time

def load_images_from_folder(folder):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        filenames.append(os.path.join(folder,filename))
        if img is not None:
            #print(filename)
            img = cv2.resize(img, (400,400))
            img = img.astype('uint8')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            images.append(img)
    #with open('images_array.pkl','wb') as file:
            #pickle.dump(images, file)
    return [images, filenames]


def check_similarity(img1, img2):
    # Initiate SURF detector
    surf = cv2.xfeatures2d.SURF_create(400)
    # find the keypoints and descriptors with SIFT
    kp1, des1 = surf.detectAndCompute(img1,None)
    kp2, des2 = surf.detectAndCompute(img2,None)

    img3 = cv2.drawKeypoints(img1,kp1,None,(255,0,0),4)
    img4 = cv2.drawKeypoints(img2,kp2,None,(255,0,0),4)
    #plt.imshow(img3),plt.show()
    #plt.imshow(img4),plt.show()
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 1)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    if len(good)>35:
        m_id = 1
    else:
        m_id = 0
    percent_good = len(good)/len(matches)
    return [m_id, len(good), percent_good]

def find_duplicates(folder):
    [images, filenames] = load_images_from_folder(folder)
    start2 = time.time()
    num_images = len(images)
    dup_list = np.zeros(num_images)
    #sim_array = np.zeros([num_images,num_images])
    #match_array = np.zeros([num_images,num_images])
    dup_counter = 1
    for i in range(num_images):
        for j in range(i+1,num_images):
            if (i == j or (dup_list[i] != 0 and dup_list[j] != 0)):
                continue
            img1 = images[i]
            img2 = images[j]

            [m_id, num_matches, percent_good] = check_similarity(img1,img2)

            if num_matches >= 20:
                dup_list[i] = dup_counter
                dup_list[j] = dup_counter
                #print('Images duplicates are: ',i+1, ' and ', j+1, ' with ', num_matches, ' number of matches and ', percent_good, ' good match percentage.')
        dup_counter = max(dup_list)+1
    #print ('duplicates list: ', dup_list)
    #end2 = time.time()
    #print ('Execution time: {} sec'.format(end2-start2))
    return [dup_list, filenames]

