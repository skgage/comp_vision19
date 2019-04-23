import numpy as np
import cv2
import os
import pickle
from matplotlib import pyplot as plt
from scipy.cluster.vq import *
from sklearn.cluster import KMeans

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            print(filename)
            img = cv2.resize(img, (400,400))
            img = img.astype('uint8')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            images.append(img)
    with open('images_array.pkl','wb') as file:
            pickle.dump(images, file)
    return images
#plt.imshow(img1)
#plt.show()
#images = load_images_from_folder("/Users/sarahgage/Downloads/prack")
#print(len(images))

with open('images_array.pkl', 'rb') as f:
    #a = pickle.load(f)
    #print ('a ',a)
    images = pickle.load(f)
    print(np.shape(images))
    
def feature_cluster(images, k):
    surf = cv2.xfeatures2d.SURF_create(500)

    #compute descriptors for each image
    des_list = np.array([])
    for img in images:
            kp, des = surf.detectAndCompute(img,None)
            print(np.shape(des))
            des_list = np.append(des_list, des)
    print(np.shape(des_list))
    des_list = np.reshape(des_list, (int(len(des_list)/64),64))
    des_list = np.float32(des_list)
    print(np.shape(des_list))
    #do k-means clustering on descriptors
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    iterations = 10
    compactness, labels, centers = cv2.kmeans(des_list, k,None, criteria, iterations, cv2.KMEANS_RANDOM_CENTERS)

    print(np.shape(labels))
feature_cluster(images, k=10)

def find_similar(img1, img2, flann):
    # Initiate SURF detector
    surf = cv2.xfeatures2d.SURF_create(500)
    #print(surf.setHessianThreshold)
    # find the keypoints and descriptors with SIFT
    kp1, des1 = surf.detectAndCompute(img1,None)
    kp2, des2 = surf.detectAndCompute(img2,None)

    img3 = cv2.drawKeypoints(img1,kp1,None,(255,0,0),4)
    img4 = cv2.drawKeypoints(img2,kp2,None,(255,0,0),4)
    plt.imshow(img3),plt.show()
    plt.imshow(img4),plt.show()
    
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    matches = flann.knnMatch(des1,des2,k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    if len(good)>35:
        
        m_id = 1
        '''
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)

        img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)'''

    else:
        #print ("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
        #matchesMask = None
        m_id = 0
    '''    
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                       singlePointColor = None,
                       matchesMask = matchesMask, # draw only inliers
                       flags = 2)

    img4 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

    plt.imshow(img4, 'gray'),plt.show()'''
    
    return [m_id, len(good)]

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 15)

flann = cv2.FlannBasedMatcher(index_params, search_params)
'''img1 = cv2.imread("/Users/sarahgage/Downloads/prack/paris_general_000001.jpg")          #1st queryImage
img2 = cv2.imread("/Users/sarahgage/Downloads/prack/paris_general_000003.jpg")
[m_id, num_matches] = find_similar(img1,img2, flann)
print('Images similarity: ',m_id, ' with ', num_matches, ' number of matches.')
'''
#1 is duplicate, 0 is similar, -1 is not similar
#Testing images supposedly similar example testing
'''
sim_array = np.zeros([len(images),len(images)])
match_array = np.zeros([len(images),len(images)])
for i in range(len(images)):
    for j in range(len(images)):
        if i == j:
            sim_array[i,j] = -1
            match_array[i,j] = -1
            continue
        img1 = images[i]
        #plt.imshow(img1)
        img2 = images[j]
        #plt.imshow(img2)
        [m_id, num_matches] = find_similar(img1,img2, flann)
        sim_array[i,j] = m_id
        match_array[i,j] = num_matches
        print('Images similarity between: ',i+1, ' and ', j+1, ' with ', num_matches, ' number of matches.')

print('similiarity array: ', sim_array)

print('number of matches array ', match_array.astype(int))'''
