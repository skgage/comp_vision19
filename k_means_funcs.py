import numpy as np
from scipy.cluster.vq import vq, kmeans, whiten
from scipy.spatial.distance import pdist, squareform
import glob
import os.path
import cv2
from collections import Counter

def km_cluster(img, k):

    # print(img.shape)

    x, y, channels = img.shape

    xi, yi = np.meshgrid(range(x), range(y))

    xi = np.reshape(xi, (x, y, 1))

    yi = np.reshape(yi, (x, y, 1))

    img_feat = np.concatenate((img, xi, yi), axis=2)

    reshaped = np.reshape(img_feat, (x*y, 5))

    # To normalize
    whitened = whiten(reshaped)

    # Returns the centroids
    codebook, distortion = kmeans(whitened, k)

    # centroids used to classify each point to nearest centroid
    codes, dist = vq(whitened, codebook)

    # Returned as 1D x by y grid of cluster each point belongs to
    clustered = np.reshape(codes, (x, y))

    return clustered

# Return list of mean rgb values of k clusters sorted
# from smallest to largest clusters. Dimensions k*3

def cluster_means(img, clusters):

    # k starting at 0, so we will add 1
    k = np.amax(clusters)
    k = k + 1

    cluster_sizes = [None] * k

    means = np.zeros((k, 3))

    for i in range(k):
        mask = (clusters == i)

        cluster_sizes[i] = np.sum(mask)

        mask_3 = mask[:, :, None]

        masked_img = img * mask_3

        means[i, :] = np.mean(masked_img, axis=(0, 1))

    # reorder by size
    sort_idx = np.argsort(cluster_sizes)

    means = means[sort_idx, :]

    means = np.reshape(means, (k*3))

    return means

# Calculate distances between set of image segment means
def calc_distance(segment_means):

    distances = pdist(segment_means)

    return squareform(distances)

# Import set of photos and calculate distances between their kmeans segments
def calc_photo_dists(files, k):

    # size to set
    set_height = 240

    # files = glob.glob(os.path.join(folder_path, '*.jpg'))

    num_files = len(files)

    file_idx = []

    means = np.zeros((num_files, k*3))

    for f, i in zip(files, range(num_files)):

        # print(f)

        img = cv2.imread(f)

        # print(img.shape)
        # Todo check for greyscale and convert

        # TODO resize
        (h, w, c) = img.shape
        ratio = set_height/h
        img = cv2.resize(img, None, fx = ratio, fy = ratio)

        file_idx.append(f)

        clustered = km_cluster(img, k)

        means[i, :] = cluster_means(img, clustered)

    # dists = calc_distance(means)

    return means, file_idx

# Take locations of image segments means and return binned index
# group size as fraction
def bin_photos(files, segment_k, max_group_frac):

    means, file_idx = calc_photo_dists(files, segment_k)

    folder_size = len(file_idx)

    whitened = whiten(means)

    bins = (folder_size // 10) + 1
    # Returns the centroids
    while bins < folder_size:

        codebook, distortion = kmeans(whitened, bins)

        # centroids used to classify each point to nearest centroid
        codes, d = vq(whitened, codebook)

        if (np.max(list(Counter(codes).values())) / folder_size) > max_group_frac:
            bins += 1
        else:
            break


    return file_idx, codes, means, bins


