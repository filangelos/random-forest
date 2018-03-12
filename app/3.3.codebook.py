import os
import glob

from collections import defaultdict

import numpy as np
import cv2

from sklearn.ensemble import RandomTreesEmbedding


def histc(labels, bins=None, return_bins=False):
    """MATLAB `histc` equivalent."""
    labels = np.array(labels, dtype=int)
    if bins is None:
        bins = np.unique(labels)
    bins = np.array(bins, dtype=int)
    bincount = np.bincount(labels)
    if len(bins) + 1 != len(bincount):
        bincount = np.append(
            bincount, [0 for _ in range(len(bins) + 1 - len(bincount))])
    if return_bins:
        return bincount[bins], bins
    else:
        return bincount[bins]


def RandomForest_Codebook(num_features, num_descriptors):
    # root folder with images
    folder_name = 'data/Caltech_101/101_ObjectCategories'
    # list of folders of images classes
    class_list = os.listdir(folder_name)
    # macOS: discart '.DS_Store' file
    if '.DS_Store' in class_list:
        class_list.remove('.DS_Store')

    # SIFT feature extractor
    sift = cv2.xfeatures2d.SIFT_create()

    # TRAINING
    # list of descriptors
    descriptors_train = []
    raw_train = defaultdict(dict)
    # iterate over image classes
    for c in range(len(class_list)):
        # subfolder pointer
        sub_folder_name = os.path.join(folder_name, class_list[c])
        # filter non-images files out
        img_list = glob.glob(os.path.join(sub_folder_name, '*.jpg'))
        # shuffle images to break correlation
        np.random.shuffle(img_list)
        # training examples
        img_train = img_list[:15]
        # iterate over image samples of a class
        for i in range(len(img_train)):
            # fetch image sample
            raw_img = cv2.imread(img_train[i])
            img = raw_img.copy()
            # convert to gray scale for SIFT compatibility
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # apply SIFT algorithm
            kp, des = sift.detectAndCompute(gray, None)
            # store descriptors
            raw_train[c][i] = des
            for d in des:
                descriptors_train.append(d)
    # NumPy-friendly array of descriptors
    descriptors_train = np.asarray(descriptors_train)
    # random selection of descriptors WITHOUT REPLACEMENT
    descriptors_random = descriptors_train[np.random.choice(
        len(descriptors_train), min(len(descriptors_train),
                                    num_descriptors),
        replace=False)]

    # TESTING
    raw_test = defaultdict(dict)
    # iterate over image classes
    for c in range(len(class_list)):
        # subfolder pointer
        sub_folder_name = os.path.join(folder_name, class_list[c])
        # filter non-images files out
        img_list = glob.glob(os.path.join(sub_folder_name, '*.jpg'))
        # testing examples
        img_test = img_list[15:30]
        # iterate over image samples of a class
        for i in range(len(img_test)):
            # fetch image sample
            raw_img = cv2.imread(img_test[i])
            img = raw_img.copy()
            # convert to gray scale for SIFT compatibility
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # apply SIFT algorithm
            kp, des = sift.detectAndCompute(gray, None)
            # store descriptors
            raw_test[c][i] = des

    # K-Means clustering algorithm
    codebook_algorithm = RandomTreesEmbedding(
        n_estimators=num_features).fit(descriptors_random)

    n_out = codebook_algorithm.transform(raw_train[0][0]).sum(axis=0).shape[1]

    # vector quantisation
    data_train = np.zeros(
        (len(class_list)*15, n_out+1))

    for i in range(len(class_list)):
        for j in range(15):
            # set features
            data_train[15 * (i)+j, :-1] = codebook_algorithm.transform(
                raw_train[i][j]).sum(axis=0).ravel()
            # set label
            data_train[15*(i)+j, -1] = i

    # vector quantisation
    data_query = np.zeros(
        (len(class_list)*15, n_out+1))

    for i in range(len(class_list)):
        for j in range(15):
            # set features
            data_query[15 *
                       (i)+j, :-1] = codebook_algorithm.transform(
                raw_test[i][j]).sum(axis=0).ravel()
            # set label
            data_query[15*(i)+j, -1] = i

    return data_train, data_query
