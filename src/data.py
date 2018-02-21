import os
import glob
import cv2
import numpy as np
import pickle

from sklearn.cluster import KMeans

import src as ya
from src.struct import Data

from collections import defaultdict


def getData(mode: str = 'Toy_Spiral',
            show_image: bool = True,
            num_descriptors: int = 1e5,
            num_training_samples_per_class: int = 30,
            num_testing_samples_per_class: int = 10) -> Data:
    """Generate training and testing data.

    Parameters
    ----------
    mode: str
        1. Toy_Spiral
        2. Caltech
    show_image: bool
        Show training & testing images and their
        image feature vector (histogram representation)
    num_descriptors: int
        Number of SIFT descriptors kept for BoW
    num_training_samples_per_class: int
        Number of samples per class used for training
    num_testing_samples_per_class: int
        Number of samples per class used for testing

    Returns
    -------
    data: NamedTuple
        * data_train: numpy.ndarray
        * data_query: numpy.ndarray
    """
    if mode == 'Toy_Spiral':
        return getSpiral()

    elif mode == 'Caltech':
        return getCaltech()


def getSpiral() -> Data:
    """Toy Spiral training and testing data generator.

    Returns
    -------
    data: NamedTuple
        * data_train: numpy.ndarray
        * data_query: numpy.ndarray
    """
    # TRAINING DATA
    # number of elements per class
    N = 50
    t = np.linspace(0.5, 2*np.pi, N)
    # class 1
    x1_1 = t * np.cos(t)
    x2_1 = t * np.sin(t)
    # class 2
    x1_2 = t * np.cos(t+2)
    x2_2 = t * np.sin(t+2)
    # class 3
    x1_3 = t * np.cos(t+4)
    x2_3 = t * np.sin(t+4)
    # design matrix
    X = np.concatenate(
        ((x1_1, x2_1), (x1_2, x2_2), (x1_3, x2_3)), axis=1).T
    # standardization
    X_standard = (X - X.mean()) / X.var()
    # labels
    Y = np.concatenate((np.ones(N), np.ones(N)*2, np.ones(N) * 3))
    # concatenate features with labels to single matrix: [x1 x2 y]
    data_train = np.insert(X_standard, 2, Y, axis=1)
    # TESTING DATA
    # meshgrid values
    x1, x2 = np.meshgrid(np.arange(-1.5, 1.502, 0.05),
                         np.arange(-1.5, 1.502, 0.05))
    x1_x2 = np.vstack([x1.reshape(-1), x2. reshape(-2)]).T
    # concatenate features with labels to single matrix: [x1 x2 y]
    data_query = np.insert(x1_x2, 2, np.zeros_like(x1_x2[0][0]), axis=1)
    return Data(data_train, data_query)


def getCaltech(codebook: str = 'knn',
               show_image: bool = True,
               num_features: int = 256,
               num_descriptors: int = int(1e5),
               num_training_samples_per_class: int = 25,
               num_testing_samples_per_class: int = 5,
               random_state: int = None,
               pickle_dump: bool = True,
               pickle_load: bool = False) -> Data:
    """Caltech 101 training and testing data generator.

    Parameters
    ----------
    codebook: str
        Codebook construction algorithm
    show_image: bool
        Show training & testing images and their
        image feature vector (histogram representation)
    num_features: int
        Number of BoW features
    num_descriptors: int
        Number of SIFT descriptors kept for BoW
    num_training_samples_per_class: int
        Number of samples per class used for training
    num_testing_samples_per_class: int
        Number of samples per class used for testing
    random_state: int
        `np.random.seed` initial state

    Returns
    -------
    data: NamedTuple
        * data_train: numpy.ndarray
        * data_query: numpy.ndarray
    """
    if pickle_load:
        try:
            return pickle.load(
                open('tmp/caltech_%s_%i' % (codebook, num_features), 'rb'))
        except Exception:
            pass
    # TRAINING
    # root folder with images
    folder_name = 'data/Caltech_101/101_ObjectCategories'
    # list of folders of images classes
    class_list = os.listdir(folder_name)
    # macOS: discart '.DS_Store' file
    if '.DS_Store' in class_list:
        class_list.remove('.DS_Store')
    # fix random generator state
    if random_state is not None:
        np.random.seed(random_state)

    # SIFT feature extractor
    sift = cv2.xfeatures2d.SIFT_create()
    # list of descriptors
    descriptors_train = []
    raw = defaultdict(dict)
    # iterate over image classes
    for c in range(len(class_list)):
        # subfolder pointer
        sub_folder_name = os.path.join(folder_name, class_list[c])
        # filter non-images files out
        img_list = glob.glob(os.path.join(sub_folder_name, '*.jpg'))
        # shuffle images to break correlation
        np.random.shuffle(img_list)
        # training examples
        img_train = img_list[:num_training_samples_per_class]
        # iterate over image samples of a class
        for i in range(len(img_train)):
            # fetch image sample
            img = cv2.imread(img_train[i])
            # convert to gray scale for SIFT compatibility
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # apply SIFT algorithm
            _, des = sift.detectAndCompute(gray, None)
            # store descriptors
            raw[c][i] = des
            for d in des:
                descriptors_train.append(d)
    # NumPy-friendly array of descriptors
    descriptors_train = np.asarray(descriptors_train)
    # random selection of descriptors WITHOUT REPLACEMENT
    descriptors_random = descriptors_train[np.random.choice(
        len(descriptors_train), num_descriptors, replace=False)]
    # K-Means clustering algorithm
    transformer = KMeans(n_clusters=num_features,
                         init='k-means++').fit(descriptors_random)
    # vector quantisation
    data_train = np.zeros(
        (len(class_list)*num_training_samples_per_class, num_features+1))

    for i in range(len(class_list)):
        for j in range(num_training_samples_per_class):
            # determine centers distribution
            idx = transformer.predict(raw[i][j])
            # set features
            data_train[num_training_samples_per_class *
                       (i)+j, :-1] = ya.util.histc(
                           idx, range(num_features)) / len(idx)
            # set label
            data_train[num_training_samples_per_class*(i)+j, -1] = i
    # TESTING
    raw_test = defaultdict(dict)
    # iterate over image classes
    for c in range(len(class_list)):
        # subfolder pointer
        sub_folder_name = os.path.join(folder_name, class_list[c])
        # filter non-images files out
        img_list = glob.glob(os.path.join(sub_folder_name, '*.jpg'))
        # testing examples
        img_test = img_list[num_training_samples_per_class:
                            num_training_samples_per_class +
                            num_testing_samples_per_class]
        # iterate over image samples of a class
        for i in range(len(img_test)):
            # fetch image sample
            img = cv2.imread(img_test[i])
            # convert to gray scale for SIFT compatibility
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # apply SIFT algorithm
            _, des = sift.detectAndCompute(gray, None)
            # store descriptors
            raw_test[c][i] = des
    # vector quantisation
    data_query = np.zeros(
        (len(class_list)*num_testing_samples_per_class, num_features+1))

    for i in range(len(class_list)):
        for j in range(num_testing_samples_per_class):
            # determine centers distribution
            idx = transformer.predict(raw_test[i][j])
            # set features
            data_query[num_testing_samples_per_class *
                       (i)+j, :-1] = ya.util.histc(
                           idx, range(num_features)) / len(idx)
            # set label
            data_query[num_testing_samples_per_class*(i)+j, -1] = i

    # cache data to avoid recalculation every time
    if pickle_dump:
        pickle.dump(Data(data_train, data_query), open(
            'tmp/caltech_%s_%i' % (codebook, num_features), 'wb'))

    return Data(data_train, data_query)
