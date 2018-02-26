import os
import glob
import pickle
import typing

import numpy as np
import cv2
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import RandomTreesEmbedding

import src as ya
from src.struct import Data

from collections import defaultdict


def getData(mode: str = 'Toy_Spiral', **kwargs) -> Data:
    """Generate training and testing data.

    Parameters
    ----------
    mode: str
        1. Toy_Spiral
        2. Caltech
    kwargs: dict
        Arguments for `getSpiral` and `getCaltech` functions

    Returns
    -------
    data: NamedTuple
        * data_train: numpy.ndarray
        * data_query: numpy.ndarray
    """
    if mode == 'Toy_Spiral':
        return getSpiral()

    elif mode == 'Caltech':
        return getCaltech(**kwargs)


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


def pickle_load__getCaltech(codebook: str,
                            num_features: int) -> typing.Optional[Data]:
    try:
        return pickle.load(
            open('tmp/caltech_%s_%i.pkl' % (codebook, num_features), 'rb'))
    except Exception:
        pass


def getCaltech_pre(num_features: int = 256,
                   num_descriptors: int = 100000,
                   num_training_samples_per_class: int = 15,
                   num_testing_samples_per_class: int = 15,
                   random_state: int = None,
                   pickle_dump: bool = True):
    num_descriptors = int(num_descriptors)
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
    raw_train = defaultdict(dict)
    # plot train raw & SIFT images
    images_train = []
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
            # images to plot
            sift_img = cv2.drawKeypoints(
                gray, kp, img,
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            images_train.append((raw_img, sift_img))
    # NumPy-friendly array of descriptors
    descriptors_train = np.asarray(descriptors_train)
    # random selection of descriptors WITHOUT REPLACEMENT
    descriptors_random = descriptors_train[np.random.choice(
        len(descriptors_train), min(len(descriptors_train), num_descriptors),
        replace=False)]

    # TESTING
    raw_test = defaultdict(dict)
    # plot train raw & SIFT images
    images_test = []
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
            raw_img = cv2.imread(img_test[i])
            img = raw_img.copy()
            # convert to gray scale for SIFT compatibility
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # apply SIFT algorithm
            kp, des = sift.detectAndCompute(gray, None)
            # store descriptors
            raw_test[c][i] = des
            # images to plot
            sift_img = cv2.drawKeypoints(
                gray, kp, img,
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            images_test.append((raw_img, sift_img))

    Output = typing.NamedTuple('Output', [('class_list', list),
                                          ('descriptors_random', np.ndarray),
                                          ('raw_train', dict),
                                          ('raw_test', dict),
                                          ('images_train', list),
                                          ('images_test', list)])

    return Output(class_list, descriptors_random,
                  raw_train, raw_test,
                  images_train, images_test)


def getCaltech_plot(class_list: typing.List[str],
                    images_train: typing.List[np.ndarray],
                    images_test: typing.List[np.ndarray]) -> None:
    # iterate over image classes
    for c in range(len(class_list)):
        # TRAINING
        fig, axes = plt.subplots(ncols=2, figsize=(6.0, 3.0))
        axes[0].imshow(images_train[c][0], interpolation='nearest')
        axes[0].set_axis_off()
        axes[0].set_title('Training Sample\n%s: Original' %
                          (class_list[c].capitalize()))
        axes[1].imshow(images_train[c][1], interpolation='nearest')
        axes[1].set_axis_off()
        axes[1].set_title('Training Sample\n%s: SIFT' %
                          (class_list[c].capitalize()))
        plt.tight_layout()
        fig.savefig('assets/3.1/examples/train/%s.pdf' % class_list[c],
                    format='pdf', dpi=300, transparent=True,
                    bbox_inches='tight', pad_inches=0.01)
        # TESTING
        fig, axes = plt.subplots(ncols=2, figsize=(6.0, 3.0))
        axes[0].imshow(images_test[c][0], interpolation='nearest')
        axes[0].set_axis_off()
        axes[0].set_title('Testing Sample\n%s: Original' %
                          (class_list[c].capitalize()))
        axes[1].imshow(images_test[c][1], interpolation='nearest')
        axes[1].set_axis_off()
        axes[1].set_title('Testing Sample\n%s: SIFT' %
                          (class_list[c].capitalize()))
        plt.tight_layout()
        fig.savefig('assets/3.1/examples/test/%s.pdf' % class_list[c],
                    format='pdf', dpi=300, transparent=True,
                    bbox_inches='tight', pad_inches=0.01)


def getCaltech_KMeans(minibatch: bool = False,
                      savefig_images: bool = False,
                      num_features: int = 256,
                      num_descriptors: int = 100000,
                      num_training_samples_per_class: int = 15,
                      num_testing_samples_per_class: int = 15,
                      random_state: int = None,
                      pickle_dump: bool = True) -> Data:
    """Caltech 101 training and testing data generator.

    Parameters
    ----------
    minibatch: bool
        Flag for `MiniBatchKMeans` codebook algorithm
    savefig_images: bool
        Save raw training & testing images and their
        SIFT masked grayscale transforms
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
    class_list, descriptors_random, raw_train, raw_test, images_train, \
        images_test = getCaltech_pre(num_features, num_descriptors,
                                     num_training_samples_per_class,
                                     num_testing_samples_per_class,
                                     random_state, pickle_dump)

    if savefig_images:
        getCaltech_plot(class_list, images_train, images_test)

    # K-Means clustering algorithm
    if not minibatch:
        codebook_algorithm = KMeans(n_clusters=num_features,
                                    init='k-means++').fit(descriptors_random)
    else:
        codebook_algorithm = MiniBatchKMeans(n_clusters=num_features,
                                             init='k-means++',
                                             batch_size=num_descriptors//100
                                             ).fit(descriptors_random)

    # vector quantisation
    data_train = np.zeros(
        (len(class_list)*num_training_samples_per_class, num_features+1))

    for i in range(len(class_list)):
        for j in range(num_training_samples_per_class):
            # determine centers distribution
            idx = codebook_algorithm.predict(raw_train[i][j])
            # set features
            data_train[num_training_samples_per_class *
                       (i)+j, :-1] = ya.util.histc(
                           idx, range(num_features)) / len(idx)
            # set label
            data_train[num_training_samples_per_class*(i)+j, -1] = i

    # vector quantisation
    data_query = np.zeros(
        (len(class_list)*num_testing_samples_per_class, num_features+1))

    for i in range(len(class_list)):
        for j in range(num_testing_samples_per_class):
            # determine centers distribution
            idx = codebook_algorithm.predict(raw_test[i][j])
            # set features
            data_query[num_testing_samples_per_class *
                       (i)+j, :-1] = ya.util.histc(
                           idx, range(num_features)) / len(idx)
            # set label
            data_query[num_testing_samples_per_class*(i)+j, -1] = i

    # cache data to avoid recalculation every time
    if pickle_dump:
        codebook = 'kmeans' if minibatch else 'minibatch-kmeans'
        pickle.dump(Data(data_train, data_query), open(
            'tmp/caltech_%s_%i.pkl' % (codebook, num_features), 'wb'))

    return Data(data_train, data_query)


def getCaltech(codebook: str = 'kmeans',
               savefig_images: bool = False,
               savefig_bars: bool = False,
               num_features: int = 256,
               num_descriptors: int = 100000,
               num_training_samples_per_class: int = 15,
               num_testing_samples_per_class: int = 15,
               random_state: int = None,
               pickle_dump: bool = True,
               pickle_load: bool = False) -> Data:
    """Caltech 101 training and testing data generator.

    Parameters
    ----------
    codebook: str
        Codebook construction algorithm
    savefig_images: bool
        Save raw training & testing images and their
        SIFT masked grayscale transforms
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
    # use cached data
    if pickle_load:
        return pickle_load__getCaltech(codebook, num_features)
    # build codebook
    if codebook == 'kmeans':
        return getCaltech_KMeans(False, savefig_images,
                                 num_features, num_descriptors,
                                 num_training_samples_per_class,
                                 num_testing_samples_per_class, random_state,
                                 pickle_dump)
    elif codebook == 'minibatch-kmeans':
        return getCaltech_KMeans(True, savefig_images,
                                 num_features, num_descriptors,
                                 num_training_samples_per_class,
                                 num_testing_samples_per_class, random_state,
                                 pickle_dump)
    elif codebook == 'randon-forest':
        pass
