import numpy as np

from src.struct import Data


def getData(mode: str = 'Toy_Spiral',
            showImage: bool = True,
            PHOW_Sizes: list = [4, 8, 10],
            PHOW_Step: int = 8) -> Data:
    """Generate training and testing data.

    Parameters
    ----------
    mode: str
        1. Toy_Spiral
        2. Caltech 101
    showImage: bool
        Show training & testing images and their
        image feature vector (histogram representation)
    PHOW_Sizes: list
        Multi-resolution, these values determine the scale of each layer.
    PHOW_Step: int
        The lower the denser. Select from {2,4,8,16}

    Returns
    -------
    data: NamedTuple
        * data_train: numpy.ndarray
        * data_query: numpy.ndarray
    """
    if mode == 'Toy_Spiral':
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

    elif mode == 'Caltech':
        pass

    return Data(data_train, data_query)
