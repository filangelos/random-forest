import numpy as np

import src as ya


def getEntropy(data: np.ndarray) -> float:
    """Get entropy of data.

    Parameters
    ----------
    data: numpy.ndarray
        Data for processing

    Returns
    -------
    entropy: float
        Entropy of data
    """
    labels = data[:, -1]
    cdist = ya.util.histc(labels)
    cdist_norm = cdist/np.sum(cdist)
    return -np.sum(cdist_norm * np.log(cdist_norm))


def getInformationGain(data: np.ndarray, idx: np.ndarray) -> float:
    """Information Gain - the 'purity' of data labels in
    both child nodes after split. The higher the purer.

    Parameters
    ----------
    data: numpy.ndarray
        Data to split
    idx: numpy.ndarray
        Indexes of left child

    Returns
    -------
    information_gain: float
        Information Gain - Purity
    """
    L = data[idx, :]
    R = data[~idx, :]
    H = getEntropy(data)
    HL = getEntropy(L)
    HR = getEntropy(R)
    return H - np.sum(idx)/len(idx)*HL - sum(~idx)/len(idx)*HR


def updateInformationGain(node, ig_best, ig, t, idx, dim, idx_best):
    """"""
    if ig > ig_best:
        ig_best = ig
        node.t = t
        node.dim = dim
        idx_best = idx
    return node, ig_best, idx_best
