import numpy as np
import typing


def histc(labels) -> typing.Tuple[np.ndarray, np.ndarray]:
    """MATLAB `histc` equivalent."""
    bins = np.unique(np.append(labels, 0)) + 1e-7
    return np.histogram(labels,
                        bins,
                        range=(bins.min() - 1e-6, bins.max() + 1e-6))


def histc_plot(labels) -> typing.Tuple[np.ndarray, np.ndarray]:
    """MATLAB `histc` equivalent."""
    bins = np.unique(labels)
    return np.bincount(labels.astype(int))[1:], bins
