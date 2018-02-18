import numpy as np
import typing


def histc(labels: np.ndarray,
          bins: np.ndarray = None,
          return_bins: bool = False) -> typing.Tuple[np.ndarray, np.ndarray]:
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
