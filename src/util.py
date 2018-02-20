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


def base_converter(number: int, base: int) -> typing.List[int]:
    """Convert `number` from decimal to `base` representation."""
    def _conv(_number: int, _acc: typing.List[int] = []) -> typing.List[int]:
        if _number < base:
            return [_number] + _acc
        else:
            return _conv(_number//base, [_number % base] + _acc)
    return _conv(number)
