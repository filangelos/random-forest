import numpy as np
import typing

from sklearn.preprocessing import PolynomialFeatures


def axis_aligned(data: np.ndarray) -> typing.Tuple[typing.List[bool],
                                                   int,
                                                   float,
                                                   typing.Callable
                                                   [[np.ndarray, int, float],
                                                    np.ndarray]]:
    """Axis-Aligned Split Function."""
    # Pick one random dimension
    # Use D-1 because the last column will hold the class
    D = data.shape[1]
    dim = np.random.randint(D-1)
    # Data range of random dimension
    d_min = np.min(data[:, dim]) + 1e-6
    d_max = np.max(data[:, dim]) - 1e-6
    # Pick random value within the range as threshold
    # d_min + np.random.rand()*(d_max-d_min)
    t = np.random.uniform(d_min, d_max)
    # return index of LEFT node
    idx_ = data[:, dim] < t
    return idx_, dim, t, lambda _data, _dim, _t: _data[:, _dim] >= _t


def polynomial(degree: int) -> typing.Callable[[np.ndarray],
                                               typing.Tuple[typing.List[bool],
                                                            int,
                                                            float,
                                                            typing.Callable
                                                            [[np.ndarray,
                                                              np.ndarray,
                                                              float],
                                                             np.ndarray]]]:
    """High-Order-Function for polynomial kernels."""
    def kernel(data: np.ndarray) -> typing.Tuple[typing.List[bool],
                                                 int,
                                                 float,
                                                 typing.Callable
                                                 [[np.ndarray,
                                                   np.ndarray,
                                                   float],
                                                  np.ndarray]]:
        """Generic polynomial kernel."""
        # Polynomial features
        # exclude first, bias, term
        poly = PolynomialFeatures(degree)
        features = poly.fit_transform(
            data[:, :-1])[:, 1:]
        # Axis ranges
        axis_max = np.max(features, axis=0)
        axis_min = np.min(features, axis=0)
        # Pick random coefficients
        dim = np.array(
            [np.random.uniform(axis_min[i], axis_max[i])
             for i in range(axis_max.shape[0])])
        # restrict threshold in range
        t = dim[0]**2
        # return index of LEFT node
        idx_ = np.dot(features, dim) < t
        return idx_, \
            dim, t, \
            lambda _data, _dim, _t: np.dot(poly.fit_transform(
                _data)[:, 1:], _dim) >= _t
    return kernel
