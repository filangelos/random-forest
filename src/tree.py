import numpy as np
import typing

import src as ya
from src.struct import Node
# from src.struct import Tree
# from src.struct import ParamTree


def splitNode(data: np.ndarray,
              node: Node,
              params: typing.Tuple[int, str],
              visualise: bool = False,
              min_samples_split: int = 5,
              savefig_path: str = None):
    """"""
    iterations, learner = params
    # Initialize child nodes
    nodeL = Node(idx=[], t=np.nan, dim=0, prob=[])
    nodeR = Node(idx=[], t=np.nan, dim=0, prob=[])
    # Make this a leaf if has less than min_samples_split data points
    if len(node.idx) <= min_samples_split:
        node.t = np.nan
        node.dim = 0
        return node, nodeL, nodeR

    idx = node.idx
    data = data[idx, :]
    N, D = data.shape
    ig_best = -np.inf
    idx_best = []
    for n in range(iterations):
        # Split function
        if learner == 'axis-aligned':
            idx_, dim, t, predictor = ya.splitfunc.axis_aligned(data)
        elif learner == 'linear':
            idx_, dim, t, predictor = ya.splitfunc.polynomial(1)(data)
        elif learner == 'quadratic':
            idx_, dim, t, predictor = ya.splitfunc.polynomial(2)(data)
        elif learner == 'cubic':
            idx_, dim, t, predictor = ya.splitfunc.polynomial(3)(data)
        else:
            idx_, dim, t, predictor = ya.splitfunc.axis_aligned(data)
        # Calculate information gain
        # Based on the split that was performed
        ig = ya.information.getInformationGain(data, idx_)

        # if visualise:
        #     ya.visualise.visualise_splitfunc(
        #         idx_, data, dim, t, ig, n, predictor, learner)

        # Check that children node are not empty
        if (np.sum(idx_) > 0 and sum(~idx_) > 0):
            node, ig_best, idx_best = ya.information.updateInformationGain(
                node, ig_best, ig, t, idx_, dim, idx_best)

    nodeL.idx = idx[idx_best]
    nodeR.idx = idx[~idx_best]
    if visualise or savefig_path is not None:
        ya.visualise.visualise_splitfunc(
            idx_best, data, node.dim, node.t, ig_best, -1, predictor, learner,
            savefig_path)
        # print('Information gain = %.3f.' % ig_best)
    return node, nodeL, nodeR, ig_best


def growTree(data: np.ndarray, param: int):
    pass
