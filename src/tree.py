import numpy as np
import typing

import src as ya
from src.struct import Node
from src.struct import Leaf
from src.struct import Tree
from src.struct import Forest
from src.struct import SplitNodeParams
from src.struct import TreeParams
from src.struct import ForestParams


def splitNode(data: np.ndarray,
              node: Node,
              params: SplitNodeParams,
              visualise: bool = False,
              savefig_path: str = None):
    """"""
    num_splits, weak_learner, min_samples_split = params
    # Initialize child nodes
    nodeL = Node(idx=[], t=np.nan, dim=-1, prob=[])
    nodeR = Node(idx=[], t=np.nan, dim=-1, prob=[])
    # Make this a leaf if has less than `min_samples_split` data points
    if len(node.idx) <= min_samples_split:
        node.t = np.nan
        node.dim = -1
        return node, nodeL, nodeR, None

    idx = node.idx
    data = data[idx, :]
    N, D = data.shape
    ig_best = -np.inf
    idx_best = []
    for n in range(num_splits):
        # Split function
        if weak_learner == 'axis-aligned':
            idx_, dim, t, predictor = ya.splitfunc.axis_aligned(data)
        elif weak_learner == 'linear':
            idx_, dim, t, predictor = ya.splitfunc.polynomial(1)(data)
        elif weak_learner == 'quadratic':
            idx_, dim, t, predictor = ya.splitfunc.polynomial(2)(data)
        elif weak_learner == 'cubic':
            idx_, dim, t, predictor = ya.splitfunc.polynomial(3)(data)
        else:
            idx_, dim, t, predictor = ya.splitfunc.axis_aligned(data)
        # Calculate information gain
        # Based on the split that was performed
        ig = ya.information.getInformationGain(data, idx_)

        # if visualise:
        #     ya.visualise.visualise_splitfunc(
        #         idx_, data, dim, t, ig, n, predictor, weak_learner)

        # Check that children node are not empty
        if (np.sum(idx_) > 0 and sum(~idx_) > 0):
            node, ig_best, idx_best = ya.information.updateInformationGain(
                node, ig_best, ig, t, idx_, dim, idx_best)

    nodeL.idx = idx[idx_best]
    nodeR.idx = idx[~idx_best]
    if visualise or savefig_path is not None:
        ya.visualise.visualise_splitfunc(
            idx_best, data, node.dim, node.t, ig_best, -1, predictor,
            weak_learner, savefig_path)
    return node, nodeL, nodeR, ig_best

# "Gini" vs "Information Gain"
# reference: https://datascience.stackexchange.com/questions/10228/gini-impurity-vs-entropy


def growTree(data: np.ndarray,
             idx: np.ndarray,
             cnt_total: int,
             params: TreeParams):
    """Create a Decision Tree."""
    tree = Tree(params.max_depth)
    tree.nodes[1] = Node(idx, np.nan, -2, [])
    for n in range(1, 2**(params.max_depth-1)):
        tree.nodes[n], \
            tree.nodes[2*n], \
            tree.nodes[2 * n+1], _ = splitNode(data=data,
                                               node=tree.nodes[n],
                                               params=SplitNodeParams(*params[2:]))
    cnt = 1
    probs = []
    bins = np.unique(data[:, -1])
    for n in range(1, 2**params.max_depth):
        if len(tree.nodes[n].idx) != 0:
            __idx = tree.nodes[n].idx
            __hist = ya.util.histc(data[__idx, -1], bins)
            __prob = __hist / np.sum(__hist)
            tree.nodes[n].prob = __prob

            if tree.nodes[n].dim == -1:  # leaf node
                tree.nodes[n].leaf_idx = cnt
                tree.leaves.append(Leaf(__prob, cnt_total))

                probs.append(__prob)

                cnt += 1
                cnt_total += 1
    return tree, cnt_total, probs


def growForest(data: np.ndarray, params: ForestParams):
    """Create a Random Forest."""
    N, D = data.shape
    frac = 1 - 1/np.exp(1)
    cnt_total = 1
    forest = Forest(params.num_trees)
    for T in range(params.num_trees):
        idx = np.random.choice(range(N), int(N*frac), True)
        tree, cnt_total, probs = growTree(
            data, idx, cnt_total, TreeParams(*params[1:]))
        forest.add(tree, probs)
    return forest
