import numpy as np
import typing


class Node:
    """Tree Node data structure.

    Properties
    ----------
    idx: list
        Data (index only) which split into this node
    t: float
        Threshold of split function
    dim: int
        Feature dimension of split function
    prob: list
        Class distribution of this node
    """

    def __init__(self,
                 idx: typing.List[bool],
                 t: float,
                 dim: int,
                 prob: typing.List[float]):
        self.idx = idx
        self.t = t
        self.dim = dim
        self.prob = prob


class Tree:
    """Decision Tree class.

    Properties
    ----------
    nodes: list
        List of `Node`s
    leaves: list
        List of `Leaf`s
    """


"""
%        Base               Each node stores:
%         1                   trees.idx       - data (index only) which split into this node
%        / \                  trees.t         - threshold of split function
%       2   3                 trees.dim       - feature dimension of split function
%      / \ / \                trees.prob      - class distribution of this node
%     4  5 6  7               trees.leaf_idx  - leaf node index (empty if it is not a leaf node) 
"""


class Data(typing.NamedTuple):
    """"""
    data_train: np.ndarray
    data_query: np.ndarray
