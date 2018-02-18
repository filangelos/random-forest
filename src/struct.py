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
    leaf_idx: int
        Leaf node index - (empty if it is not a leaf node) 
    """

    def __init__(self,
                 idx: typing.List[bool] = [],
                 t: float = None,
                 dim: int = None,
                 prob: typing.List[float] = []):
        self.idx = idx
        self.t = t
        self.dim = dim
        self.prob = prob
        self.leaf_idx = None


class Leaf:
    """Tree Leaf data structure.

    Properties
    ----------
    prob: list
        Class distribution of this node
    label: int
        Unique label of leaf
    """

    def __init__(self,
                 prob: typing.List[float] = None,
                 label: int = None):
        self.prob = prob
        self.label = label


class Tree:
    """Decision Tree class.

    Properties
    ----------
    nodes: list
        List of `Node`s
    leaves: list
        List of `Leaf`s
    """

    def __init__(self, max_depth: int):
        self.nodes = [None] + [None] * (2**(max_depth) - 1)
        self.leaves = []


class Forest:
    """Random Forest class.

    Properties
    ----------
    trees: list
        List of `Tree`s
    probs: list
        List of class distributions of leaves
    """

    def __init__(self, num_trees: int):
        self._idx_tree = 0
        self.trees = [None] * num_trees
        self.probs = [None]

    def add_tree(self, tree: Tree):
        self.trees[self._idx_tree] = tree
        self._idx_tree += 1

    def add_probs(self, probs: np.ndarray):
        for row in probs:
            self.probs.append(row)

    def add(self, tree: Tree, probs: np.ndarray):
        self.add_tree(tree)
        self.add_probs(probs)


class SplitNodeParams(typing.NamedTuple):
    """Hyperparameters for `splitNode` method.

    Properties
    ----------
    num_splits: int
        Number of random splits
    weak_learner: str {'axis-aligned', 'linear', 'quadratic', 'cubic'}
        Weak Learner function - Split Function
    min_samples_split: int
        The minimum number of samples required to split an internal node
    """
    num_splits: int
    weak_learner: str
    min_samples_split: int = 5


class TreeParams(typing.NamedTuple):
    """Hyperparameters for `growTree` method.

    Properties
    ----------
    max_depth: int
        Maximum depth of the tree
    criterion: str {'IG'}
        Split criterion for comparison
    num_splits: int
        Number of random splits
    weak_learner: str {'axis-aligned', 'linear', 'quadratic', 'cubic'}
        Weak Learner function - Split Function
    min_samples_split: int
        The minimum number of samples required to split an internal node
    """
    max_depth: int
    criterion: str = 'IG'
    num_splits: int = 10
    weak_learner: str = 'axis-aligned'
    min_samples_split: int = 5


class ForestParams(typing.NamedTuple):
    """Hyperparameters for `growForest` method.

    Properties
    ----------
    num_trees: int
        Number of trees
    max_depth: int
        Maximum depth of the tree
    criterion: str {'IG'}
        Split criterion for comparison
    num_splits: int
        Number of random splits
    weak_learner: str {'axis-aligned', 'linear', 'quadratic', 'cubic'}
        Weak Learner function - Split Function
    min_samples_split: int
        The minimum number of samples required to split an internal node
    """
    num_trees: int
    max_depth: int
    criterion: str = 'IG'
    num_splits: int = 10
    weak_learner: str = 'axis-aligned'
    min_samples_split: int = 5


class Data(typing.NamedTuple):
    """"""
    data_train: np.ndarray
    data_query: np.ndarray
