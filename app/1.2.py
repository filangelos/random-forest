# Python 3 ImportError
import sys
sys.path.append('.')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import typing
import time

import src as ya
from src.struct import SplitNodeParams

np.random.seed(0)

# fetch data
data_train, data_query = ya.data.getData('Toy_Spiral')
N, D = data_train.shape

###########################################################################
# Split functions Comparison and Sparsity
###########################################################################
for frac in [1.00, 0.50, 0.25, 0.10]:
    # random dataset
    idx = np.random.choice(range(N), int(N*frac), True)
    # root node
    root = ya.tree.Node(idx=idx, t=np.nan, dim=-2, prob=[])
    # number of splits
    numSplit = 10
    # weak learners
    kernels = ['axis-aligned', 'linear', 'quadratic', 'cubic']

    for kernel in kernels:
        # reset seed
        np.random.seed(0)
        # get information gain
        _ = ya.tree.splitNode(data_train,
                              root, SplitNodeParams(numSplit, kernel),
                              savefig_path='1.2/%s_%.2f' % (kernel, frac))

###########################################################################
# `numSplit` vs weak-learners
###########################################################################
# random dataset
idx = np.random.choice(range(N), N, True)
# root node
root = ya.tree.Node(idx=idx, t=np.nan, dim=-2, prob=[])
# range of number of splits
numSplits = [1, 5, 10, 25, 50, 100, 1000]
# weak learners
kernels = ['axis-aligned', 'linear', 'quadratic', 'cubic']

IGS = pd.DataFrame(columns=kernels, index=numSplits)
for j, numSplit in enumerate(numSplits):
    # weak-learners
    for kernel in kernels:
        # reset seed
        np.random.seed(0)
        # get information gain
        _, _, _, ig = ya.tree.splitNode(data_train,
                                        root, SplitNodeParams(numSplit, kernel))
        IGS.loc[numSplit, kernel] = ig

# table to be used for report
print('\n', IGS.to_latex(), '\n')

# we could also generate a qualitative comparison with a matrix
# of decision boundaries and IGs
# reference: Figure 4 from https://github.com/sagarpatel9410/mlcv/blob/master/CW1/report/mlcv.pdf
