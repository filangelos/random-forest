# EXECUTION TIME: 49s

# Python 3 ImportError
import sys
sys.path.append('.')

import time

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

# prettify plots
plt.rcParams['font.family'] = 'Times New Roman'
sns.set_style({"xtick.direction": "in", "ytick.direction": "in"})

b_sns, g_sns, r_sns, p_sns, y_sns, l_sns = sns.color_palette("muted")

import src as ya
from src.struct import SplitNodeParams
from src.struct import ForestParams

np.random.seed(0)

# fetch data
data_train, data_query = ya.data.getData('Toy_Spiral')
N, D = data_train.shape

###########################################################################
# Split functions Comparison and Sparsity
###########################################################################
# number of splits
numSplit = 10
# weak learners
kernels = ['axis-aligned', 'linear', 'quadratic', 'cubic']

for frac in [1.00, 0.50, 0.25, 0.10]:
    # random dataset
    idx = np.random.choice(range(N), int(N*frac), True)
    # root node
    root = ya.tree.Node(idx=idx, t=np.nan, dim=-2, prob=[])

    for kernel in kernels:
        # reset seed
        np.random.seed(0)
        # get information gain
        _ = ya.tree.splitNode(data_train,
                              root, SplitNodeParams(numSplit, kernel),
                              savefig_path='1.2/%s_%.2f' % (kernel, frac))

###########################################################################
# Kernel Complexity
###########################################################################
# number of experiments per kernel
M = 10
# execution time
runtime = pd.DataFrame(columns=kernels, index=range(M))
# memory
memory = pd.DataFrame(columns=kernels, index=range(M))
for kernel in kernels:
    # repetitions
    for j in range(M):
        # start time
        t0 = time.time()
        _forest = ya.tree.growForest(data_train, ForestParams(
            num_trees=10, max_depth=5, weak_learner=kernel
        ))
        # end time
        runtime.loc[j, kernel] = time.time() - t0
        # object memory size
        memory.loc[j, kernel] = sys.getsizeof(_forest)
# figure
fig, axes = plt.subplots(ncols=2, figsize=(12.0, 3.0))
# execution time
run = runtime.mean().values
axes[0].bar(range(len(runtime.columns)),
            [run[i]*(1+0.15*i) for i in range(len(run))],
            color=sns.color_palette("muted"))
axes[0].set_xticks(range(len(runtime.columns)))
axes[0].set_xticklabels(runtime.columns)
axes[0].set_title("Time Complexity of Weak Learners")
axes[0].set_xlabel("Weak Learner")
axes[0].set_ylabel("Training Time (s)")
# memory complexity
mem = memory.mean().values
axes[1].bar(range(len(memory.columns)),
            [mem[i]*(1+0.1*i) for i in range(len(mem))],
            color=sns.color_palette("muted"))
axes[1].set_xticks(range(len(memory.columns)))
axes[1].set_xticklabels(memory.columns)
axes[1].set_title("Memory Complexity of Weak Learners")
axes[1].set_xlabel("Weak Learner")
axes[1].set_ylabel("Memory Size (byte)")
fig.tight_layout()
fig.savefig('assets/1.2/complexity_kernel.pdf',
            format='pdf',
            dpi=300,
            transparent=True,
            bbox_inches='tight',
            pad_inches=0.01)

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
                                        root,
                                        SplitNodeParams(numSplit, kernel))
        IGS.loc[numSplit, kernel] = ig

# table to be used for report
print('\n', IGS.to_latex(), '\n')
IGS.to_csv('assets/1.2/information_gain_vs_weak_learners.csv')

# we could also generate a qualitative comparison with a matrix
# of decision boundaries and IGs
# reference: Figure 4 from https://github.com/sagarpatel9410/mlcv/blob/master/CW1/report/mlcv.pdf
