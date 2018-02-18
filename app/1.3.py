# Python 3 ImportError
import sys
sys.path.append('.')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import src as ya
from src.struct import ForestParams

# prettify plots
plt.rcParams['figure.figsize'] = [12.0, 3.0]
sns.set_style({"xtick.direction": "in", "ytick.direction": "in"})

b_sns, g_sns, r_sns, p_sns, y_sns, l_sns = sns.color_palette("muted")

np.random.seed(0)

# fetch data
data_train, data_query = ya.data.getData('Toy_Spiral')
N, D = data_train.shape

###########################################################################
# Grow a Tree - Forest with Single Tree
###########################################################################

forest = ya.tree.growForest(
    data_train, ForestParams(num_trees=10,
                             max_depth=5,
                             criterion='IG',
                             num_splits=5,
                             weak_learner='axis-aligned',
                             min_samples_split=5))

# num_leaves
num_leaves = 4
assert(len(forest.probs[1:]) >= num_leaves)

# matplotlib figure
fig, axes = plt.subplots(ncols=num_leaves)

# x-axis bins
bins = np.unique(data_train[:, -1]).astype(int)
# maximum y-axis value
ymax = np.max(forest.probs[1:])
for_idx = np.random.choice(range(1, len(forest.probs)), num_leaves, False)
for j in range(num_leaves):
    axes[j].bar(bins, 100*forest.probs[for_idx[j]],
                color=[b_sns, g_sns, r_sns])
    axes[j].set_title('Class histogram of\n$\\mathbf{Leaf\\ %i}$' % (j+1))
    axes[j].set_xlim([0.5, 3.5])
    axes[j].set_ylim([0, ymax*105])
    axes[j].set_xticks(bins)
plt.tight_layout()

fig.savefig('assets/1.3/leaf_cdist.pdf', format='pdf', dpi=300,
            transparent=True, bbox_inches='tight', pad_inches=0.01)
