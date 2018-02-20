# Python 3 ImportError
import sys
sys.path.append('.')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import src as ya
from sklearn import tree
import graphviz

# prettify plots
sns.set_style({"xtick.direction": "in", "ytick.direction": "in"})

b_sns, g_sns, r_sns, p_sns, y_sns, l_sns = sns.color_palette("muted")

np.random.seed(0)

# fetch data
data_train, data_query = ya.data.getData('Toy_Spiral')
N, D = data_train.shape

###########################################################################
# Visualize Leaf Distributions
###########################################################################

# Supervised Data
X_train, y_train = data_train[:, :-1], data_train[:, -1]

# Decision Tree Classifier Training
clf = tree.DecisionTreeClassifier(criterion='entropy',
                                  max_depth=5,
                                  min_samples_split=5,
                                  min_impurity_decrease=0.05).fit(X_train, y_train)

###########################################################################
# Grow a Tree - Visualize Leaf Distributions
###########################################################################

# Leave Indexes
leaves_idx = (clf.tree_.children_left == -1) & (clf.tree_.children_right == -1)

# Number of samples at leaves
leaves_values = np.squeeze(clf.tree_.value[leaves_idx], axis=1)

# Leaves Distributions
leaves_dist = np.apply_along_axis(lambda r: r/np.sum(r), 1, leaves_values)

# num_leaves
ncols = 4
nrows = 2
plt.rcParams['figure.figsize'] = [4.0 * ncols, 4.0 * nrows]
num_leaves = nrows * ncols
# check if leaves available for visualization
assert(leaves_dist.shape[0] >= num_leaves)

# matplotlib figure
fig, axes = plt.subplots(nrows=nrows, ncols=ncols)

# x-axis bins
bins = np.unique(y_train).astype(int)
# maximum y-axis value
ymax = np.max(leaves_dist)
# for_idx = np.random.choice(len(leaves_dist), num_leaves, False)
for_idx = range(len(leaves_dist))
for j, ax in enumerate(axes.flatten()):
    ax.bar(bins, 100*leaves_dist[for_idx[j]],
           color=[b_sns, g_sns, r_sns])
    ax.set_title('Class histogram of\n$\\mathbf{Leaf\\ %i}$' % (j+1))
    ax.set_xlim([0.5, 3.5])
    ax.set_ylim([0, ymax*105])
    ax.set_xticks(bins)
plt.tight_layout()

fig.savefig('assets/1.3/leaf_cdist.pdf', format='pdf', dpi=300,
            transparent=True, bbox_inches='tight', pad_inches=0.01)

###########################################################################
# Visualize Tree - Using `graphviz`
###########################################################################

# dot graph
dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names=['X1', 'X2'],
                                filled=True, rounded=True,
                                special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("assets/1.3/graph")
