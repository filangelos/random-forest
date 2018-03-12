# EXECUTION TIME: 6s

# Python 3 ImportError
import sys
sys.path.append('.')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

import src as ya
from src.struct import ForestParams

# prettify plots
plt.rcParams['font.family'] = 'Times New Roman'
sns.set_style({"xtick.direction": "in", "ytick.direction": "in"})

b_sns, g_sns, r_sns, p_sns, y_sns, l_sns = sns.color_palette("muted")

np.random.seed(13)

# fetch data
data_train, data_query = ya.data.getData('Toy_Spiral')
N, D = data_train.shape

###########################################################################
# Grow a Forest
###########################################################################

# Supervised Data
X_train, y_train = data_train[:, :-1], data_train[:, -1]

# test points for classification
test_points = np.array([[-.5, -.7], [.4, .3], [-.7, .4], [.5, -.5]])

# number of trees
n_estimators = 3

forest = RandomForestClassifier(n_estimators=n_estimators,
                                criterion='entropy',
                                max_depth=5,
                                min_samples_split=5,
                                min_impurity_decrease=0.05
                                ).fit(X_train, y_train)

# number of classes
n_classes = forest.n_classes_

###########################################################################
# Visualization of Class Distributions
###########################################################################

# the class distributions of the leaf nodes
# which the data point arrives
distributions = np.empty((len(test_points), n_estimators, n_classes))
for i, x_test in enumerate(test_points):
    dist = np.empty((n_estimators, n_classes))
    for j, tree in enumerate(forest.estimators_):
        dist[j] = tree.predict_proba(x_test.reshape(1, -1)).ravel()
    distributions[i, :, :] = dist

# the averaged class distribution
averaged_distributions = forest.predict_proba(test_points)

# matplotlib figure
nrows = len(test_points)
ncols = forest.n_estimators + 2
plt.rcParams['figure.figsize'] = [3.0 * ncols, 3.0 * nrows]
fig, axes = plt.subplots(nrows=nrows, ncols=ncols)

# x-axis bins
bins = np.unique(y_train).astype(int)
# maximum y-axis value
ymax = np.max(distributions)

# data range
r = [-1.5, 1.5]
# split function grid
xx, yy = np.meshgrid(np.linspace(*r, 200), np.linspace(*r, 200))

for i in range(len(test_points)):
    for j, ax in enumerate(axes[i][:-1].flatten()):
        if j < forest.n_estimators:
            dist = distributions[i, j]
            title = 'Test Point %i - Tree %i' % (i+1, j+1)
        else:
            dist = averaged_distributions[i]
            title = 'Test Point %i - Averaged' % (i+1)
        ax.bar(bins, 100*dist,
               color=[b_sns, g_sns, r_sns])
        ax.set_title(title)
        ax.set_xlim([0.5, 3.5])
        ax.set_ylim([0, ymax*105])
        ax.set_xticks(bins)
    # color map
    cmap = {0: y_sns, 1: b_sns, 2: g_sns, 3: r_sns}
    # plot toy data
    axes[i][-1].scatter(data_train[:, 0], data_train[:, 1], c=list(
        map(lambda l: cmap[l], data_train[:, 2])), alpha=0.75)
    # test points
    axes[i][-1].scatter(test_points[i, 0], test_points[i, 1], c=list(
        map(lambda l: cmap[l], forest.predict([test_points[i]])),
    ), edgecolors='k')
    # axis limits
    axes[i][-1].set_xlim(r)
    axes[i][-1].set_ylim(r)
    # title
    axes[i][-1].set_title('Evaluating Test Point %i' % (i+1))
    # # grid predictions
    Z = forest.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # # decision boundary line
    axes[i][-1].contour(xx, yy, Z, linewidths=0.8, colors='k')
    # # decision surfaces
    _colors = np.ones((3, 3))
    _idx = int(forest.predict([test_points[i]])) - 1
    _colors[_idx] = [b_sns, g_sns, r_sns][_idx]
    axes[i][-1].contourf(xx,
                         yy,
                         Z,
                         cmap=plt.cm.jet.from_list(
                             'contourf', _colors, 3),
                         alpha=0.4)


fig.tight_layout()
fig.savefig('assets/2/eval_test_points.pdf', format='pdf', dpi=300,
            transparent=True, bbox_inches='tight', pad_inches=0.01)
