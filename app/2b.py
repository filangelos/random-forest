# EXECUTION TIME:

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
sns.set_style({"xtick.direction": "in", "ytick.direction": "in"})

b_sns, g_sns, r_sns, p_sns, y_sns, l_sns = sns.color_palette("muted")

np.random.seed(13)

# fetch data
data_train, data_query = ya.data.getData('Toy_Spiral')
N, D = data_train.shape

# Supervised Data
X_train, y_train = data_train[:, :-1], data_train[:, -1]

# test points for classification
test_points = np.array([[-.5, -.7], [.4, .3], [-.7, .4], [.5, -.5]])

###########################################################################
# Validation of Hyperparameters
###########################################################################

grid_params = {'n_estimators': [1, 5, 10, 20],
               'max_depth': [2, 5, 7, 11],
               'min_samples_split': [2, 5, 10, 15],
               'min_impurity_decrease': [0.0, 0.01, 0.02, 0.05]
               }

# Cross-Validation Container
search = GridSearchCV(RandomForestClassifier(),
                      param_grid=grid_params, cv=10).fit(X_train, y_train)
# Best Parameters
best_params_ = search.best_params_

print(best_params_)

###########################################################################
# Visualization of Hyperparameters Effect
###########################################################################

# data range
r = [-1.5, 1.5]
# split function grid
xx, yy = np.meshgrid(np.linspace(*r, 200), np.linspace(*r, 200))

for param in grid_params.keys():
    kwargs = {}
    for key in grid_params.keys():
        if key != param:
            kwargs[key] = best_params_[key]
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6.0, 6.0))
    for ax, cv_param in zip(axes.flatten(), grid_params[param]):
        kwargs[param] = cv_param
        clf = RandomForestClassifier(**kwargs).fit(X_train, y_train)
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        # decision boundary line
        ax.contour(xx, yy, Z, linewidths=0.8, colors='k')
        # decision surfaces
        ax.contourf(xx,
                    yy,
                    Z,
                    cmap=plt.cm.jet.from_list(
                        'contourf', [b_sns, g_sns, r_sns], 3),
                    alpha=0.4)
        ax.set_title('%s=%s' % (param, cv_param))
        # color map
        cmap = {0: y_sns, 1: b_sns, 2: g_sns, 3: r_sns}
        # plot toy data
        ax.scatter(data_train[:, 0], data_train[:, 1], c=list(
            map(lambda l: cmap[l], data_train[:, 2])), alpha=0.4)
        # plot test points
        ax.scatter(test_points[:, 0], test_points[:, 1], c=list(
            map(lambda l: cmap[l], clf.predict(test_points)),
        ), edgecolors='k')

        fig.savefig('assets/2/%s.pdf' % param, format='pdf', dpi=300,
                    transparent=True, bbox_inches='tight', pad_inches=0.01)

###########################################################################
# Weak Learner - Visualization
###########################################################################
