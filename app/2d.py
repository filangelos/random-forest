# EXECUTION TIME: 7s

# Python 3 ImportError
import sys
sys.path.append('.')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures

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

# Supervised Data
X_train, y_train = data_train[:, :-1], data_train[:, -1]

# test points for classification
test_points = np.array([[-.5, -.7], [.4, .3], [-.7, .4], [.5, -.5]])

###########################################################################
# Visualization of Decision Boundaries
###########################################################################

# parameter configurations
configurations = {
    'max_depth': [
        # max_depth 2
        {'max_depth': 2, 'min_samples_split': 5, 'n_estimators': 2},
        # max_depth 5
        {'max_depth': 5, 'n_estimators': 20},
        # max_depth 7
        {'max_depth': 7, 'n_estimators': 4},
        # max_depth 11
        {'max_depth': 11, 'n_estimators': 4}
    ],
    'n_estimators': [
        # n_estimators 1
        {'max_depth': 5, 'n_estimators': 1},
        # n_estimators 5
        {'max_depth': 5, 'n_estimators': 5},
        # n_estimators 10
        {'max_depth': 7, 'max_features': 2, 'n_estimators': 10},
        # n_estimators 20
        {'max_depth': 7, 'max_features': 2, 'n_estimators': 20},
    ],
    'min_impurity_decrease': [
        # min_impurity_decrease 0.05
        {'max_depth': 5, 'min_impurity_decrease': 0.05, 'n_estimators': 5},
        # min_impurity_decrease 0.02
        {'max_depth': 5, 'min_impurity_decrease': 0.02, 'n_estimators': 5},
        # min_impurity_decrease 0.01
        {'max_depth': 5, 'min_impurity_decrease': 0.01, 'n_estimators': 10},
        # min_impurity_decrease 0.0
        {'max_depth': 7, 'min_impurity_decrease': 0.005, 'n_estimators': 20},
    ],
    'min_samples_split': [
        # min_samples_split 15
        {'max_depth': 5, 'min_samples_split': 15, 'n_estimators': 3},
        # min_samples_split 10
        {'max_depth': 5, 'min_samples_split': 10, 'n_estimators': 3},
        # min_samples_split 5
        {'max_depth': 5, 'min_samples_split': 5, 'n_estimators': 20},
        # min_samples_split 2
        {'max_depth': 5, 'min_samples_split': 2, 'n_estimators': 7},
    ]
}

# title of plots
pretty_params = {
    'n_estimators': 'Number of Trees',
    'max_depth': 'Maximum of Tree Depth',
    'min_impurity_decrease': 'Number of Splits',
    'min_samples_split': 'Minimum Samples at Node'
}

# data range
r = [-1.5, 1.5]
# split function grid
xx, yy = np.meshgrid(np.linspace(*r, 200), np.linspace(*r, 200))

for param_name, config_grid in configurations.items():
    # figures
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6.0, 6.0))
    for ax, config in zip(axes.flatten(), config_grid):
        # classifier
        clf = RandomForestClassifier(**config).fit(X_train, y_train)
        # decision boundary calculation
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
        if param_name != 'min_impurity_decrease':
            ax.set_title('%s = %s' %
                         (pretty_params[param_name], config[param_name]))
        else:
            ax.set_title('%s = %s' %
                         (pretty_params[param_name],
                          int(np.ceil(0.05/config[param_name]))))
        # color map
        cmap = {0: y_sns, 1: b_sns, 2: g_sns, 3: r_sns}
        # plot toy data
        ax.scatter(data_train[:, 0], data_train[:, 1], c=list(
            map(lambda l: cmap[l], data_train[:, 2])), alpha=0.4)
        # plot test points
        ax.scatter(test_points[:, 0], test_points[:, 1], c=list(
            map(lambda l: cmap[l], clf.predict(test_points)),
        ), edgecolors='k')
        # y-ticks
        ax.set_yticks([-1, 0, 1])

    fig.savefig('assets/2/boundaries/%s.pdf' % param_name, format='pdf',
                dpi=300, transparent=True, bbox_inches='tight', pad_inches=0.01)

# classifier parameters
poly_configurations = [
    # axis-aligned
    {'max_depth': 5, 'min_samples_split': 5, 'n_estimators': 10},
    # linear
    {'max_depth': 5, 'min_samples_split': 5, 'n_estimators': 5},
    # quadratic
    {'max_depth': 5, 'min_samples_split': 5, 'n_estimators': 3},
    # cubic
    {'max_depth': 11, 'min_samples_split': 5, 'n_estimators': 3}
]
# weak learners
learners = [
    (1, 'axis-aligned'),
    (1, 'linear'),
    (2, 'quadratic'),
    (3, 'cubic')
]

# figures
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6.0, 6.0))
for ax, config, (degree, learner) in zip(axes.flatten(),
                                         poly_configurations, learners):
    # polynomial feature preprocessor
    poly = PolynomialFeatures(degree)
    # new features
    X_poly = poly.fit_transform(X_train)
    # classifier
    clf = RandomForestClassifier(**config).fit(X_poly, y_train)
    # decision boundary calculation
    Z = clf.predict(poly.transform(
        np.c_[xx.ravel(), yy.ravel()])).reshape(xx.shape)
    # decision boundary line
    ax.contour(xx, yy, Z, linewidths=0.8, colors='k')
    # decision surfaces
    ax.contourf(xx,
                yy,
                Z,
                cmap=plt.cm.jet.from_list(
                    'contourf', [b_sns, g_sns, r_sns], 3),
                alpha=0.4)
    ax.set_title('%s: $\\mathtt{%s}$' %
                 ('Weak Learner', learner))
    # color map
    cmap = {0: y_sns, 1: b_sns, 2: g_sns, 3: r_sns}
    # plot toy data
    ax.scatter(data_train[:, 0], data_train[:, 1], c=list(
        map(lambda l: cmap[l], data_train[:, 2])), alpha=0.4)
    # plot test points
    ax.scatter(test_points[:, 0], test_points[:, 1], c=list(
        map(lambda l: cmap[l], clf.predict(poly.transform(test_points))),
    ), edgecolors='k')
    # y-ticks
    ax.set_yticks([-1, 0, 1])

fig.savefig('assets/2/boundaries/%s.pdf' % 'weak_learner', format='pdf',
            dpi=300, transparent=True, bbox_inches='tight', pad_inches=0.01)
