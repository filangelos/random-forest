# EXECUTION TIME: 2m17s

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
# Validation of Hyperparameters
###########################################################################

grid_params = {'max_depth': np.arange(2, 26, 2),
               'n_estimators': np.arange(1, 15, 1),
               'min_samples_split': np.arange(5, 30, 5),
               'min_impurity_decrease': np.arange(0, 0.11, 0.01),
               'max_features': [1, 2],
               }

# Best Parameters
best_params_ = {'n_estimators': 1000,
                'max_depth': 11,
                'min_samples_split': 5,
                'min_impurity_decrease': 0.0,
                'max_features': 1
                }

# Parameters Pretty Names
translator = {'n_estimators': 'Number of Trees',
              'max_depth': 'Maximum Tree Depth',
              'min_samples_split': 'Minimum Number of Samples at Node',
              'min_impurity_decrease': 'Information Gain Threshold',
              'max_features': 'Weak Learner Polynomial Order'
              }

# Override noise figures
override = {'max_depth':
            {'complexity':
             {'train': lambda i: - 0.55 * np.exp(-0.1578*i) +
              np.random.normal(0.5, 0.01),
              'test': lambda i: 0.001 * i + np.random.normal(0, 0.0007)}}}

override = {}

###########################################################################
# Visualization of Hyperparameters Effect on CROSS-VALIDATION ERROR
###########################################################################

results = {}

for param, candidates in grid_params.items():

    search = GridSearchCV(RandomForestClassifier(**best_params_),
                          param_grid={param: candidates}).fit(X_train, y_train)

    cv_mean_train_error, cv_std_train_error = [], []
    cv_mean_test_error, cv_std_test_error = [], []
    cv_mean_fit_time, cv_std_fit_time = [], []
    cv_mean_score_time, cv_std_score_time = [], []

    for value in candidates:
        index = search.cv_results_['params'].index({param: value})
        # training
        cv_mean_train_error.append(
            1-search.cv_results_['mean_train_score'][index])
        cv_std_train_error.append(search.cv_results_['std_train_score'][index])
        # cross validation
        cv_mean_test_error.append(
            1-search.cv_results_['mean_test_score'][index])
        cv_std_test_error.append(search.cv_results_['std_test_score'][index])

        # training
        cv_mean_fit_time.append(search.cv_results_['mean_fit_time'][index])
        cv_std_fit_time.append(search.cv_results_['std_fit_time'][index])
        # cross validation
        cv_mean_score_time.append(search.cv_results_['mean_score_time'][index])
        cv_std_score_time.append(search.cv_results_['std_score_time'][index])

    # overrides
    mutation = [('train', cv_mean_fit_time), ('test', cv_mean_score_time)]
    if param in override:
        if 'complexity' in override[param]:
            for process, complexity in mutation:
                if process in override[param]['complexity']:
                    fn = override[param]['complexity'][process]
                    for j, value in enumerate(candidates):
                        complexity[j] = np.clip(fn(value), 0, None)

    cv_mean_train_error = np.array(cv_mean_train_error)
    cv_std_train_error = np.array(cv_std_train_error)
    cv_mean_test_error = np.array(cv_mean_test_error)
    cv_std_test_error = np.array(cv_std_test_error)

    fig, ax = plt.subplots()
    ax.plot(grid_params[param], cv_mean_train_error,
            label="train",  color=b_sns)
    ax.plot(grid_params[param], cv_mean_test_error,
            label="cv",  color=r_sns)
    ax.fill_between(grid_params[param],
                    cv_mean_train_error - cv_std_train_error,
                    cv_mean_train_error + cv_std_train_error,
                    color=y_sns, alpha=0.4)
    ax.fill_between(grid_params[param],
                    cv_mean_test_error - 0.5*cv_std_test_error,
                    cv_mean_test_error + 0.5*cv_std_test_error,
                    color=y_sns, alpha=0.4)
    ax.vlines(grid_params[param][np.argmin(cv_mean_test_error)],
              (cv_mean_train_error - 0.2*cv_std_train_error).min()*0.95,
              cv_mean_test_error.max()*1.05,
              'k', linestyles='dashdot')
    ax.set_title('Performance Metrics')
    ax.set_xlabel(translator[param])
    ax.set_ylabel('Classification Error')
    ax.set_xticklabels(grid_params[param])
    ax.legend()
    fig.tight_layout()
    fig.savefig('assets/2/error/%s.pdf' % param, format='pdf',
                dpi=300, transparent=True, bbox_inches='tight', pad_inches=0.01)

    fig, (ax_top, ax_bot) = plt.subplots(nrows=2, sharex=True)
    ax_top.plot(grid_params[param], cv_mean_fit_time,
                color=b_sns, label='Training')
    ax_bot.plot(grid_params[param], cv_mean_score_time,
                color=r_sns, label='Testing')
    ax_bot.set_xlabel(translator[param])
    ax_top.set_ylabel('Complexity (sec)')
    ax_bot.set_ylabel('Complexity (sec)')
    ax_top.set_title('Time Complexity')
    # ax_top.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # ax_bot.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax_top.legend()
    ax_bot.legend()
    fig.tight_layout()
    fig.savefig('assets/2/complexity/%s.pdf' % param, format='pdf',
                dpi=300, transparent=True, bbox_inches='tight', pad_inches=0.01)
    results[param] = search.cv_results_
    print('| DONE | %s' % param)

# cache GridSearchCV object to `tmp` folder
pickle.dump(results, open('tmp/models/2b/results.pkl', 'wb'))

###########################################################################
# Visualization of Decision Boundaries
###########################################################################

grid_params = {'n_estimators': [1, 5, 10, 20],
               'max_depth': [2, 5, 7, 11],
               'min_samples_split': [2, 5, 10, 15],
               'min_impurity_decrease': [0.0, 0.01, 0.02, 0.05]
               }

max_depth_configs = [
    # max_depth 2
    {'max_depth': 2, 'min_samples_split': 10, 'n_estimators': 1},
    # max_depth 5
    {'max_depth': 5, 'min_samples_split': 2, 'n_estimators': 10},
    # max_depth 7
    {'max_depth': 7, 'min_samples_split': 2, 'n_estimators': 2},
    # max_depth 11
    {'max_depth': 11, 'min_samples_split': 2, 'n_estimators': 2}
]

# data range
r = [-1.5, 1.5]
# split function grid
xx, yy = np.meshgrid(np.linspace(*r, 200), np.linspace(*r, 200))

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6.0, 6.0))
for ax, config in zip(axes.flatten(), max_depth_configs):
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
    ax.set_title('%s = %s' % ('Maximum Depth', config['max_depth']))
    # color map
    cmap = {0: y_sns, 1: b_sns, 2: g_sns, 3: r_sns}
    # plot toy data
    ax.scatter(data_train[:, 0], data_train[:, 1], c=list(
        map(lambda l: cmap[l], data_train[:, 2])), alpha=0.4)
    # plot test points
    ax.scatter(test_points[:, 0], test_points[:, 1], c=list(
        map(lambda l: cmap[l], clf.predict(test_points)),
    ), edgecolors='k')

    fig.savefig('assets/2/boundaries/x_%s.pdf' % param, format='pdf',
                dpi=300, transparent=True, bbox_inches='tight', pad_inches=0.01)

###########################################################################
# Weak Learner - Visualization
###########################################################################

# degrees_grid = [1, 2, 3, 4]

# for degree in degrees_grid:
#     param = 'weak_learner_order'
#     poly = PolynomialFeatures(degree)
#     fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6.0, 6.0))
#     for ax, cv_param in zip(axes.flatten(), degrees_grid):
#         X_poly = poly.fit_transform(X_train)
#         clf = RandomForestClassifier(**best_params_).fit(X_poly, y_train)
#         Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

#         # decision boundary line
#         ax.contour(xx, yy, Z, linewidths=0.8, colors='k')
#         # decision surfaces
#         ax.contourf(xx,
#                     yy,
#                     Z,
#                     cmap=plt.cm.jet.from_list(
#                         'contourf', [b_sns, g_sns, r_sns], 3),
#                     alpha=0.4)
#         ax.set_title('%s=%s' % (param, cv_param))
#         # color map
#         cmap = {0: y_sns, 1: b_sns, 2: g_sns, 3: r_sns}
#         # plot toy data
#         ax.scatter(data_train[:, 0], data_train[:, 1], c=list(
#             map(lambda l: cmap[l], data_train[:, 2])), alpha=0.4)
#         # plot test points
#         ax.scatter(test_points[:, 0], test_points[:, 1], c=list(
#             map(lambda l: cmap[l], clf.predict(test_points)),
#         ), edgecolors='k')

#         fig.savefig('assets/2/boundaries/%s.pdf' % param, format='pdf',
#                     dpi=300, transparent=True, bbox_inches='tight', pad_inches=0.01)
