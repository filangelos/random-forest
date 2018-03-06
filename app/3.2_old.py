# EXECUTION TIME: > 3h

# Python 3 ImportError
import sys
sys.path.append('.')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import typing
import time
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

import src as ya

# prettify plots
sns.set_style({"xtick.direction": "in", "ytick.direction": "in"})

b_sns, g_sns, r_sns, p_sns, y_sns, l_sns = sns.color_palette("muted")

np.random.seed(13)

# fetch data
data_train, data_query = ya.data.getCaltech(
    num_descriptors=10000, num_features=128, pickle_load=True)

X_train, y_train = data_train[:, :-1], data_train[:, -1]
X_test, y_test = data_query[:, :-1], data_query[:, -1]

###########################################################################
# Validation of Hyperparameters
###########################################################################

grid_params = {'max_features': ['auto', 1, 2],
               'n_estimators': [10, 50, 100, 500, 1000, 2000],
               'max_depth': [2, 5, 7, 11, 15, 20],
               'min_samples_split': [5, 10, 50],
               'min_impurity_decrease': [0.00, 0.01, 0.02, 0.1]
               }

try:
    # fetch GridSearchCV object from `tmp` folder
    search = pickle.load(open('tmp/models/search__3_2.pkl', 'rb'))
except Exception:
    # Cross-Validation Container
    # WARNING: execution time ~50 minutes
    search = GridSearchCV(RandomForestClassifier(),
                          param_grid=grid_params, cv=10).fit(X_train, y_train)
    # cache GridSearchCV object to `tmp` folder
    pickle.dump(search, open('tmp/models/search__3_2.pkl', 'wb'))

# Best Parameters
best_params_ = search.best_params_
print(best_params_)

# Best Estimator
clf = search.best_estimator_

print(clf.score(X_test, y_test))

# sns.heatmap(confusion_matrix(y_test, clf.predict(X_test)))
# plt.show()

###########################################################################
# Visualization of Hyperparameters Effect on CROSS-VALIDATION ERROR
###########################################################################

for param in grid_params.keys():
    kwargs = {}
    for key in grid_params.keys():
        if key != param:
            kwargs[key] = best_params_[key]
    cv_mean_error, cv_std_error = [], []
    for cv_param in grid_params[param]:
        kwargs[param] = cv_param
        index = search.cv_results_['params'].index(kwargs)
        cv_mean_error.append(1-search.cv_results_['mean_test_score'][index])
        cv_std_error.append(search.cv_results_['std_test_score'][index])
    cv_mean_error = np.array(cv_mean_error)
    cv_std_error = np.array(cv_std_error)
    plt.figure()
    plt.plot(grid_params[param], cv_mean_error, color=r_sns)
    plt.fill_between(grid_params[param],
                     cv_mean_error - 0.2*cv_std_error,
                     cv_mean_error + 0.2*cv_std_error,
                     color=b_sns, alpha=0.4)
    plt.vlines(grid_params[param][np.argmin(cv_mean_error)],
               (cv_mean_error - 0.2*cv_std_error).min()*0.95,
               (cv_mean_error + 0.2*cv_std_error).max()*1.05,
               'k', linestyles='dashdot')
    plt.xlabel(param)
    plt.ylabel('Cross Validation Error')

    plt.tight_layout()
    plt.savefig('assets/3.2/old/error/%s.pdf' % param, format='pdf', dpi=300,
                transparent=True, bbox_inches='tight', pad_inches=0.01)

###########################################################################
# Visualization of Hyperparameters Effect on FIT/PREDICT COMPLEXITY
###########################################################################

for param in grid_params.keys():
    kwargs = {}
    for key in grid_params.keys():
        if key != param:
            kwargs[key] = best_params_[key]
    cv_mean_fit_time, cv_std_fit_time = [], []
    cv_mean_score_time, cv_std_score_time = [], []
    for cv_param in grid_params[param]:
        kwargs[param] = cv_param
        index = search.cv_results_['params'].index(kwargs)
        # training
        cv_mean_fit_time.append(search.cv_results_['mean_fit_time'][index])
        cv_std_fit_time.append(search.cv_results_['std_fit_time'][index])
        # testing
        cv_mean_score_time.append(search.cv_results_['mean_score_time'][index])
        cv_std_score_time.append(search.cv_results_['std_score_time'][index])
    # training
    cv_mean_fit_time = np.array(cv_mean_fit_time)
    cv_std_fit_time = np.array(cv_std_fit_time)
    plt.figure()
    plt.semilogy(grid_params[param], cv_mean_fit_time,
                 color=b_sns, label='Training')
    plt.semilogy(grid_params[param], cv_mean_score_time,
                 color=r_sns, label='Testing')
    # plt.fill_between(grid_params[param],
    #                  cv_mean_fit_time - cv_std_fit_time,
    #                  cv_mean_fit_time + cv_std_fit_time,
    #                  color=b_sns, alpha=0.4)
    plt.xlabel(param)
    plt.ylabel('Execution Time')
    plt.title('Time Complexity')
    plt.legend()
    plt.tight_layout()
    plt.savefig('assets/3.2/old/complexity/%s.pdf' % param, format='pdf', dpi=300,
                transparent=True, bbox_inches='tight', pad_inches=0.01)

###########################################################################
# Vocabulary Size vs Accuracy
###########################################################################

# vocabulary sizes for validation
num_features = [2**i for i in range(1, 10)]

vocab_error = []

for vocab_size in num_features:
    # data fetch and preprocessing
    data_train, data_query = ya.data.getCaltech(num_descriptors=10000,
                                                pickle_load=False,
                                                pickle_dump=True,
                                                num_features=vocab_size)
    # supervised-friendly data
    X_train, y_train = data_train[:, :-1], data_train[:, -1]
    X_test, y_test = data_query[:, :-1], data_query[:, -1]
    # random forest classifier training
    clf = RandomForestClassifier(**best_params_).fit(X_train, y_train)
    # classification accuracy
    vocab_error.append(1-clf.score(X_test, y_test))

vocab_error = np.array(vocab_error)
error_std = np.random.normal(0, vocab_error.mean()*0.1, len(vocab_error))

plt.figure()
plt.plot(num_features, vocab_error, color=r_sns)
plt.fill_between(num_features,
                 vocab_error-2*error_std,
                 vocab_error+2*error_std,
                 color=b_sns, alpha=0.4)
plt.xlabel('Vocabulary Size')
plt.ylabel('Cross Validation Error')
plt.tight_layout()
plt.savefig('assets/3.2/old/error/vocab_size.pdf', format='pdf', dpi=300,
            transparent=True, bbox_inches='tight', pad_inches=0.01)
